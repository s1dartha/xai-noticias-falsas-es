"""
Fase 5: Índice de Manipulación Lingüística (IML) y estilos discursivos.

Construcción (deliberadamente sencilla)
---------------------------------------
1. Variables: las de la tabla única (construir_tabla_variables.py), agrupadas en las cinco dimensiones del IML:
     D1 Sensacionalismo (S) · D2 Carga emocional (E) · D3 Reiteración (R) · D4 Énfasis ortotipográfico (O)
     D5 Aserción sin respaldo (L)
   A cada variable se le asigna ANTES de mirar las etiquetas una dirección (+1 si más valor = estilo más manipulativo,
   −1 si menos valor = más manipulativo), justificada en la literatura. Las variables sin una dirección defendible
   (por ejemplo, la longitud o la dispersión) no entran al IML.
2. Escala: cada variable se convierte en su percentil dentro del corpus de referencia (0–100), invertido si la
   dirección es −1. Sin fórmulas ad hoc: un 80 significa «más marcado que el 80 % de las noticias del corpus».
3. Dimensión = percentil (dentro del corpus) de la media de los percentiles de sus variables, para que las cinco
   tengan la misma escala y peso. IML = media de las cinco dimensiones.
4. Estilos discursivos: k-medias sobre las cinco dimensiones (escala 0–100). El número de estilos se elige entre 3 y 6
   con el coeficiente de silueta, exigiendo estabilidad con remuestreo (índice de Rand ajustado >= 0,75).
   Cada noticia recibe un GRADO DE PERTENENCIA a cada estilo (inverso de la distancia al centro, normalizado), no
   una etiqueta rígida: una noticia puede ser 60 % de un estilo y 30 % de otro.
5. Los nombres de los estilos se ponen después, leyendo los centros: cada estilo recibe un nombre interpretativo
   (Institucional / Técnico, Opinión / Crónica, Alarmista / Viral) asignado por el PERFIL de su centro, no por el
   índice del grupo, y el script se detiene si los centros no cuadran con esos perfiles. Se conserva además una
   descripción automática del centro («perfil»: qué dimensiones están por encima o por debajo de la mediana).
   Los estilos describen la FORMA del texto, no su veracidad.

La etiqueta de veracidad no interviene en ningún paso; solo se usa al final para describir (no para construir).
Salidas: reportes/iml_estilos_discursivos_4418.csv, reportes/metricas_iml_estilos_discursivos.json,
codigos/referencia_iml_estilos.json (percentiles y centros para perfilar textos nuevos) y figuras en graficos/estilos/.
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, roc_auc_score

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config_proyecto.py").exists())))
import config_proyecto as cfg

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Dimensión -> (nombre, rasgo que describe, {variable: dirección})
DIMENSIONES = {
    'D1': ('Sensacionalismo', 'sensacionalista',
           {'sens_p_documento': +1, 'sens_p_media_oraciones': +1, 'sens_p_max_oraciones': +1}),
    'D2': ('Carga emocional', 'emotivo',
           {'emo_intensidad_media': +1, 'emo_intensidad_max': +1, 'emo_ira_media': +1, 'emo_ira_max': +1,
            'emo_miedo_media': +1, 'emo_miedo_max': +1, 'emo_sorpresa_media': +1, 'emo_sorpresa_max': +1}),
    'D3': ('Reiteración', 'reiterativo',
           {'red_sim_max': +1, 'red_sim_media': +1, 'red_sim_top3': +1, 'red_consecutivas_media': +1}),
    'D4': ('Énfasis ortotipográfico', 'enfático',
           {'orto_mayusculas_sostenidas': +1, 'orto_proporcion_mayusculas': +1, 'orto_exclamaciones': +1,
            'orto_interrogaciones': +1, 'orto_puntos_suspensivos': +1, 'orto_comillas': +1}),
    'D5': ('Aserción sin respaldo', 'asertivo sin fuentes',
           {'ling_intensificadores': +1, 'ling_pronombres_1p': +1, 'ling_adjetivos_sustantivos': +1,
            'ling_atenuadores': -1, 'ling_verbos_dicendi': -1, 'ling_cifras': -1, 'ling_referencias_temporales': -1}),
}
JUSTIFICACION = {
    'D1': 'Probabilidad de sensacionalismo del documento y de sus oraciones (JJNeila). El tono sensacionalista y la oración gatillo son rasgos '
          'recurrentes de la desinformación viral (Horne y Adalı, 2017; Hamby et al., 2023).',
    'D2': 'Intensidad emocional (1 − P(sin emoción)) y emociones de alta activación: ira, miedo y sorpresa (RoBERTuito). La desinformación explota '
          'emociones negativas de alta activación (Hamby et al., 2023). Tristeza y alegría quedan fuera por ser de baja activación o valencia positiva.',
    'D3': 'Similitud semántica entre oraciones (hiiamsid), continua y sin umbrales. La repetición de una misma idea refuerza su credibilidad '
          '(efecto de verdad ilusoria; Hasher et al., 1977) y el cuerpo de las noticias falsas es más repetitivo (Horne y Adalı, 2017).',
    'D4': 'Mayúsculas sostenidas, proporción de mayúsculas, exclamaciones, interrogaciones, puntos suspensivos y comillas por palabra. '
          'Las noticias falsas usan más mayúsculas y signos de puntuación enfáticos (Horne y Adalı, 2017).',
    'D5': 'Certeza sin respaldo: más intensificadores, pronombres de primera persona y adjetivación (+), y menos atenuadores, verbos de atribución '
          '(dijo, declaró), cifras y referencias temporales (−), que son los anclajes verificables del periodismo informativo (Zhou y Zafarani, 2020).',
}
EXCLUIDAS = {
    'sens_sd_oraciones': 'dispersión: no indica más ni menos sensacionalismo',
    'red_sim_varianza': 'dispersión de similitudes, sin dirección',
    'red_consecutivas_min': 'una oración inconexa puede ser tanto manipulación como estilo informativo',
    'emo_tristeza_media': 'emoción de baja activación', 'emo_tristeza_max': 'emoción de baja activación',
    'emo_alegria_media': 'valencia positiva', 'emo_alegria_max': 'valencia positiva',
    'ling_flesch_szigriszt': 'legibilidad: describe complejidad, no manipulación',
    'ling_guiraud': 'riqueza léxica: sin dirección clara en la literatura en español',
    'ling_hapax': 'riqueza léxica: sin dirección clara', 'ling_adverbios': 'sin dirección clara',
    'ling_pronombres_3p': 'sin dirección clara', 'ling_porcentajes': 'redundante con ling_cifras',
    'ling_palabras': 'longitud', 'ling_oraciones': 'longitud',
}
K_RANGO = range(2, 9)
K_MIN, K_MAX = 3, 6
N_ESTABILIDAD = 30
IMG = cfg.EXPLICABILIDAD_GRAFICOS_DIR / 'estilos'

# Nombres interpretativos de los estilos (en el orden en que se numeran). Se asignan por el perfil del centro
# (asignar_nombres), nunca por el índice del grupo de k-medias.
ESTILOS_NOMBRADOS = [
    {'nombre': 'Institucional / Técnico', 'ingles': 'Official / Technical Bulletin',
     'lema': 'Tono formal y factual, con ideas clave reiteradas y fuentes citadas.'},
    {'nombre': 'Opinión / Crónica', 'ingles': 'Opinion / Direct Narrative',
     'lema': 'Relato o comentario fluido, subjetivo pero con escritura estándar.'},
    {'nombre': 'Alarmista / Viral', 'ingles': 'Tabloid / Viral Alarm',
     'lema': 'Alto dramatismo, tono emocional y énfasis visual (mayúsculas y signos).'},
]
REGLA_NOMBRES = ('Alarmista / Viral = mayor media de D1, D2 y D4; de los otros dos, Institucional / Técnico = menor D1 + D2 y '
                 'menor D5 (más anclajes verificables); Opinión / Crónica = el restante, que además debe tener menor D3 (menos reiteración).')


# ---------------------------------------------------------------------------
# Funciones reutilizables (también para perfilar una noticia nueva)
# ---------------------------------------------------------------------------
def percentil(valor, referencia_ordenada):
    """Percentil (0–100) de `valor` dentro de la referencia ordenada (rango medio en los empates)."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return np.nan
    izq = np.searchsorted(referencia_ordenada, valor, side='left')
    der = np.searchsorted(referencia_ordenada, valor, side='right')
    return 100.0 * (izq + der) / (2 * len(referencia_ordenada))


def dimensiones_brutas(variables, referencia):
    """Media de los percentiles (con dirección) de las variables de cada dimensión."""
    out = {}
    for d, (_, _, vs) in DIMENSIONES.items():
        ps = []
        for v, sentido in vs.items():
            p = percentil(variables.get(v, np.nan), np.asarray(referencia['percentiles'][v]))
            if not np.isnan(p):
                ps.append(p if sentido > 0 else 100 - p)
        out[d] = float(np.mean(ps)) if ps else np.nan
    return out


def dimensiones_de(variables, referencia):
    """Puntajes D1–D5 (0–100): percentil de la media anterior dentro del corpus. Así las cinco dimensiones tienen la
    misma escala y el mismo peso (promediar percentiles poco correlacionados encoge la varianza hacia 50)."""
    brutas = dimensiones_brutas(variables, referencia)
    return {d: percentil(v, np.asarray(referencia['dimensiones_brutas'][d])) for d, v in brutas.items()}


def pertenencias(dims, centros):
    """Grado de pertenencia (%) a cada estilo: inverso de la distancia euclídea al centro, normalizado."""
    x = np.array([dims[d] for d in DIMENSIONES])
    dist = np.array([np.linalg.norm(x - np.array(c)) for c in centros])
    w = 1.0 / np.maximum(dist, 1e-6)
    return list(100 * w / w.sum())


def perfilar(variables, referencia=None):
    """Perfil completo de una noticia: dimensiones, IML, estilo dominante, secundario y pertenencias."""
    referencia = referencia or json.loads(cfg.REFERENCIA_ESTILOS_JSON.read_text(encoding='utf-8'))
    dims = dimensiones_de(variables, referencia)
    iml = float(np.nanmean(list(dims.values())))
    dims_imp = {d: (v if not np.isnan(v) else 50.0) for d, v in dims.items()}
    per = pertenencias(dims_imp, referencia['centros'])
    orden = np.argsort(per)[::-1]
    nombres = referencia['nombres_estilos']
    return {'dimensiones': {d: round(v, 1) for d, v in dims.items()}, 'iml': round(iml, 1),
            'estilo': nombres[orden[0]], 'lema_estilo': referencia['lemas_estilos'][orden[0]],
            'indice_estilo': int(orden[0]), 'pertenencia_estilo': round(per[orden[0]], 1),
            'estilo_secundario': nombres[orden[1]], 'pertenencia_secundario': round(per[orden[1]], 1),
            'pertenencias': {nombres[i]: round(per[i], 1) for i in range(len(nombres))}}


SUSTANTIVO = {'D1': 'sensacionalismo', 'D2': 'carga emocional', 'D3': 'reiteración', 'D4': 'énfasis ortotipográfico', 'D5': 'aserción sin fuentes'}


def _enumerar(xs, conj):
    return xs[0] if len(xs) == 1 else f"{', '.join(xs[:-1])} {conj} {xs[-1]}"


def nombrar(centro, mediana=50.0, margen=10.0):
    """Nombre descriptivo a partir del centro: rasgos claramente por encima (>= 60) y por debajo (<= 40) de la mediana."""
    altos = [DIMENSIONES[d][1] for d, v in zip(DIMENSIONES, centro) if v >= mediana + margen]
    bajos = [SUSTANTIVO[d] for d, v in zip(DIMENSIONES, centro) if v <= mediana - margen]
    nombre = _enumerar(altos, 'y') if altos else ('Sobrio' if len(bajos) >= 3 else 'Moderado')
    if bajos and not (not altos and len(bajos) >= 3):
        nombre += f", sin {_enumerar(bajos, 'ni')}"
    elif not altos and len(bajos) >= 3:
        nombre += f" (sin {_enumerar(bajos, 'ni')})"
    return nombre[0].upper() + nombre[1:]


def asignar_nombres(centros):
    """Orden de los centros según ESTILOS_NOMBRADOS (Institucional, Opinión, Alarmista), decidido por su perfil.
    Falla si los centros no cuadran con los perfiles que justifican los nombres."""
    c = np.asarray(centros)
    if len(c) != len(ESTILOS_NOMBRADOS):
        raise ValueError(f"Los nombres están definidos para {len(ESTILOS_NOMBRADOS)} estilos y k-medias eligió {len(c)}: "
                         "revise ESTILOS_NOMBRADOS antes de continuar.")
    D1, D2, D3, D4, D5 = c.T
    alarmista = int(np.argmax((D1 + D2 + D4) / 3))
    resto = [i for i in range(len(c)) if i != alarmista]
    inst = min(resto, key=lambda i: D1[i] + D2[i])
    opin = next(i for i in resto if i != inst)
    problemas = []
    if not D5[inst] < D5[opin]:
        problemas.append(f"el candidato a Institucional no tiene menor D5 ({D5[inst]:.1f} frente a {D5[opin]:.1f})")
    if not D3[opin] < D3[inst]:
        problemas.append(f"el candidato a Opinión no tiene menor D3 ({D3[opin]:.1f} frente a {D3[inst]:.1f})")
    if not all(c[alarmista, j] >= c[opin, j] for j in (0, 1, 3)):
        problemas.append("el candidato a Alarmista no supera a Opinión en D1, D2 y D4")
    if problemas:
        raise ValueError("Los centros no cuadran con los nombres de ESTILOS_NOMBRADOS: " + "; ".join(problemas) +
                         f". Centros (D1–D5): {np.round(c, 1).tolist()}")
    return [inst, opin, alarmista]


# ---------------------------------------------------------------------------
# Proyecciones y figuras
# ---------------------------------------------------------------------------
N_VECINOS = 15


def proyecciones(X, idx_estilo, y, centros):
    """PCA, t-SNE y UMAP de las noticias en el espacio D1–D5, y cuánto comparten estilo o clase los vecinos cercanos."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.neighbors import NearestNeighbors
    import umap
    pca = PCA(n_components=2, random_state=cfg.SEED).fit(X)
    emb = {'PCA': pca.transform(X),
           't-SNE': TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=cfg.SEED).fit_transform(X),
           'UMAP': umap.UMAP(n_neighbors=N_VECINOS, min_dist=0.1, random_state=cfg.SEED).fit_transform(X),
           'centros_PCA': pca.transform(centros)}
    # Vecindad en el espacio original (5 dimensiones), sin proyectar
    vec = NearestNeighbors(n_neighbors=N_VECINOS + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    p_est = np.bincount(idx_estilo) / len(idx_estilo)
    p_f = y.mean()
    resumen = {
        'pca': {'varianza_explicada': [round(float(v), 4) for v in pca.explained_variance_ratio_],
                'cargas': {f'PC{i + 1}': dict(zip(DIMENSIONES, np.round(pca.components_[i], 3).tolist())) for i in range(2)}},
        'tsne': {'perplejidad': 30, 'inicializacion': 'pca'},
        'umap': {'n_vecinos': N_VECINOS, 'distancia_minima': 0.1},
        'vecindad': {'k': N_VECINOS,
                     'mismo_estilo': float((idx_estilo[vec] == idx_estilo[:, None]).mean()),
                     'mismo_estilo_azar': float((p_est ** 2).sum()),
                     'misma_clase': float((y[vec] == y[:, None]).mean()),
                     'misma_clase_azar': float(p_f ** 2 + (1 - p_f) ** 2)},
    }
    return resumen, emb


def dibujar_radar(ax, series, titulo=None, tam_etiquetas=8.5):
    """Radar (pentágono) D1–D5 en escala 0–100. `series`: lista de dict(valores, color, etiqueta, estilo_linea, relleno)."""
    import graficos_comunes as gc
    ang = np.linspace(0, 2 * np.pi, len(DIMENSIONES), endpoint=False)
    cerrar = lambda v: np.concatenate([v, v[:1]])
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25', '50', '75', '100'], fontsize=7, color=gc.TINTA_SUAVE)
    ax.set_rlabel_position(180 / len(DIMENSIONES))
    ax.set_xticks(ang)
    ax.set_xticklabels([f"{d}\n{DIMENSIONES[d][0].replace(' ', chr(10), 1)}" for d in DIMENSIONES], fontsize=tam_etiquetas, color=gc.TINTA)
    ax.tick_params(axis='x', pad=16)
    ax.grid(color=gc.REJILLA, linewidth=0.7)
    ax.spines['polar'].set_color(gc.EJE)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(th, np.full_like(th, 50), color=gc.TINTA_SUAVE, linewidth=0.9, linestyle=':', zorder=1)   # mediana del corpus
    for s in series:
        v = cerrar(np.asarray(s['valores'], float))
        ax.plot(cerrar(ang), v, color=s['color'], linewidth=2, linestyle=s.get('estilo_linea', '-'), label=s.get('etiqueta'), zorder=3)
        ax.scatter(ang, v[:-1], s=22, color=s['color'], edgecolor='white', linewidth=1, zorder=4)
        if s.get('relleno', True):
            ax.fill(cerrar(ang), v, color=s['color'], alpha=0.10, zorder=2)
    if titulo:
        ax.set_title(titulo, fontsize=9.5, color=gc.TINTA, loc='center', pad=18)


def generar_figuras(desc, eleccion, k_opt, centros, nombres, u, idx_estilo, y, emb):
    import graficos_comunes as gc
    plt = gc.pyplot()
    IMG.mkdir(parents=True, exist_ok=True)
    col_est = [gc.ESTILOS[i + 1] for i in range(k_opt)]
    corto = [n.split(': ', 1)[1] for n in nombres]
    figuras = {}

    # Elección de k
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ks = list(K_RANGO)
    for ax, clave, tit in [(axes[0], 'silueta', 'Coeficiente de silueta'), (axes[1], 'estabilidad_ari_media', 'Estabilidad (ARI medio, 30 remuestreos)')]:
        vals = [eleccion[k][clave] for k in ks]
        ax.plot(ks, vals, color=gc.SERIE[0], marker='o', markersize=6)
        ax.scatter([k_opt], [eleccion[k_opt][clave]], s=110, color=gc.SERIE[0], edgecolor='white', linewidth=2, zorder=3)
        ax.annotate(f'k = {k_opt}', (k_opt, eleccion[k_opt][clave]), xytext=(8, 8), textcoords='offset points', fontsize=9, color=gc.TINTA)
        if clave == 'estabilidad_ari_media':
            ax.axhline(0.75, color=gc.EJE, linewidth=0.9, linestyle=':')
            ax.text(ks[-1], 0.755, 'umbral 0,75', ha='right', va='bottom', fontsize=8, color=gc.TINTA_SUAVE)
        ax.set_xticks(ks)
        ax.set_xlabel('Número de estilos (k)')
        ax.set_title(tit)
        gc.coma(ax, 'y', 2)
    fig.tight_layout()
    figuras['eleccion_k'] = gc.guardar(fig, IMG / 'eleccion_numero_estilos.png')

    # Mapa de calor de los centros
    fig, ax = plt.subplots(figsize=(8.2, 0.8 * k_opt + 2.0))
    im = ax.imshow(centros, cmap=gc.secuencial(), vmin=0, vmax=100, aspect='auto')
    ax.grid(False)
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"{d}\n" + DIMENSIONES[d][0].replace(' ', '\n', 1) for d in DIMENSIONES], fontsize=8.5, color=gc.TINTA)
    ax.set_yticks(range(k_opt))
    ax.set_yticklabels([f"{n.split(':')[0]}: {c}\n(n = {e['n']})" for n, c, e in zip(nombres, corto, desc['estilos'])], fontsize=9, color=gc.TINTA)
    for i in range(k_opt):
        for j in range(5):
            v = centros[i, j]
            ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=9.5, color='white' if v > 60 else gc.TINTA)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.ax.tick_params(labelsize=8)
    cb.set_label('Percentil (0–100)', fontsize=8.5, color=gc.TINTA_2)
    ax.set_title('Centro de cada estilo en las cinco dimensiones del IML (percentil; 50 = mediana del corpus)')
    fig.tight_layout()
    figuras['centros'] = gc.guardar(fig, IMG / 'centros_estilos_discursivos.png')

    # Radar de los centros
    fig = plt.figure(figsize=(7.2, 6.4))
    ax = fig.add_subplot(projection='polar')
    dibujar_radar(ax, [{'valores': centros[i], 'color': col_est[i], 'etiqueta': f"{nombres[i]} (n = {desc['estilos'][i]['n']})"} for i in range(k_opt)],
                  tam_etiquetas=9)
    fig.legend(loc='lower center', ncol=1, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Perfil de los estilos discursivos en las cinco dimensiones del IML\n(percentil del corpus; la línea punteada es la mediana, 50)',
                 fontsize=10.5, color=gc.TINTA, x=0.02, ha='left')
    fig.subplots_adjust(top=0.82, bottom=0.2)
    figuras['radar'] = gc.guardar(fig, IMG / 'radar_estilos_discursivos.png')

    # Proyecciones 2D por estilo y por clase
    rng = np.random.default_rng(cfg.SEED)
    orden = rng.permutation(len(idx_estilo))   # orden aleatorio de dibujo: ningún grupo tapa sistemáticamente a otro
    var = desc['proyecciones']['pca']['varianza_explicada']
    titulos = {'PCA': f"PCA (varianza explicada: PC1 {var[0] * 100:.0f} %, PC2 {var[1] * 100:.0f} %)".replace('.', ','),
               't-SNE': 't-SNE (perplejidad 30)', 'UMAP': f'UMAP ({N_VECINOS} vecinos)'}
    for clave, colores, etiquetas, codigo in [
            ('proyecciones_estilo', col_est, [f"{nombres[i]}" for i in range(k_opt)], idx_estilo),
            ('proyecciones_clase', [gc.CLASE['REAL'], gc.CLASE['FAKE']], ['Real', 'Falsa'], y)]:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.9))
        for ax, m in zip(axes, ['PCA', 't-SNE', 'UMAP']):
            P = emb[m][orden]
            c = np.array(colores)[codigo[orden]]
            ax.scatter(P[:, 0], P[:, 1], s=5, c=c, alpha=0.55, linewidths=0)
            if m == 'PCA' and clave == 'proyecciones_estilo':
                C = emb['centros_PCA']
                ax.scatter(C[:, 0], C[:, 1], s=170, c=col_est, edgecolor='white', linewidth=2.2, marker='o', zorder=4)
                for i in range(k_opt):
                    ax.annotate(f"Estilo {i + 1}", C[i], xytext=(9, 7), textcoords='offset points', fontsize=9, color=gc.TINTA, weight='bold')
            ax.set_title(titulos[m], fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ['left', 'bottom']:
                ax.spines[s].set_color(gc.EJE)
        from matplotlib.lines import Line2D
        fig.legend(handles=[Line2D([], [], marker='o', linestyle='', markersize=8, color=col, label=et) for col, et in zip(colores, etiquetas)],
                   loc='lower center', ncol=len(etiquetas), fontsize=9.5, bbox_to_anchor=(0.5, -0.01))
        sub = 'estilo discursivo dominante' if clave == 'proyecciones_estilo' else 'clase (real o falsa)'
        fig.suptitle(f"Noticias del corpus ({len(idx_estilo)} textos únicos) proyectadas desde las cinco dimensiones del IML, coloreadas por {sub}",
                     fontsize=10.5, color=gc.TINTA, x=0.01, ha='left')
        fig.tight_layout(rect=(0, 0.07, 1, 0.95))
        figuras[clave] = gc.guardar(fig, IMG / f"{clave}.png")

    # Distribución del IML por clase
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    bins = np.linspace(0, 100, 41)
    for clase, nombre in [(0, 'Reales'), (1, 'Falsas')]:
        ax.hist(u.loc[u['label_num'] == clase, 'IML'], bins=bins, color=gc.CLASE[clase], alpha=0.55, label=nombre, edgecolor='white', linewidth=0.5)
    ax.set_xlabel('IML (0–100)')
    ax.set_ylabel('Noticias')
    ax.set_title(f"Distribución del IML por clase (AUC = {desc['asociacion_veracidad']['IML']['auc_falsa']:.3f})".replace('.', ','))
    ax.legend(fontsize=9)
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    figuras['iml_clase'] = gc.guardar(fig, IMG / 'distribucion_iml_por_clase.png')

    # Tamaño de cada estilo y proporción de falsas
    fig, ax = plt.subplots(figsize=(8.8, 0.6 * k_opt + 1.5))
    for i, e in enumerate(desc['estilos']):
        ax.barh(i, e['proporcion'] * 100, color=col_est[i], height=0.6, edgecolor='white')
        ax.text(e['proporcion'] * 100 + 0.5, i, f"{e['proporcion'] * 100:.1f} % · {e['tasa_falsas'] * 100:.0f} % falsas".replace('.', ','),
                va='center', fontsize=8.5, color=gc.TINTA)
    ax.set_yticks(range(k_opt))
    ax.set_yticklabels(nombres, fontsize=9, color=gc.TINTA)
    ax.invert_yaxis()
    ax.set_xlabel('% de las noticias del corpus')
    ax.set_title(f"Tamaño de cada estilo y proporción de falsas (global: {desc['tasa_falsas_global'] * 100:.0f} %)")
    ax.grid(axis='y', visible=False)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(e['proporcion'] for e in desc['estilos']) * 100 * 1.45)
    fig.tight_layout()
    figuras['tamanos'] = gc.guardar(fig, IMG / 'tamano_estilos_y_tasa_falsas.png')
    return figuras


# ---------------------------------------------------------------------------
# Construcción sobre el corpus
# ---------------------------------------------------------------------------
def main():
    tabla = pd.read_csv(cfg.TABLA_VARIABLES_CSV)
    meta_tabla = json.loads(cfg.TABLA_VARIABLES_META_JSON.read_text(encoding='utf-8'))
    todas = [v for g in meta_tabla['grupos'].values() for v in g]
    usadas = [v for _, _, vs in DIMENSIONES.values() for v in vs]
    assert set(usadas) | set(EXCLUIDAS) == set(todas), set(todas) ^ (set(usadas) | set(EXCLUIDAS))
    unicos = tabla.drop_duplicates('id_texto').reset_index(drop=True)
    print(f"Referencia: {len(unicos)} textos únicos | variables en el IML: {len(usadas)} de {len(todas)}", flush=True)

    referencia = {'percentiles': {v: np.sort(unicos[v].dropna().values).tolist() for v in usadas},
                  'dimensiones': {d: {'nombre': n, 'rasgo': r, 'variables': vs} for d, (n, r, vs) in DIMENSIONES.items()}}
    B = pd.DataFrame([dimensiones_brutas(row, referencia) for row in unicos[usadas].to_dict('records')])
    referencia['dimensiones_brutas'] = {d: np.sort(B[d].dropna().values).tolist() for d in DIMENSIONES}
    D = pd.DataFrame([dimensiones_de(row, referencia) for row in unicos[usadas].to_dict('records')])
    D_imp = D.fillna(50.0)   # solo D3 puede faltar (noticias de una oración): se usa el valor neutro
    X = D_imp.values

    print("Eligiendo el número de estilos (silueta y estabilidad)...", flush=True)
    rng = np.random.default_rng(cfg.SEED)
    eleccion = {}
    for k in K_RANGO:
        km = KMeans(n_clusters=k, n_init=20, random_state=cfg.SEED).fit(X)
        sil = silhouette_score(X, km.labels_, sample_size=3000, random_state=cfg.SEED)
        aris = []
        for _ in range(N_ESTABILIDAD):
            i = rng.choice(len(X), len(X), replace=True)
            kb = KMeans(n_clusters=k, n_init=5, random_state=int(rng.integers(1e6))).fit(X[i])
            aris.append(adjusted_rand_score(km.labels_, kb.predict(X)))
        eleccion[k] = {'silueta': float(sil), 'estabilidad_ari_media': float(np.mean(aris)), 'estabilidad_ari_p5': float(np.percentile(aris, 5)),
                       'inercia': float(km.inertia_)}
        print(f"  k={k}: silueta={sil:.3f} | estabilidad ARI={np.mean(aris):.3f}", flush=True)
    # Regla: k entre K_MIN y K_MAX (k = 2 solo separa «más» de «menos» a lo largo de un eje y no describe combinaciones
    # de rasgos); dentro de ese rango, mayor silueta entre los k estables (ARI medio >= 0,75)
    candidatos = [k for k in K_RANGO if K_MIN <= k <= K_MAX]
    estables = [k for k in candidatos if eleccion[k]['estabilidad_ari_media'] >= 0.75]
    k_opt = max(estables or candidatos, key=lambda k: eleccion[k]['silueta'])
    km = KMeans(n_clusters=k_opt, n_init=50, random_state=cfg.SEED).fit(X)
    orden = asignar_nombres(km.cluster_centers_)   # Institucional, Opinión, Alarmista según el perfil del centro
    centros = km.cluster_centers_[orden]
    perfiles = [nombrar(c) for c in centros]          # descripción automática del centro (justifica el nombre)
    nombres = [f"Estilo {i + 1}: {e['nombre']}" for i, e in enumerate(ESTILOS_NOMBRADOS)]
    lemas = [e['lema'] for e in ESTILOS_NOMBRADOS]
    referencia.update({'k': int(k_opt), 'centros': centros.tolist(), 'nombres_estilos': nombres, 'lemas_estilos': lemas,
                       'nombres_estilos_en': [e['ingles'] for e in ESTILOS_NOMBRADOS], 'perfiles_estilos': perfiles})
    print(f"k elegido = {k_opt}: {nombres} | perfiles: {perfiles}", flush=True)

    # Perfil de todas las noticias (incluidos duplicados)
    filas = [perfilar(r, referencia) for r in tabla[usadas].to_dict('records')]
    out = tabla[['id_texto', 'label_num', 'region', 'Fuente']].copy()
    for d in DIMENSIONES:
        out[d] = [f['dimensiones'][d] for f in filas]
    out['IML'] = [f['iml'] for f in filas]
    out['estilo'] = [f['estilo'] for f in filas]
    out['pertenencia_estilo'] = [f['pertenencia_estilo'] for f in filas]
    out['estilo_secundario'] = [f['estilo_secundario'] for f in filas]
    out['pertenencia_secundario'] = [f['pertenencia_secundario'] for f in filas]
    for n in nombres:
        out[f'pertenencia__{n}'] = [f['pertenencias'][n] for f in filas]
    out.to_csv(cfg.ESTILOS_DISCURSIVOS_CSV, index=False)
    cfg.REFERENCIA_ESTILOS_JSON.write_text(json.dumps(referencia, ensure_ascii=False), encoding='utf-8')

    # ---------------- Descripción (la etiqueta solo se usa aquí) ----------------
    u = out.drop_duplicates('id_texto')
    y = u['label_num'].values
    desc = {
        'n_noticias': int(len(out)), 'n_textos_unicos': int(len(u)),
        'variables_usadas': usadas, 'variables_excluidas': EXCLUIDAS,
        'dimensiones': {d: {'nombre': n, 'rasgo': r, 'variables': vs, 'justificacion': JUSTIFICACION[d]} for d, (n, r, vs) in DIMENSIONES.items()},
        'eleccion_k': {str(k): v for k, v in eleccion.items()}, 'k_elegido': int(k_opt),
        'regla_k': f'k entre {K_MIN} y {K_MAX}; mayor silueta entre los k con estabilidad ARI media >= 0,75 (30 remuestreos bootstrap)',
        'regla_nombres': REGLA_NOMBRES,
        'estilos': [], 'correlacion_dimensiones_spearman': u[list(DIMENSIONES)].corr('spearman').round(3).to_dict(),
        'direccion_empirica': {}, 'asociacion_veracidad': {},
    }
    for i, n in enumerate(nombres):
        m = (u['estilo'] == n).values
        desc['estilos'].append({
            'nombre': n, 'lema': lemas[i], 'nombre_en': ESTILOS_NOMBRADOS[i]['ingles'], 'perfil': perfiles[i], 'centro': dict(zip(DIMENSIONES, np.round(centros[i], 1).tolist())), 'iml_centro': float(np.round(centros[i].mean(), 1)),
            'n': int(m.sum()), 'proporcion': float(m.mean()),
            'pertenencia_media': float(u.loc[m, 'pertenencia_estilo'].mean()),
            'falsas': int(y[m].sum()), 'tasa_falsas': float(y[m].mean()) if m.any() else None,
            'por_region': u[m]['region'].value_counts().to_dict(),
        })
    for v in usadas:
        x = unicos[v].values
        ok = ~np.isnan(x)
        auc = roc_auc_score(unicos['label_num'].values[ok], x[ok])
        sentido = next(vs[v] for _, _, vs in DIMENSIONES.values() if v in vs)
        desc['direccion_empirica'][v] = {'direccion_a_priori': sentido, 'auc_falsa': float(auc),
                                         'coincide': bool((auc - 0.5) * sentido > 0)}
    for c in list(DIMENSIONES) + ['IML']:
        desc['asociacion_veracidad'][c] = {'auc_falsa': float(roc_auc_score(y, u[c].fillna(50).values)),
                                           'media_falsas': float(u.loc[y == 1, c].mean()), 'media_reales': float(u.loc[y == 0, c].mean())}
    desc['tasa_falsas_global'] = float(y.mean())

    # ---------------- Proyecciones 2D y vecindad ----------------
    print("Proyecciones 2D (PCA, t-SNE, UMAP)...", flush=True)
    idx_estilo = np.array([nombres.index(e) for e in u['estilo']])
    desc['proyecciones'], emb = proyecciones(X, idx_estilo, y, centros)
    p = desc['proyecciones']
    print(f"  PCA: varianza explicada {p['pca']['varianza_explicada']} | vecinos del mismo estilo "
          f"{p['vecindad']['mismo_estilo']:.3f} (azar {p['vecindad']['mismo_estilo_azar']:.3f}) | misma clase "
          f"{p['vecindad']['misma_clase']:.3f} (azar {p['vecindad']['misma_clase_azar']:.3f})", flush=True)

    # ---------------- Figuras ----------------
    figuras = generar_figuras(desc, eleccion, k_opt, centros, nombres, u, idx_estilo, y, emb)

    desc['figuras'] = {k: str(v.relative_to(cfg.EXPLICABILIDAD_DIR)).replace('\\', '/') for k, v in figuras.items()}
    cfg.METRICAS_ESTILOS_DISCURSIVOS_JSON.write_text(json.dumps(desc, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    print(f"[OK] {cfg.ESTILOS_DISCURSIVOS_CSV}\n[OK] {cfg.METRICAS_ESTILOS_DISCURSIVOS_JSON}")
    for e in desc['estilos']:
        print(f"  {e['nombre']:<55} n={e['n']:>4} ({e['proporcion'] * 100:4.1f} %) falsas={e['tasa_falsas'] * 100:4.1f} % centro={e['centro']}")
    print("  AUC (descriptivo):", {k: round(v['auc_falsa'], 3) for k, v in desc['asociacion_veracidad'].items()})


if __name__ == '__main__':
    main()
