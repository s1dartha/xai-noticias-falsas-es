"""
Genera reportes/Reporte_Evaluacion_Modelos_Sensacionalismo.md a partir de reportes/metricas_modelos_sensacionalismo.json.
Todas las cifras del reporte se leen del JSON; no hay valores escritos a mano.

Regla de selección (fijada antes de mirar los resultados de prueba): se elige el modelo con mayor F1 macro medio,
con su umbral calibrado, en los dos conjuntos de prueba que miden sensacionalismo (Salud y Amarillismo humano).
TA1C mide clickbait, otro constructo, y se reporta aparte.
"""

import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config_proyecto.py").exists())))
import config_proyecto as cfg
import graficos_comunes as gc

IMG = cfg.SENSACIONALISMO_REPORTES_DIR / "Imagenes"
NOMBRE_MODELO = {'jjneila': 'JJNeila (sensacionalismo)', 'ggcastle_beto_clickbait': 'BETO clickbait (ggcastle)',
                 'dcere_ta1c_base': 'TA1C base (dcere)', 'dcere_ta1c_large': 'TA1C large (dcere)',
                 'christinacdl_xlmr_clickbait': 'XLM-R clickbait (christinacdl)'}
CONJUNTOS_FIG = [('salud_prueba', 'Salud (prueba)'), ('amarillismo_humano_cuerpo', 'Amarillismo humano'),
                 ('ta1c_test_cuerpo', 'TA1C test (clickbait)')]
CONJUNTOS_SENSACIONALISMO = ['salud_prueba', 'amarillismo_humano_cuerpo']
CONJUNTO_CLICKBAIT = 'ta1c_test_cuerpo'
NOMBRES = {
    'ta1c_test_titular': 'TA1C test (titular)', 'ta1c_test_cuerpo': 'TA1C test (titular + cuerpo)',
    'salud_prueba': 'Salud (mitad de prueba)', 'amarillismo_humano_titular': 'Amarillismo humano (titular)',
    'amarillismo_humano_cuerpo': 'Amarillismo humano (titular + copete + cuerpo)',
    'ia_sintetico': 'IA sintético (control)', 'noticia_clickbait': 'NoticIA (solo clickbait)',
}


def f(x, d=3):
    return '—' if x is None else f"{x:.{d}f}".replace('.', ',')


def ic(v):
    return f"[{f(v[0])}; {f(v[1])}]" if v else '—'


def probabilidades(modelos):
    """Etiquetas y P(positiva) por modelo y conjunto a partir de la caché de evaluar_modelos_sensacionalismo.py (sin inferencia)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import evaluar_modelos_sensacionalismo as ev
    conjuntos = ev.cargar_conjuntos()
    y = {n: d['y'] for n, d in conjuntos.items()}
    P = {}
    for m in ev.MODELOS:
        if m['clave'] not in modelos:
            continue
        meta = json.loads((ev.CACHE / f"{m['clave']}__meta.json").read_text(encoding='utf-8'))
        crudas = {n: np.load(ev.CACHE / f"{m['clave']}__{n}.npy") for n in conjuntos}
        P[m['clave']], _ = ev.prob_positiva(m, crudas, meta, conjuntos)
        # Comprobación: la caché reproduce el F1 macro del JSON
        from sklearn.metrics import f1_score
        tau = modelos[m['clave']]['umbral_calibrado']
        f1 = f1_score(y['salud_prueba'], (P[m['clave']]['salud_prueba'] >= tau).astype(int), average='macro', zero_division=0)
        assert abs(f1 - modelos[m['clave']]['conjuntos']['salud_prueba']['umbral_calibrado']['f1_macro']) < 1e-9, m['clave']
    return y, P


def figuras(modelos, ranking, mejor, criterio):
    from sklearn.metrics import roc_curve, f1_score
    plt = gc.pyplot()
    y, P = probabilidades(modelos)
    color = {k: gc.SERIE[i] for i, k in enumerate(modelos)}
    out = {}

    # 1. F1 macro (umbral calibrado) con IC por conjunto
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
    for ax, (n, nombre) in zip(axs, CONJUNTOS_FIG):
        orden = ranking[::-1]
        for i, k in enumerate(orden):
            c = modelos[k]['conjuntos'][n]['umbral_calibrado']
            lo, hi = c['f1_macro_ic95']
            ax.barh(i, c['f1_macro'], color=color[k], height=0.62)
            ax.errorbar(c['f1_macro'], i, xerr=[[c['f1_macro'] - lo], [hi - c['f1_macro']]], color=gc.TINTA_2, capsize=3, lw=1)
        ax.axvline(criterio, color=gc.TINTA_2, ls='--', lw=1)
        ax.axvline(0.5, color=gc.TINTA_SUAVE, ls=':', lw=1)
        ax.set_yticks(range(len(orden)))
        ax.set_yticklabels([NOMBRE_MODELO[k] for k in orden] if ax is axs[0] else [])
        ax.set_xlim(0, 1)
        ax.set_title(nombre)
        ax.set_xlabel('F1 macro (umbral calibrado)')
        ax.grid(axis='y', visible=False)
        gc.coma(ax, 'x', 1)
    axs[-1].text(criterio + 0.01, len(ranking) - 0.4, f"criterio {criterio:.2f}".replace('.', ','), fontsize=8, color=gc.TINTA_2)
    fig.suptitle('F1 macro con IC 95 % (bootstrap) por modelo y conjunto de prueba', x=0.01, ha='left', fontsize=11)
    fig.tight_layout()
    out['f1'] = gc.guardar(fig, IMG / 'f1_macro_por_modelo_y_conjunto.png')

    # 2. Curvas ROC
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (n, nombre) in zip(axs, CONJUNTOS_FIG):
        for k in modelos:
            fpr, tpr, _ = roc_curve(y[n], P[k][n])
            auc = modelos[k]['conjuntos'][n]['umbral_calibrado']['roc_auc']
            ax.plot(fpr, tpr, color=color[k], lw=2.4 if k == mejor else 1.4,
                    label=f"{NOMBRE_MODELO[k]} ({auc:.2f})".replace('.', ','))
        ax.plot([0, 1], [0, 1], color=gc.TINTA_SUAVE, ls=':', lw=1)
        ax.set_title(nombre)
        ax.set_xlabel('Tasa de falsos positivos')
        ax.set_aspect('equal')
        ax.legend(fontsize=7, loc='lower right', title='Modelo (AUC)', title_fontsize=7)
        gc.coma(ax, 'x', 1); gc.coma(ax, 'y', 1)
    axs[0].set_ylabel('Tasa de verdaderos positivos')
    fig.tight_layout()
    out['roc'] = gc.guardar(fig, IMG / 'curvas_roc_por_conjunto.png')

    # 3. Distribución de P(sensacionalista) del modelo elegido por clase
    tau = modelos[mejor]['umbral_calibrado']
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True)
    bins = np.linspace(0, 1, 21)
    for ax, (n, nombre) in zip(axs, CONJUNTOS_FIG):
        for clase, et, col in [(0, 'No sensacionalista', gc.SERIE[0]), (1, 'Sensacionalista', gc.SERIE[1])]:
            v = P[mejor][n][y[n] == clase]
            ax.hist(v, bins=bins, color=col, alpha=0.55, label=f"{et} (n = {len(v)})", weights=np.ones(len(v)) / len(v))
        ax.axvline(tau, color=gc.TINTA, ls='--', lw=1.2)
        ax.axvline(0.5, color=gc.TINTA_SUAVE, ls=':', lw=1)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
        ax.text(tau - 0.02, ax.get_ylim()[1] * 0.95, f"τ = {tau:.2f}".replace('.', ','), ha='right', va='top', fontsize=8, color=gc.TINTA)
        ax.set_title(f"{nombre} (n = {int((y[n] == 0).sum())} / {int((y[n] == 1).sum())})")
        ax.set_xlabel(f'P(sensacionalista) según {NOMBRE_MODELO[mejor].split(" (")[0]}')
        gc.coma(ax, 'x', 1); gc.coma(ax, 'y', 1)
    axs[0].set_ylabel('Proporción de textos de la clase')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, ['No sensacionalista', 'Sensacionalista'], loc='lower center', ncol=2, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out['dist'] = gc.guardar(fig, IMG / 'distribucion_probabilidad_modelo_elegido.png')

    # 4. Matrices de confusión del modelo elegido
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.6))
    for ax, (n, nombre) in zip(axs, CONJUNTOS_FIG):
        cm = np.array(modelos[mejor]['conjuntos'][n]['umbral_calibrado']['cm'])
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap=gc.secuencial(), vmin=0, vmax=1)
        for i in range(2):
            for jj in range(2):
                ax.text(jj, i, f"{cm[i, jj]}\n{100 * norm[i, jj]:.0f} %", ha='center', va='center', fontsize=9,
                        color='white' if norm[i, jj] > 0.6 else gc.TINTA)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['No sens.', 'Sens.']); ax.set_yticklabels(['No sens.', 'Sens.'])
        ax.set_xlabel('Predicción'); ax.grid(False)
        ax.set_title(nombre)
    axs[0].set_ylabel('Etiqueta real')
    fig.suptitle(f"{NOMBRE_MODELO[mejor]} con τ = {tau:.2f} (porcentaje por fila)".replace('.', ','), x=0.01, ha='left', fontsize=11)
    fig.tight_layout()
    out['cm'] = gc.guardar(fig, IMG / 'matrices_confusion_modelo_elegido.png')

    # 5. Efecto del umbral: F1 macro del modelo elegido en función de τ
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    fig, ax = plt.subplots(figsize=(8, 4))
    curvas = [('ta1c_validation_cuerpo', 'TA1C validation (calibración)', gc.SERIE[6], ':'),
              ('salud_calibracion', 'Salud (calibración)', gc.SERIE[3], ':'),
              ('salud_prueba', 'Salud (prueba)', gc.SERIE[0], '-'),
              ('amarillismo_humano_cuerpo', 'Amarillismo humano (prueba)', gc.SERIE[1], '-')]
    for n, et, col, ls in curvas:
        ax.plot(grid, [f1_score(y[n], (P[mejor][n] >= t).astype(int), average='macro', zero_division=0) for t in grid],
                color=col, ls=ls, label=et)
    ax.axvline(tau, color=gc.TINTA, ls='--', lw=1.2)
    ax.text(tau - 0.01, 0.05, f"τ calibrado = {tau:.2f}".replace('.', ','), ha='right', fontsize=8.5)
    ax.axhline(criterio, color=gc.TINTA_SUAVE, ls='--', lw=1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Umbral de decisión τ'); ax.set_ylabel('F1 macro')
    ax.set_title(f'{NOMBRE_MODELO[mejor]}: F1 macro según el umbral')
    ax.legend(fontsize=8, loc='upper left')
    gc.coma(ax, 'x', 2); gc.coma(ax, 'y', 1)
    out['umbral'] = gc.guardar(fig, IMG / 'f1_macro_segun_umbral_modelo_elegido.png')
    return {k: 'Imagenes/' + v.name for k, v in out.items()}


def main():
    j = json.load(open(cfg.METRICAS_SENSACIONALISMO_JSON, encoding='utf-8'))
    meta, modelos = j['metadata'], {k: r for k, r in j['modelos'].items() if 'error' not in r}
    criterio = float(meta['criterio_aceptacion'].split('>=')[1].split()[0])

    def media_sens(r):
        return sum(r['conjuntos'][n]['umbral_calibrado']['f1_macro'] for n in CONJUNTOS_SENSACIONALISMO) / len(CONJUNTOS_SENSACIONALISMO)

    ranking = sorted(modelos, key=lambda k: media_sens(modelos[k]), reverse=True)
    mejor = ranking[0]
    rm = modelos[mejor]
    cumple = all(rm['conjuntos'][n]['umbral_calibrado']['f1_macro'] >= criterio for n in CONJUNTOS_SENSACIONALISMO)
    figs = figuras(modelos, ranking, mejor, criterio)

    L = ["# Evaluación de modelos de sensacionalismo en español (sin ajuste fino)", "",
         "**Generado por:** `Evaluacion_Modelos/generar_reporte_sensacionalismo.py` a partir de `reportes/metricas_modelos_sensacionalismo.json`  ",
         "**Evaluación:** `Evaluacion_Modelos/evaluar_modelos_sensacionalismo.py` (predicciones en `Evaluacion_Modelos/cache/`)  ",
         f"**Clase positiva:** {meta['clase_positiva']}  ",
         f"**Umbral:** {meta['umbral']}  ",
         f"**Criterio de aceptación:** F1 macro ≥ {f(criterio, 2)} en los conjuntos de prueba de sensacionalismo", "",
         gc.diagrama_md('evaluacion_sensacionalismo.png', cfg.SENSACIONALISMO_REPORTES_DIR, 'Flujo de la evaluación de sensacionalismo'),
         "## 1. Conjuntos de evaluación", "",
         "| Conjunto | n | Positivos | Rol | Unidad | Constructo |", "|:---|---:|---:|:---|:---|:---|"]
    for n, c in meta['conjuntos'].items():
        L.append(f"| `{n}` | {c['n']} | {c['positivos']} | {c['rol']} | {c['unidad']} | {c['constructo']} |")
    L += ["", "> No existe un corpus público de sensacionalismo en español con etiquetas humanas documentadas. "
          "TA1C (Mordecki y Moncecchi, 2025, arXiv:2507.09777; tarea de IberLEF 2025) mide **clickbait**, un constructo distinto, y se usa como referencia externa. "
          "Las etiquetas de Salud no documentan su procedencia. Amarillismo humano reúne solo noticias reales (sin las generadas con GPT).", "",
          "## 2. Modelos evaluados", "",
          "| Modelo | Hugging Face | Constructo de entrenamiento | Índice de la clase positiva | Umbral calibrado |", "|:---|:---|:---|:---|---:|"]
    for k, r in modelos.items():
        L.append(f"| `{k}` | `{r['id']}` (rev. `{(r['revision'] or '')[:12]}`) | {r['constructo']} | {r['regla_indice']} | {f(r['umbral_calibrado'], 2)} |")

    L += ["", "## 3. Ranking en los conjuntos de sensacionalismo (umbral calibrado)", "",
          "| # | Modelo | F1 macro medio | Salud F1 macro [IC 95 %] | Salud recall | Salud AUC | Amarillismo F1 macro [IC 95 %] | Amarillismo recall | Amarillismo AUC |",
          "|---:|:---|---:|:---:|---:|---:|:---:|---:|---:|"]
    for i, k in enumerate(ranking, 1):
        s = modelos[k]['conjuntos']['salud_prueba']['umbral_calibrado']
        a = modelos[k]['conjuntos']['amarillismo_humano_cuerpo']['umbral_calibrado']
        L.append(f"| {i} | `{k}` | **{f(media_sens(modelos[k]))}** | {f(s['f1_macro'])} {ic(s['f1_macro_ic95'])} | {f(s['recall'])} | {f(s['roc_auc'])} | "
                 f"{f(a['f1_macro'])} {ic(a['f1_macro_ic95'])} | {f(a['recall'])} | {f(a['roc_auc'])} |")

    alcanzan = {n: [k for k in modelos if modelos[k]['conjuntos'][n]['umbral_calibrado']['f1_macro'] >= criterio]
                for n, _ in CONJUNTOS_FIG}
    mejor_auc = {n: max(modelos, key=lambda k: modelos[k]['conjuntos'][n]['umbral_calibrado']['roc_auc']) for n in CONJUNTOS_SENSACIONALISMO}
    L += ["", f"![F1 macro por modelo y conjunto]({figs['f1']})", "",
          "Cada barra es el F1 macro con el umbral calibrado de cada modelo y la línea horizontal su intervalo de confianza del 95 % (bootstrap). "
          f"La línea discontinua marca el criterio de aceptación ({f(criterio, 2)}) y la punteada el azar (0,5). " +
          ("En los conjuntos de sensacionalismo ningún modelo alcanza el criterio" if not any(alcanzan[n] for n in CONJUNTOS_SENSACIONALISMO)
           else "Alcanzan el criterio en los conjuntos de sensacionalismo: " + ", ".join(f"`{k}`" for n in CONJUNTOS_SENSACIONALISMO for k in alcanzan[n])) +
          f"; en TA1C, que mide clickbait, lo alcanzan {len(alcanzan[CONJUNTO_CLICKBAIT])} de {len(modelos)} modelos.", "",
          f"![Curvas ROC]({figs['roc']})", "",
          "Las curvas ROC muestran la capacidad de ordenar los textos sin depender del umbral (entre paréntesis, el AUC). Los modelos de clickbait "
          "quedan cerca de la diagonal (azar) en los conjuntos de sensacionalismo y se separan de ella en TA1C. "
          + (f"En Salud y en Amarillismo humano la curva más alejada de la diagonal es la del modelo elegido (línea más gruesa)."
             if all(v == mejor for v in mejor_auc.values()) else
             "Mayor AUC en sensacionalismo: " + ", ".join(f"{NOMBRES[n]}: `{k}`" for n, k in mejor_auc.items()) + "."), "",
          "## 4. Detalle por modelo y conjunto", "",
          "Para cada conjunto se muestran las métricas con el umbral calibrado y, entre paréntesis, con 0,5. "
          "En NoticIA todos los textos son positivos, por lo que solo se informa el recall.", ""]
    for k in ranking:
        r = modelos[k]
        L += [f"### `{k}` — umbral calibrado {f(r['umbral_calibrado'], 2)}", "",
              "| Conjunto | Recall | Precisión | F1 positiva | F1 macro [IC 95 %] | Especificidad | ROC-AUC | PR-AUC | Matriz [[VN, FP], [FN, VP]] |",
              "|:---|---:|---:|---:|:---:|---:|---:|---:|:---|"]
        for n, nombre in NOMBRES.items():
            c, h = r['conjuntos'][n]['umbral_calibrado'], r['conjuntos'][n]['umbral_0_5']
            if 'f1_macro' not in c:
                L.append(f"| {nombre} | {f(c['recall'])} ({f(h['recall'])}) | — | — | — | — | — | — | {c['cm']} |")
                continue
            L.append(f"| {nombre} | {f(c['recall'])} ({f(h['recall'])}) | {f(c['precision'])} ({f(h['precision'])}) | "
                     f"{f(c['f1_positiva'])} ({f(h['f1_positiva'])}) | {f(c['f1_macro'])} {ic(c['f1_macro_ic95'])} ({f(h['f1_macro'])}) | "
                     f"{f(c['especificidad'])} | {f(c['roc_auc'])} | {f(c['pr_auc'])} | {c['cm']} |")
        L.append("")

    L += [f"### Comportamiento del modelo elegido (`{mejor}`)", "",
          f"![Distribución de probabilidades]({figs['dist']})", "",
          f"Proporción de textos de cada clase según la probabilidad de sensacionalismo que asigna `{mejor}`. Si el modelo separara bien las clases, "
          "las dos distribuciones quedarían a lados opuestos del umbral τ (línea discontinua). El solapamiento entre ambas es lo que explica sus errores.", "",
          f"![Matrices de confusión]({figs['cm']})", "",
          "Matrices de confusión con el umbral calibrado (conteo y porcentaje por fila). La celda inferior izquierda son los falsos negativos "
          "(textos sensacionalistas que el modelo no detecta) y la superior derecha los falsos positivos.", "",
          f"![F1 macro según el umbral]({figs['umbral']})", "",
          "F1 macro del modelo elegido para cada umbral posible. Las curvas punteadas son los conjuntos de calibración, donde se eligió τ; las "
          "continuas son los de prueba, que no intervinieron en la elección. La línea horizontal es el criterio de aceptación.", ""]

    clickbait = max(modelos, key=lambda k: modelos[k]['conjuntos'][CONJUNTO_CLICKBAIT]['umbral_calibrado']['f1_macro'])
    cb = modelos[clickbait]['conjuntos'][CONJUNTO_CLICKBAIT]['umbral_calibrado']
    s, a = rm['conjuntos']['salud_prueba']['umbral_calibrado'], rm['conjuntos']['amarillismo_humano_cuerpo']['umbral_calibrado']
    L += ["## 5. Decisión", "",
          f"- **Modelo elegido:** `{mejor}` (`{rm['id']}`), con umbral calibrado τ = {f(rm['umbral_calibrado'], 2)}. "
          f"Es el de mayor F1 macro medio en los conjuntos de sensacionalismo ({f(media_sens(rm))}) y el único entrenado para ese constructo.",
          ("- **Cumple el criterio** de F1 macro ≥ " + f(criterio, 2) + " en los conjuntos de sensacionalismo." if cumple else
           f"- **No cumple el criterio** de F1 macro ≥ {f(criterio, 2)}: obtiene {f(s['f1_macro'])} en Salud (recall {f(s['recall'])}, AUC {f(s['roc_auc'])}) "
           f"y {f(a['f1_macro'])} en Amarillismo humano (recall {f(a['recall'])}, AUC {f(a['roc_auc'])}). "
           "Siguiendo lo acordado, no se hace ajuste fino: se mantiene el modelo con su umbral calibrado y el resultado se declara como limitación."),
          f"- Los modelos de clickbait funcionan bien en TA1C (el mejor es `{clickbait}`, F1 macro {f(cb['f1_macro'])}, AUC {f(cb['roc_auc'])}), "
          "pero quedan cerca del azar en Salud y en Amarillismo humano. Esto confirma que clickbait y sensacionalismo son constructos distintos y que un detector de clickbait no sustituye a uno de sensacionalismo.",
          "- El umbral es único por modelo y se elige en la unión de TA1C validation y la mitad de calibración de Salud. Para los modelos de clickbait esa mezcla lleva el umbral al extremo de la rejilla (0,05); "
          "es una consecuencia de calibrar un solo umbral para dos constructos y se informa tal cual.", "",
          "## 6. Limitaciones que deben declararse", "",
          f"1. El modelo de sensacionalismo elegido no alcanza F1 macro ≥ {f(criterio, 2)} en datos con etiquetas humanas; su puntaje debe interpretarse como un **indicador de estilo débil**, no como una clasificación fiable.",
          "2. La tarjeta de `JJNeila/bert-spanish-sensationalism-oss` no documenta la anotación de su corpus; es probable que sus etiquetas se hayan generado con un LLM (inferencia, no confirmada).",
          "3. Las etiquetas de Salud no tienen procedencia documentada y sus textos son solo titulares.",
          f"4. Amarillismo humano tiene solo {meta['conjuntos']['amarillismo_humano_cuerpo']['n']} noticias; los intervalos de confianza son amplios.",
          "5. El acierto perfecto en IA sintético es un control de cordura: esos textos son caricaturas del estilo sensacionalista y no representan noticias reales.",
          "6. No se hizo ajuste fino por decisión metodológica; un corpus anotado por personas en español sería el paso necesario para mejorar este componente.", ""]

    salida = cfg.SENSACIONALISMO_REPORTES_DIR / "Reporte_Evaluacion_Modelos_Sensacionalismo.md"
    salida.write_text("\n".join(L) + "\n", encoding='utf-8')
    print(f"[OK] {salida}  (elegido: {mejor}, tau = {rm['umbral_calibrado']}, cumple criterio: {cumple})")


if __name__ == '__main__':
    main()
