"""
Genera Reportes/Reporte_Evaluacion_Modelos_Emociones.md a partir de Reportes/metricas_modelos_emociones.json.
Todas las cifras del reporte se leen del JSON; no hay valores escritos a mano.

Regla de selección (fijada antes de ver los resultados): se elige el modelo con mayor F1 macro (7 clases) en la
partición de prueba de EmoEvent. Si los intervalos de confianza se solapan, se informa como empate estadístico y se
desempata por el F1 macro de las 6 emociones (sin «others»), que es lo que aporta a la explicabilidad.
"""

import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config_proyecto.py").exists())))
import config_proyecto as cfg
import graficos_comunes as gc

IMG = cfg.EMOCIONES_REPORTES_DIR / "Imagenes"
CACHE = Path(__file__).resolve().parent / "cache"
NOMBRE_MODELO = {'robertuito_pysentimiento': 'RoBERTuito (pysentimiento)', 'beto_pysentimiento': 'BETO (pysentimiento)',
                 'xlmr_emoevales_gsiupm': 'XLM-R (GSI-UPM)'}


def f(x, d=3):
    return f"{x:.{d}f}".replace('.', ',')


def pct(x, d=1):
    return f"{x * 100:.{d}f}".replace('.', ',') + " %"


def muestra_noticias():
    """Misma muestra estratificada que evaluar_modelos_emociones.py (mismo orden de filas que la caché)."""
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from evaluar_modelos_emociones import N_MUESTRA_NOTICIAS
    n = pd.read_csv(cfg.CORPUS_AMPLIADO_CSV).drop_duplicates(subset='Text')
    return n.groupby(['region', 'clase_num']).sample(frac=N_MUESTRA_NOTICIAS / len(n), random_state=cfg.SEED)


def figuras(j, ranking, mejor):
    plt = gc.pyplot()
    meta, modelos = j['metadata'], j['modelos']
    E, ES = meta['clases'], meta['nombres_es']
    color = {k: gc.SERIE[i] for i, k in enumerate(modelos)}
    out = {}

    # 1. F1 macro (7 clases) con IC y F1 macro de las 6 emociones
    fig, ax = plt.subplots(figsize=(8, 2.9))
    orden = ranking[::-1]
    for i, k in enumerate(orden):
        e = modelos[k]['emoevent_test']
        lo, hi = e['f1_macro_ic95']
        ax.barh(i, e['f1_macro'], color=color[k], height=0.6)
        ax.errorbar(e['f1_macro'], i, xerr=[[e['f1_macro'] - lo], [hi - e['f1_macro']]], color=gc.TINTA_2, capsize=3, lw=1)
        ax.plot(e['f1_macro_6_emociones'], i, marker='D', color=gc.TINTA, ms=7, ls='none',
                label='F1 macro de las 6 emociones (sin «others»)' if i == 0 else None)
    criterio = float(meta['criterio_referencia'].split('>=')[1].split()[0])
    ax.axvline(criterio, color=gc.TINTA_2, ls='--', lw=1, label=f"Criterio de referencia ({criterio:.2f})".replace('.', ','))
    ax.set_yticks(range(len(orden))); ax.set_yticklabels([NOMBRE_MODELO[k] for k in orden])
    ax.set_xlim(0, 1); ax.grid(axis='y', visible=False)
    ax.set_xlabel('F1 macro en EmoEvent (prueba)')
    ax.set_title('F1 macro de las 7 clases (barra, IC 95 %) y de las 6 emociones (rombo)')
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2)
    gc.coma(ax, 'x', 1)
    out['f1'] = gc.guardar(fig, IMG / 'f1_macro_por_modelo.png')

    # 2. F1 por emoción y modelo
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ancho = 0.8 / len(modelos)
    for i, k in enumerate(modelos):
        pc = modelos[k]['emoevent_test']['por_clase']
        ax.barh(np.arange(len(E)) + (i - 1) * ancho, [pc[e]['f1'] for e in E], height=ancho * 0.92, color=color[k], label=NOMBRE_MODELO[k])
        for jj, e in enumerate(E):
            if pc[e]['f1'] == 0:
                ax.text(0.005, jj + (i - 1) * ancho, '0', va='center', fontsize=7.5, color=color[k])
    sop = modelos[mejor]['emoevent_test']['por_clase']
    ax.set_yticks(range(len(E)))
    ax.set_yticklabels([f"{ES[e]} (n = {sop[e]['soporte']})" for e in E])
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.grid(axis='y', visible=False)
    ax.set_xlabel('F1 en EmoEvent (prueba)')
    ax.set_title('F1 por emoción')
    ax.legend(fontsize=8, loc='lower right')
    gc.coma(ax, 'x', 1)
    out['f1_emocion'] = gc.guardar(fig, IMG / 'f1_por_emocion_y_modelo.png')

    # 3. Matriz de confusión del modelo elegido (normalizada por fila)
    cm = np.array(modelos[mejor]['emoevent_test']['matriz_confusion'], dtype=float)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.2, 6))
    im = ax.imshow(norm, cmap=gc.secuencial(), vmin=0, vmax=1)
    for a in range(len(E)):
        for b in range(len(E)):
            ax.text(b, a, f"{100 * norm[a, b]:.0f}", ha='center', va='center', fontsize=8.5, color='white' if norm[a, b] > 0.6 else gc.TINTA)
    ax.set_xticks(range(len(E))); ax.set_yticks(range(len(E)))
    ax.set_xticklabels([ES[e] for e in E], rotation=35, ha='right'); ax.set_yticklabels([ES[e] for e in E])
    ax.set_xlabel('Predicción'); ax.set_ylabel('Etiqueta humana'); ax.grid(False)
    ax.set_title(f'{NOMBRE_MODELO[mejor]}: matriz de confusión (% por fila)')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out['cm'] = gc.guardar(fig, IMG / 'matriz_confusion_modelo_elegido.png')

    # 4. Emoción dominante predicha en noticias por modelo
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, k in enumerate(modelos):
        d = modelos[k]['noticias_muestra']['distribucion_predicha']
        ax.bar(np.arange(len(E)) + (i - 1) * ancho, [d[e] for e in E], width=ancho * 0.92, color=color[k], label=NOMBRE_MODELO[k])
    ax.set_xticks(range(len(E))); ax.set_xticklabels([ES[e] for e in E], rotation=20, ha='right')
    ax.set_ylabel('Proporción de noticias'); ax.grid(axis='x', visible=False)
    ax.set_title(f"Emoción dominante predicha en {meta['conjuntos']['noticias_muestra']['n']} noticias del corpus")
    ax.legend(fontsize=8)
    gc.coma(ax, 'y', 1)
    out['noticias'] = gc.guardar(fig, IMG / 'emocion_dominante_en_noticias.png')

    # 5. Intensidad emocional del modelo elegido en noticias, por región y clase
    mu = muestra_noticias()
    p = np.load(CACHE / f"{mejor}__noticias_muestra.npy")
    assert len(p) == len(mu), (len(p), len(mu))
    assert abs(float((1 - p[:, E.index('others')]).mean()) - modelos[mejor]['noticias_muestra']['intensidad_emocional_media']) < 1e-6
    mu = mu.assign(intensidad=1 - p[:, E.index('others')])
    fig, ax = plt.subplots(figsize=(8, 3.8))
    grupos, pos, cols = [], [], []
    for r_i, reg in enumerate(sorted(mu['region'].unique())):
        for c in (0, 1):
            grupos.append(mu.loc[(mu['region'] == reg) & (mu['clase_num'] == c), 'intensidad'].values)
            pos.append(r_i * 3 + c)
            cols.append(gc.CLASE[c])
    bp = ax.boxplot(grupos, positions=pos, widths=0.7, patch_artist=True, showfliers=True,
                    medianprops={'color': gc.TINTA}, flierprops={'markersize': 3, 'markeredgecolor': gc.TINTA_SUAVE})
    for b, c in zip(bp['boxes'], cols):
        b.set(facecolor=c, alpha=0.55, edgecolor=c)
    regiones = sorted(mu['region'].unique())
    ax.set_xticks([i * 3 + 0.5 for i in range(len(regiones))])
    ax.set_xticklabels([f"{reg}\n(n = {int(((mu['region'] == reg) & (mu['clase_num'] == 0)).sum())} reales / "
                        f"{int(((mu['region'] == reg) & (mu['clase_num'] == 1)).sum())} falsas)" for reg in regiones])
    ax.set_ylabel('Intensidad emocional (1 − P(others))'); ax.set_ylim(0, 1); ax.grid(axis='x', visible=False)
    ax.set_title(f'{NOMBRE_MODELO[mejor]}: intensidad emocional en noticias por región y clase')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=gc.CLASE[0], alpha=0.55, label='Real'), Patch(color=gc.CLASE[1], alpha=0.55, label='Falsa')], fontsize=8)
    gc.coma(ax, 'y', 1)
    out['intensidad'] = gc.guardar(fig, IMG / 'intensidad_emocional_noticias_por_clase.png')
    med = {(reg, c): float(np.median(mu.loc[(mu['region'] == reg) & (mu['clase_num'] == c), 'intensidad'])) for reg in regiones for c in (0, 1)}
    return {k: 'Imagenes/' + v.name for k, v in out.items()}, med


def main():
    j = json.load(open(cfg.METRICAS_EMOCIONES_JSON, encoding='utf-8'))
    meta, modelos, acuerdo = j['metadata'], j['modelos'], j['acuerdo_en_noticias']
    E, ES = meta['clases'], meta['nombres_es']
    ranking = sorted(modelos, key=lambda k: modelos[k]['emoevent_test']['f1_macro'], reverse=True)
    mejor, segundo = ranking[0], ranking[1]
    ic1, ic2 = modelos[mejor]['emoevent_test']['f1_macro_ic95'], modelos[segundo]['emoevent_test']['f1_macro_ic95']
    empate = ic2[1] >= ic1[0]
    if empate and modelos[segundo]['emoevent_test']['f1_macro_6_emociones'] > modelos[mejor]['emoevent_test']['f1_macro_6_emociones']:
        mejor, segundo = segundo, mejor
    rm = modelos[mejor]
    criterio = float(meta['criterio_referencia'].split('>=')[1].split()[0])
    ev = meta['conjuntos']['emoevent_test']
    dg = ev['diagnostico_etiquetas']
    figs, med = figuras(j, ranking, mejor)

    L = ["# Evaluación de modelos de emociones en español (sin ajuste fino)", "",
         "**Generado por:** `Evaluacion_Modelos/generar_reporte_emociones.py` a partir de `Reportes/metricas_modelos_emociones.json`  ",
         "**Evaluación:** `Evaluacion_Modelos/evaluar_modelos_emociones.py` (probabilidades en `Evaluacion_Modelos/cache/`)  ",
         f"**Clases:** {', '.join(f'{e} ({ES[e]})' for e in E)}  ",
         f"**Decisión:** {meta['decision']}  ",
         f"**Criterio de referencia:** F1 macro ≥ {f(criterio, 2)} en la partición de prueba de EmoEvent", "",
         gc.diagrama_md('evaluacion_emociones.png', cfg.EMOCIONES_REPORTES_DIR, 'Flujo de la evaluación de emociones'),
         "## 1. Por qué estos tres modelos", "",
         "Son los modelos de emociones en español con respaldo en publicaciones revisadas por pares y usados por otros autores. "
         "Cubren las tres familias de modelos de lenguaje más empleadas en español (RoBERTa entrenado en tuits, BERT en español y XLM-RoBERTa multilingüe):", "",
         "| Modelo | Hugging Face (revisión) | Referencia | Preprocesamiento |", "|:---|:---|:---|:---|"]
    for k, r in modelos.items():
        L.append(f"| `{k}` | `{r['id']}` (`{r['revision'][:12]}`) | {r['referencia']} | {r['preproceso']} |")
    L += ["", "Las salidas se reordenan por el nombre de cada etiqueta (`id2label`) porque el orden de los índices difiere entre modelos "
          "(por ejemplo, el índice 0 es «others» en pysentimiento y «sadness» en el modelo de GSI-UPM).", "",
          "## 2. Datos de validación", "",
          f"- **EmoEvent / EmoEvalEs** (Plaza-del-Arco et al., LREC 2020; tarea EmoEvalEs de IberLEF 2021): {ev['n']} tuits en español de la partición de prueba, "
          "anotados por personas. Ninguno de los tres modelos vio esta partición durante su entrenamiento.",
          "  Distribución: " + ", ".join(f"{ES[e]} {v}" for e, v in ev['distribucion'].items()) + ".", "",
          "> **Corrección de etiquetas.** En el archivo publicado `splits/es/test.tsv`, la columna de emoción está desplazada 30 filas respecto de los textos "
          f"(el archivo termina con {dg['test']['sin_texto']} filas que solo tienen etiqueta). La etiqueta de cada tuit se tomó del corpus completo publicado "
          "(`emoevent_es.csv`), cruzando por el texto normalizado. Comprobación:", "",
          "| Partición | Filas | Sin texto | Encontradas en el corpus completo | Etiqueta del archivo = etiqueta del corpus |", "|:---|---:|---:|---:|---:|",
          *[f"| {p} | {dg[p]['filas']} | {dg[p]['sin_texto']} | {dg[p]['encontradas_en_corpus']} | {pct(dg[p]['coincidencia_etiqueta_archivo_vs_corpus'], 2)} |" for p in ['train', 'dev', 'test']],
          "", "En train y dev las etiquetas coinciden prácticamente por completo, lo que valida el cruce; en test solo coinciden por azar. "
          f"Los {dg['test']['filas'] - dg['test']['sin_texto'] - dg['test']['encontradas_en_corpus']} tuits de prueba que no se encontraron en el corpus completo se excluyen.", "",
          f"- **Noticias del corpus de la tesis:** muestra estratificada de {meta['conjuntos']['noticias_muestra']['n']} noticias sin etiquetas de emoción "
          "(" + ", ".join(f"{k}: {v}" for k, v in meta['conjuntos']['noticias_muestra']['estratos'].items()) + "). "
          "Se usa para medir si los modelos coinciden entre sí al aplicarse a noticias.", "",
          "> No se encontró un corpus público de noticias en español anotado con emociones por personas. "
          "Por eso la validación con etiquetas se hace sobre tuits y la transferencia al dominio de noticias se evalúa de forma indirecta (acuerdo entre modelos).", "",
          "## 3. Resultados en EmoEvent (prueba)", "",
          "| # | Modelo | F1 macro [IC 95 %] | F1 macro 6 emociones | F1 ponderado | Exactitud | Recall macro | ROC-AUC macro |",
          "|---:|:---|:---:|---:|---:|---:|---:|---:|"]
    for i, k in enumerate(ranking, 1):
        e = modelos[k]['emoevent_test']
        L.append(f"| {i} | `{k}` | **{f(e['f1_macro'])}** [{f(e['f1_macro_ic95'][0])}; {f(e['f1_macro_ic95'][1])}] | {f(e['f1_macro_6_emociones'])} | "
                 f"{f(e['f1_ponderado'])} | {f(e['accuracy'])} | {f(e['recall_macro'])} | {f(e['roc_auc_macro_ovr'])} |")
    L += ["", "### Recall y F1 por emoción", "",
          "| Emoción | Soporte | " + " | ".join(f"`{k}` recall / F1" for k in ranking) + " |",
          "|:---|---:|" + "---:|" * len(ranking)]
    for e in E:
        sop = modelos[ranking[0]]['emoevent_test']['por_clase'][e]['soporte']
        L.append(f"| {ES[e]} | {sop} | " + " | ".join(
            f"{f(modelos[k]['emoevent_test']['por_clase'][e]['recall'], 2)} / {f(modelos[k]['emoevent_test']['por_clase'][e]['f1'], 2)}" for k in ranking) + " |")

    L += ["", f"![F1 macro por modelo]({figs['f1']})", "",
          "F1 macro de las 7 clases con su intervalo de confianza del 95 % (barra) y F1 macro de las 6 emociones sin «others» (rombo). "
          "Los intervalos de los tres modelos se solapan en buena parte; la línea discontinua es el criterio de referencia.", "",
          f"![F1 por emoción]({figs['f1_emocion']})", "",
          "F1 de cada emoción (entre paréntesis, cuántos tuits de prueba tiene). Las emociones con pocos ejemplos son las más inestables y las que más "
          "varían entre modelos.", "",
          f"![Matriz de confusión]({figs['cm']})", "",
          f"Matriz de confusión de `{mejor}` en porcentaje por fila: la diagonal es el acierto de cada emoción y el resto muestra con qué se confunde. "
          "Buena parte de los errores van hacia «sin emoción dominante», la clase más frecuente." +
          (lambda cm, nunca: (f" El modelo nunca predice {', '.join('«' + ES[E[c]] + '»' for c in nunca)}: " +
                              "; ".join(f"«{ES[E[c]]}» se confunde sobre todo con «{ES[E[int(np.argmax(cm[c]))]]}»" for c in nunca) + ".") if nunca else '')(
              np.array(rm['emoevent_test']['matriz_confusion']),
              [c for c in range(len(E)) if np.array(rm['emoevent_test']['matriz_confusion'])[:, c].sum() == 0]), "",
          "## 4. Comportamiento en noticias", "",
          "| Modelo | " + " | ".join(ES[e] for e in E) + " | Intensidad emocional media | ms por noticia (CPU) |",
          "|:---|" + "---:|" * (len(E) + 2)]
    for k in ranking:
        n = modelos[k]['noticias_muestra']
        L.append(f"| `{k}` | " + " | ".join(pct(n['distribucion_predicha'][e]) for e in E) +
                 f" | {f(n['intensidad_emocional_media'])} | {f(modelos[k]['ms_por_noticia'], 0)} |")
    L += ["", "La intensidad emocional es 1 − P(sin emoción dominante): cuánto peso pone el modelo en alguna emoción.", "",
          "| Par de modelos | Kappa de Cohen | Acuerdo bruto |", "|:---|---:|---:|"]
    for par, a in acuerdo.items():
        L.append(f"| {par} | {f(a['kappa_cohen'])} | {pct(a['acuerdo_bruto'])} |")
    L += ["", f"![Emoción dominante en noticias]({figs['noticias']})", "",
          "Proporción de noticias de la muestra según la emoción con mayor probabilidad para cada modelo. Muestra cuánto difieren los modelos al "
          "pasar de tuits a noticias.", "",
          f"![Intensidad emocional por clase]({figs['intensidad']})", "",
          f"Intensidad emocional que asigna `{mejor}` a las noticias de la muestra, por región y clase (medianas: " +
          "; ".join(f"{reg} {'real' if c == 0 else 'falsa'} {f(v, 2)}" for (reg, c), v in med.items()) +
          "). Es una descripción de la muestra, no una prueba estadística; la asociación de las variables de emoción con la veracidad se analiza en el "
          "Modelo sobre Variables."]

    e = rm['emoevent_test']
    L += ["", "## 5. Decisión", "",
          f"- **Modelo elegido:** `{mejor}` (`{rm['id']}`), F1 macro = {f(e['f1_macro'])} [{f(e['f1_macro_ic95'][0])}; {f(e['f1_macro_ic95'][1])}], "
          f"F1 macro de las 6 emociones = {f(e['f1_macro_6_emociones'])}.",
          (f"- Su intervalo de confianza se solapa con el de `{segundo}`: la diferencia no es estadísticamente clara y la elección se apoya en el F1 de las 6 emociones."
           if empate else f"- Su intervalo de confianza no se solapa con el del segundo (`{segundo}`)."),
          (f"- **Cumple** el criterio de referencia (F1 macro ≥ {f(criterio, 2)})." if e['f1_macro'] >= criterio else
           f"- **No cumple** el criterio de referencia (F1 macro ≥ {f(criterio, 2)}). Sin ajuste fino, se usa tal cual y se declara como limitación."),
          "- En la tesis, el modelo se usa como **tercer pilar de la explicabilidad propia** (junto con sensacionalismo y redundancia): "
          "aporta a cada noticia la probabilidad de cada emoción y una intensidad emocional, que entran al IML y a los estilos discursivos.", "",
          "## 6. Limitaciones", "",
          "1. La validación con etiquetas humanas se hace sobre tuits, no sobre noticias; el desempeño en noticias puede ser distinto.",
          "2. Las clases minoritarias (asco, miedo, sorpresa) tienen pocos ejemplos en la prueba; su F1 es inestable.",
          "3. Cada modelo lee solo el comienzo de la noticia (tokens por noticia: " +
          ", ".join(f"`{k}` {r['max_len_efectivo']['noticias_muestra']}" for k, r in modelos.items()) + ").",
          "4. No se hizo ajuste fino, por decisión metodológica.", ""]

    salida = cfg.EMOCIONES_REPORTES_DIR / "Reporte_Evaluacion_Modelos_Emociones.md"
    salida.write_text("\n".join(L) + "\n", encoding='utf-8')
    print(f"[OK] {salida}  (elegido: {mejor}, empate estadístico con el segundo: {empate})")


if __name__ == '__main__':
    main()
