"""
Genera Reportes/Reporte_Redundancia_hiiamsid.md a partir de Reportes/metricas_redundancia_hiiamsid.json.
Todas las cifras del reporte se leen del JSON; no hay valores escritos a mano.
"""

import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config_proyecto.py").exists())))
import config_proyecto as cfg
import graficos_comunes as gc

IMG = cfg.REDUNDANCIA_REPORTES_DIR / "Imagenes"
NOMBRE_UMBRAL = {'tau_tema': 'Mismo tema', 'tau_parafrasis': 'Paráfrasis aproximada', 'tau_redundante': 'Redundante'}
NOMBRE_VAR = {'max_intra_similarity': 'Similitud máxima', 'second_max_intra_similarity': 'Segunda similitud máxima',
              'mean_intra_similarity': 'Similitud media', 'consec_sim_mean': 'Similitud media entre oraciones consecutivas'}


def figuras(j):
    import pandas as pd
    plt = gc.pyplot()
    sts, paws, c, a = j['sts_es'], j['pawsx_es'], j['corpus'], j['asociacion_veracidad_textos_unicos']
    U = list(sts['umbrales'])
    col_u = {u: gc.SERIE[i + 2] for i, u in enumerate(U)}
    out = {}

    # 1. Calidad de cada umbral en STS-B (test) y AUC frente a PAWS-X
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={'width_ratios': [2.2, 1]})
    mets = [('precision', 'Precisión'), ('recall', 'Exhaustividad'), ('f1', 'F1')]
    for i, (m, et) in enumerate(mets):
        a1.bar(np.arange(len(U)) + (i - 1) * 0.26, [sts['umbrales'][u]['test'][m] for u in U], width=0.24, color=gc.SERIE[i], label=et)
    a1.set_xticks(range(len(U)))
    a1.set_xticklabels([f"{NOMBRE_UMBRAL[u]}\nτ = {sts['umbrales'][u]['tau']:.2f} (STS ≥ {sts['umbrales'][u]['corte_sts']:g})".replace('.', ',') for u in U])
    a1.set_ylim(0, 1); a1.grid(axis='x', visible=False)
    a1.set_title('Umbrales elegidos en STS-B dev, evaluados en STS-B test')
    a1.legend(fontsize=8, ncol=3, loc='upper right')
    gc.coma(a1, 'y', 1)
    aucs = [(f"STS-B: {NOMBRE_UMBRAL[u].lower()}", sts['umbrales'][u]['test']['roc_auc'], gc.AZUL_SEC[3 + i]) for i, u in enumerate(U)] + \
           [('PAWS-X (paráfrasis adversariales)', paws['test_con_tau_youden']['roc_auc'], gc.TINTA_SUAVE)]
    a2.barh(range(len(aucs))[::-1], [x[1] for x in aucs], color=[x[2] for x in aucs], height=0.6)
    a2.set_yticks(range(len(aucs))[::-1]); a2.set_yticklabels([x[0] for x in aucs], fontsize=8.5)
    for i, (_, v, _) in enumerate(aucs):
        a2.text(v + 0.01, len(aucs) - 1 - i, f"{v:.2f}".replace('.', ','), va='center', fontsize=8.5)
    a2.axvline(0.5, color=gc.TINTA_SUAVE, ls=':', lw=1)
    a2.set_xlim(0, 1.08); a2.grid(axis='y', visible=False)
    a2.set_title('ROC-AUC (test)'); gc.coma(a2, 'x', 1)
    fig.tight_layout()
    out['umbrales'] = gc.guardar(fig, IMG / 'calidad_umbrales_sts_y_pawsx.png')

    # 2. Similitudes de todos los pares intra-documentales con los umbrales
    d = pd.read_csv(cfg.REDUNDANCIA_CORPUS_CSV)
    pares = np.concatenate([json.loads(s) for s in d['pairwise_similarities'].dropna() if s and s != '[]'])
    assert len(pares) == c['n_pares'], (len(pares), c['n_pares'])
    fig, (b1, b2) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={'width_ratios': [2.2, 1]})
    b1.hist(pares, bins=60, color=gc.SERIE[0], alpha=0.75, weights=np.ones(len(pares)) / len(pares))
    for i, u in enumerate(U):
        t = sts['umbrales'][u]['tau']
        b1.axvline(t, color=col_u[u], lw=1.6, ls='--')
        b1.text(t + 0.008, b1.get_ylim()[1] * (0.97 - 0.17 * i), f"{NOMBRE_UMBRAL[u]}\n{100 * c['pct_pares_sobre_umbral'][u]:.1f} % de pares".replace('.', ','),
                fontsize=8, color=gc.TINTA_2, va='top')
    b1.set_xlabel('Similitud coseno entre dos oraciones de la misma noticia'); b1.set_ylabel('Proporción de pares')
    b1.set_title(f"{c['n_pares']:,} pares de {c['n_con_2_o_mas_oraciones']:,} noticias con 2 o más oraciones".replace(',', '.'))
    gc.coma(b1, 'x', 1); gc.coma(b1, 'y', 2)
    v = [c['pct_noticias_con_algun_par_sobre_umbral'][u] for u in U]
    b2.barh(range(len(U))[::-1], v, color=[col_u[u] for u in U], height=0.6)
    b2.set_yticks(range(len(U))[::-1]); b2.set_yticklabels([NOMBRE_UMBRAL[u] for u in U])
    for i, x in enumerate(v):
        b2.text(x + 0.01, len(U) - 1 - i, f"{100 * x:.1f} %".replace('.', ','), va='center', fontsize=8.5)
    b2.set_xlim(0, 1); b2.grid(axis='y', visible=False)
    b2.set_title('Noticias con al menos un par sobre el umbral'); gc.coma(b2, 'x', 1)
    fig.tight_layout()
    out['pares'] = gc.guardar(fig, IMG / 'distribucion_similitud_pares_y_umbrales.png')

    # 3. Variables de redundancia por clase (textos únicos con 2 o más oraciones)
    u = d[d['is_unique_text'] & (d['num_sentences'] >= 2)]
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.4))
    for ax, var in zip(axs, NOMBRE_VAR):
        r = a[var]
        for clase, et in [(0, 'Real'), (1, 'Falsa')]:
            x = u.loc[u['label_num'] == clase, var].dropna().values
            assert abs(np.median(x) - (r['mediana_reales'] if clase == 0 else r['mediana_falsas'])) < 1e-6, var
            ax.hist(x, bins=30, range=(0, 1), color=gc.CLASE[clase], alpha=0.5, weights=np.ones(len(x)) / len(x), label=et)
            ax.axvline(np.median(x), color=gc.CLASE[clase], lw=1.4)
        ax.set_title(f"{NOMBRE_VAR[var]}\nAUC = {r['auc_univariado_falsa']:.3f} · n = {r['n_reales']} reales / {r['n_falsas']} falsas".replace('.', ','), fontsize=9.5)
        ax.set_xlabel('Similitud coseno')
        gc.coma(ax, 'x', 1); gc.coma(ax, 'y', 2)
    axs[0].set_ylabel('Proporción de noticias de la clase')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, ['Real (barras; línea = mediana)', 'Falsa (barras; línea = mediana)'], loc='lower center', ncol=2, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out['clases'] = gc.guardar(fig, IMG / 'redundancia_por_clase.png')
    return {k: 'Imagenes/' + p.name for k, p in out.items()}


def f(x, d=4):
    return f"{x:.{d}f}".replace('.', ',')


def pct(x, d=1):
    return f"{x * 100:.{d}f}".replace('.', ',') + " %"


def miles(n):
    return f"{n:,}".replace(',', '.')


def main():
    j = json.load(open(cfg.METRICAS_REDUNDANCIA_JSON, encoding='utf-8'))
    m, sts, paws, c, a = j['metadata'], j['sts_es'], j['pawsx_es'], j['corpus'], j['asociacion_veracidad_textos_unicos']
    figs = figuras(j)
    L = []
    L += ["# Redundancia semántica intra-documental con `hiiamsid/sentence_similarity_spanish_es`", "",
          f"**Generado por:** `generar_reporte_redundancia.py` a partir de `Reportes/metricas_redundancia_hiiamsid.json`  ",
          f"**Pipeline:** `pipeline_redundancia.py` · modelo `{m['modelo']}` (revisión `{m['revision'][:12]}`) · semilla {m['semilla']}  ",
          f"**Segmentación:** {m['segmentacion']}  ",
          f"**Etiqueta de veracidad:** {m['etiqueta_veracidad']}", "",
          gc.diagrama_md('evaluacion_redundancia.png', cfg.REDUNDANCIA_REPORTES_DIR, 'Flujo de la evaluación de redundancia'),
          "## Qué cambió respecto de la versión anterior", "",
          "| Falla anterior | Corrección |", "|:---|:---|",
          "| El umbral τ = 0,34 (mezcla de gaussianas) solo indicaba «mismo tema» | Los umbrales se calibran con similitud anotada por personas (STS-B en español) y tienen una interpretación explícita |",
          "| Los umbrales P90 / P95 / P99 marcaban por construcción un porcentaje fijo de pares | Se eliminaron; los tres umbrales vigentes son absolutos y validados |",
          "| La validación en PAWS-X no coincidía con su JSON y los estratos se ajustaban con etiquetas invertidas | PAWS-X se evalúa con umbral elegido en validation y medido en test; no hay estratos supervisados y la etiqueta es 1 = falsa |",
          "| Se usaban dos modelos SBERT distintos según la etapa | Un único modelo (`hiiamsid`) en todo el pipeline, con revisión fija |", "",
          "## 1. Calibración de umbrales con STS-B en español", "",
          f"Pares anotados por personas en una escala de 0 a 5 (dev = {miles(sts['n_dev'])}, test = {miles(sts['n_test'])}). "
          f"El coseno de `hiiamsid` correlaciona con el juicio humano: Spearman = **{f(sts['spearman_test'])}**, Pearson = {f(sts['pearson_test'])} (test).", "",
          "Cada umbral se elige con el índice de Youden en dev y se evalúa en test:", "",
          "| Umbral | Significado (nivel STS) | τ | ROC-AUC | Precisión | Exhaustividad | F1 |", "|:---|:---|---:|---:|---:|---:|---:|"]
    for n, u in sts['umbrales'].items():
        t = u['test']
        L.append(f"| `{n}` | {u['descripcion']} | {f(u['tau'])} | {f(t['roc_auc'])} | {f(t['precision'])} | {f(t['recall'])} | {f(t['f1'])} |")
    L += ["", "> `tau_redundante` es el umbral que corresponde a **repetir la misma afirmación**. Su precisión moderada indica que una parte de los pares sobre el umbral son solo aproximadamente equivalentes; los resultados se interpretan con esa cautela.", "",
          "## 2. Validación externa con PAWS-X en español", "",
          f"| Criterio | τ | ROC-AUC | Exactitud | F1 |", "|:---|---:|---:|---:|---:|",
          f"| Youden en validation | {f(paws['tau_youden_validation'])} | {f(paws['test_con_tau_youden']['roc_auc'])} | {f(paws['test_con_tau_youden']['accuracy'])} | {f(paws['test_con_tau_youden']['f1'])} |",
          f"| `tau_redundante` (STS) | {f(sts['umbrales']['tau_redundante']['tau'])} | {f(paws['test_con_tau_redundante_sts']['roc_auc'])} | {f(paws['test_con_tau_redundante_sts']['accuracy'])} | {f(paws['test_con_tau_redundante_sts']['f1'])} |",
          "", f"> {paws['nota']} Por eso PAWS-X no se usa para fijar umbrales: se reporta como limitación del enfoque basado en coseno.", "",
          f"![Calidad de los umbrales]({figs['umbrales']})", "",
          "A la izquierda, precisión, exhaustividad y F1 de cada umbral en STS-B test: cuanto más exigente es el umbral, más cae la precisión, "
          "porque hay menos pares que de verdad repiten la misma afirmación. A la derecha, el ROC-AUC de los tres cortes en STS-B frente al de PAWS-X: "
          "en paráfrasis adversariales (mismas palabras, distinto significado) el coseno queda cerca del azar (línea punteada).", "",
          "> Las similitudes par a par de STS-B y PAWS-X no se guardaron (solo sus métricas); por eso no se grafica la similitud frente al puntaje humano.", "",
          "## 3. Corpus base", "",
          f"- Noticias: {miles(c['n_noticias'])}; con al menos 2 oraciones: {miles(c['n_con_2_o_mas_oraciones'])}; textos únicos entre ellas: {miles(c['n_unicos_validos'])}.",
          f"- Oraciones: {miles(c['n_oraciones'])} ({miles(c['n_oraciones_en_validos'])} en noticias con al menos 2); pares intra-documentales: {miles(c['n_pares'])}.", "",
          "| Umbral | Pares sobre el umbral | Noticias con al menos un par sobre el umbral |", "|:---|---:|---:|"]
    for n in sts['umbrales']:
        L.append(f"| `{n}` | {pct(c['pct_pares_sobre_umbral'][n])} | {pct(c['pct_noticias_con_algun_par_sobre_umbral'][n])} |")
    L += ["", f"![Distribución de similitudes]({figs['pares']})", "",
          "Distribución de la similitud de todos los pares de oraciones dentro de una misma noticia, con los tres umbrales (líneas discontinuas). "
          "La mayoría de los pares solo comparten tema; los pares que repiten la misma afirmación son una fracción muy pequeña. A la derecha, la "
          "proporción de noticias que tienen al menos un par sobre cada umbral."]
    L += ["", "## 4. Asociación con la veracidad (textos únicos, prueba U de Mann-Whitney)", "",
          "| Variable | Mediana falsas | Mediana reales | AUC univariado (falsa) | Correlación biserial de rangos | IC 95 % diferencia de medianas | p | p (Holm) |",
          "|:---|---:|---:|---:|---:|:---:|---:|---:|"]
    for v, r in a.items():
        ic = r['ic95_diferencia_medianas']
        L.append(f"| `{v}` | {f(r['mediana_falsas'])} | {f(r['mediana_reales'])} | {f(r['auc_univariado_falsa'], 3)} | {f(r['correlacion_biserial_rangos'], 3)} | "
                 f"[{f(ic[0])}; {f(ic[1])}] | {r['p_valor']:.3g} | {r['p_holm']:.3g} |")
    L += ["", f"![Redundancia por clase]({figs['clases']})", "",
          "Distribución de cuatro variables de redundancia en noticias reales y falsas (textos únicos), con su mediana (línea vertical) y el AUC "
          "univariado. Las distribuciones de ambas clases prácticamente se superponen."]
    sig = [v for v, r in a.items() if r['p_holm'] < 0.05]
    L += ["", "### Lectura", "",
          ("- Ninguna variable de redundancia muestra una diferencia significativa entre noticias falsas y reales tras la corrección de Holm (α = 0,05)."
           if not sig else f"- Variables con diferencia significativa tras Holm: {', '.join(sig)}."),
          f"- Los AUC univariados quedan entre {f(min(r['auc_univariado_falsa'] for r in a.values()), 3)} y {f(max(r['auc_univariado_falsa'] for r in a.values()), 3)}, es decir, en torno al azar (0,5).",
          "- En este corpus, la redundancia semántica medida con `hiiamsid` **no discrimina la veracidad** por sí sola. Puede seguir usándose como descriptor del estilo del texto (cuánto reitera una noticia), pero no como indicador de desinformación.",
          "- Este resultado contradice las afirmaciones anteriores de la tesis basadas en estratos ajustados con etiquetas invertidas (por ejemplo, «79,2 % de falsas en la banda de hiper-redundancia»).", "",
          "## Archivos", "",
          "- `Dataset/redundancia_corpus_hiiamsid.csv`: métricas por noticia (similitudes, densidades por umbral, similitud entre oraciones consecutivas, `is_unique_text`).",
          "- `Dataset/benchmarks_externos/`: STS-B y PAWS-X en español usados para calibrar y validar.",
          "- `Reportes/metricas_redundancia_hiiamsid.json`: todas las cifras de este reporte."]
    salida = cfg.REDUNDANCIA_REPORTES_DIR / "Reporte_Redundancia_hiiamsid.md"
    salida.write_text("\n".join(L) + "\n", encoding='utf-8')
    print(f"[OK] {salida}")


if __name__ == '__main__':
    main()
