"""
Genera Reportes/Reporte_3_Modelos_FakeNews.md y sus figuras a partir de Reportes/metricas_3_modelos_fakenews.json
y Reportes/predicciones_3_modelos_2604.csv (producidos por run_3_modelos_fakenews.py). Todas las cifras se leen
de esos archivos y del corpus base; no hay valores escritos a mano.
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config_proyecto.py").exists())))
import config_proyecto as cfg
import graficos_comunes as gc

REP = cfg.LITERATURA_REPORTES_DIR
IMG = cfg.LITERATURA_IMG_DIR
ORDEN = ['RoBERTa_BNE_Narrativaai', 'SaBERT_Verificado', 'Spanish_FakeNews_Juanillaberia']
CLAVE_CSV = {'RoBERTa_BNE_Narrativaai': 'm1_roberta_bne', 'SaBERT_Verificado': 'm2_sabert', 'Spanish_FakeNews_Juanillaberia': 'm3_juanillaberia'}
NUM = {'RoBERTa_BNE_Narrativaai': 'M1', 'SaBERT_Verificado': 'M2', 'Spanish_FakeNews_Juanillaberia': 'M3'}
COLOR = {k: gc.SERIE[i] for i, k in enumerate(ORDEN)}   # color fijo por modelo
ID2LABEL = {'RoBERTa_BNE_Narrativaai': '`{0: REAL, 1: FAKE}`', 'SaBERT_Verificado': '`{0: False, 1: True}`',
            'Spanish_FakeNews_Juanillaberia': 'No definido'}
VERIFICACION = {'RoBERTa_BNE_Narrativaai': 'Declarado en el config', 'SaBERT_Verificado': '`False` = la noticia es falsa'}
FIGURAS = ['metricas_globales_3_modelos.png', 'matrices_confusion_3_modelos.png', 'curvas_roc_pr_3_modelos.png',
           'distribucion_probabilidad_3_modelos.png', 'errores_3_modelos.png', 'desempeno_por_fuente_3_modelos.png',
           'latencia_3_modelos.png']


def f(x, d=4):
    return '—' if x is None else f"{x:.{d}f}".replace('.', ',')


def pct(x, d=2):
    return '—' if x is None else f"{100 * x:.{d}f}".replace('.', ',') + " %"


def n(x):
    return f"{int(x):,}".replace(',', '.')


def nombre(J, k):
    return f"{NUM[k]} {J['modelos'][k]['Etiqueta_corta']}"


def figuras(J, P, y):
    from sklearn.metrics import roc_curve, precision_recall_curve
    plt = gc.pyplot()
    M = J['modelos']
    etq = {k: nombre(J, k) for k in ORDEN}

    # 1. Métricas globales
    met = [('Accuracy', 'Exactitud'), ('F1_Macro', 'F1 macro'), ('F1-Score', 'F1 (falsa)'), ('Recall', 'Exhaustividad (falsa)'),
           ('AUC', 'ROC-AUC'), ('AP', 'AP')]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x, w = np.arange(len(met)), 0.26
    for j, k in enumerate(ORDEN):
        v = [M[k][m] for m, _ in met]
        b = ax.bar(x + (j - 1) * w, v, w, color=COLOR[k], label=etq[k])
        for r, val in zip(b, v):
            ax.text(r.get_x() + r.get_width() / 2, val + 0.01, f"{val:.2f}".replace('.', ','), ha='center', va='bottom', fontsize=7.5, color=gc.TINTA_2)
    ax.set_xticks(x, [t for _, t in met])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Valor')
    gc.coma(ax, 'y', 1)
    ax.grid(axis='x', visible=False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    ax.set_title(f"Métricas globales en el corpus base (N = {n(len(y))}; umbral 0,5)")
    gc.guardar(fig, IMG / FIGURAS[0])

    # 2. Matrices de confusión con conteo y % por fila
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    for ax, k in zip(axes, ORDEN):
        cm = np.array(M[k]['CM'])
        fila = cm / cm.sum(1, keepdims=True)
        ax.imshow(fila, cmap=gc.secuencial(), vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{n(cm[i, j])}\n({pct(fila[i, j], 1)})", ha='center', va='center', fontsize=10,
                        color='white' if fila[i, j] > 0.55 else gc.TINTA)
        ax.set_xticks([0, 1], ['Predicha real', 'Predicha falsa'])
        ax.set_yticks([0, 1], ['Real', 'Falsa'])
        ax.grid(False)
        ax.set_title(etq[k], fontsize=10)
    fig.suptitle('Matrices de confusión (filas: etiqueta verdadera; % sobre la fila)', x=0.01, ha='left', fontsize=11)
    fig.tight_layout()
    gc.guardar(fig, IMG / FIGURAS[1])

    # 3. ROC y PR
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for k in ORDEN:
        p = P[f"prob_{CLAVE_CSV[k]}"].values
        fpr, tpr, _ = roc_curve(y, p)
        a1.plot(fpr, tpr, color=COLOR[k], label=f"{etq[k]} (AUC {f(M[k]['AUC'], 3)})")
        pr, rc, _ = precision_recall_curve(y, p)
        a2.plot(rc, pr, color=COLOR[k], label=f"{etq[k]} (AP {f(M[k]['AP'], 3)})")
    a1.plot([0, 1], [0, 1], ls=':', color=gc.TINTA_SUAVE, lw=1.2, label='Azar')
    prev = y.mean()
    a2.axhline(prev, ls=':', color=gc.TINTA_SUAVE, lw=1.2, label=f"Prevalencia de falsas ({f(prev, 3)})")
    a1.set(xlabel='Tasa de falsos positivos', ylabel='Tasa de verdaderos positivos', title='Curvas ROC')
    a2.set(xlabel='Exhaustividad (falsa)', ylabel='Precisión (falsa)', title='Curvas precisión-exhaustividad', ylim=(0.3, 1.02))
    for a in (a1, a2):
        gc.coma(a, 'x', 1)
        gc.coma(a, 'y', 1)
        a.legend(loc='lower right' if a is a1 else 'lower left', fontsize=8.5)
    gc.guardar(fig, IMG / FIGURAS[2])

    # 4. Distribución de P(falsa) por clase
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    bins = np.linspace(0, 1, 26)
    for ax, k in zip(axes, ORDEN):
        p = P[f"prob_{CLAVE_CSV[k]}"].values
        for c, et in [(0, 'Reales'), (1, 'Falsas')]:
            ax.hist(p[y == c], bins=bins, color=gc.CLASE[c], alpha=0.55, label=f"{et} (n = {n((y == c).sum())})")
        ax.axvline(0.5, color=gc.TINTA_2, ls='--', lw=1.2, label='Umbral 0,5')
        ax.set_title(etq[k], fontsize=10)
        ax.set_xlabel('P(falsa)')
        gc.coma(ax, 'x', 1)
    axes[0].set_ylabel('Noticias')
    fig.suptitle('Distribución de la probabilidad de «falsa» por clase verdadera', x=0.01, ha='left', fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=9)
    gc.guardar(fig, IMG / FIGURAS[3])

    # 5. Falsos positivos y falsos negativos
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x, w = np.arange(2), 0.26
    for j, k in enumerate(ORDEN):
        (tn, fp), (fn, tp) = M[k]['CM']
        v = [fp / (tn + fp), fn / (fn + tp)]
        b = ax.bar(x + (j - 1) * w, v, w, color=COLOR[k], label=etq[k])
        for r, val, cnt in zip(b, v, [fp, fn]):
            ax.text(r.get_x() + r.get_width() / 2, val + 0.01, f"{pct(val, 1)}\n({n(cnt)})", ha='center', va='bottom', fontsize=7.5, color=gc.TINTA_2)
    ax.set_xticks(x, ['Tasa de falsos positivos\n(reales marcadas como falsas)', 'Tasa de falsos negativos\n(falsas que pasan como reales)'])
    ax.set_ylim(0, max(0.7, ax.get_ylim()[1] + 0.08))
    ax.set_ylabel('Proporción')
    gc.coma(ax, 'y', 1)
    ax.grid(axis='x', visible=False)
    ax.legend(fontsize=8.5, loc='upper left')
    ax.set_title('Errores de cada modelo con el umbral 0,5')
    gc.guardar(fig, IMG / FIGURAS[4])

    # 6. Por fuente (N >= 30)
    F = {s: e for s, e in J['por_fuente'].items() if e['N'] >= 30}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    x, w = np.arange(len(F)), 0.26
    for j, k in enumerate(ORDEN):
        for a, m in [(a1, 'F1_Macro'), (a2, 'FNR_Falsa')]:
            v = [F[s]['modelos'][k][m] for s in F]
            b = a.bar(x + (j - 1) * w, v, w, color=COLOR[k], label=etq[k])
            for r, val in zip(b, v):
                a.text(r.get_x() + r.get_width() / 2, val + 0.01, f"{val:.2f}".replace('.', ','), ha='center', va='bottom', fontsize=7.5, color=gc.TINTA_2)
    et_f = [f"{s}\n{F[s]['Region']} (N = {n(F[s]['N'])})" for s in F]
    a1.set(title='F1 macro por fuente', ylim=(0, 1.08))
    a2.set(title='Tasa de falsos negativos por fuente (↓ mejor)', ylim=(0, 1.08))
    for a in (a1, a2):
        a.set_xticks(x, et_f, fontsize=8.5)
        gc.coma(a, 'y', 1)
        a.grid(axis='x', visible=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=9)
    gc.guardar(fig, IMG / FIGURAS[5])

    # 7. Latencia
    fig, ax = plt.subplots(figsize=(7, 3.2))
    v = [M[k]['Latencia (ms)'] for k in ORDEN]
    b = ax.barh([etq[k] for k in ORDEN], v, color=[COLOR[k] for k in ORDEN], height=0.55)
    for r, val in zip(b, v):
        ax.text(val + max(v) * 0.01, r.get_y() + r.get_height() / 2, f"{val:.1f} ms".replace('.', ','), va='center', fontsize=9, color=gc.TINTA_2)
    ax.invert_yaxis()
    ax.set_xlabel('Milisegundos por noticia (corrida que generó la caché; hardware no registrado)')
    ax.set_xlim(0, max(v) * 1.15)
    ax.grid(axis='y', visible=False)
    ax.set_title('Latencia de inferencia')
    gc.guardar(fig, IMG / FIGURAS[6])


def main():
    J = json.load(open(cfg.METRICAS_3_MODELOS_JSON, encoding='utf-8'))
    P = pd.read_csv(cfg.PREDICCIONES_3_MODELOS_CSV)
    meta, M = J['metadata'], J['modelos']
    y = P['y_true'].values.astype(int)
    base = pd.read_excel(cfg.CORPUS_BASE_XLSX)
    assert np.array_equal(base['class'].apply(cfg.etiqueta_binaria).astype(int).values, y), "Predicciones desalineadas con el corpus base"
    vc = base['Text'].astype(str).value_counts()
    grupos_dup = int((vc > 1).sum())
    figuras(J, P, y)

    por_auc = sorted(ORDEN, key=lambda k: -M[k]['AUC'])
    mejor, peor = por_auc[0], por_auc[-1]
    mas_exh = max(ORDEN, key=lambda k: M[k]['Recall'])
    N, nr, nf = meta['total_registros'], meta['clase_real_0'], meta['clase_falsa_1']

    def esp(k):
        (tn, fp), _ = M[k]['CM']
        return tn / (tn + fp)

    L = ["# Evaluación comparativa de 3 clasificadores de noticias falsas en español (corpus base, N = " + n(N) + ")", "",
         "**Generado por:** `Codigos/generar_reporte_3_modelos.py` (lo llama al final `Codigos/run_3_modelos_fakenews.py`) a partir de "
         "[`metricas_3_modelos_fakenews.json`](metricas_3_modelos_fakenews.json) y [`predicciones_3_modelos_2604.csv`](predicciones_3_modelos_2604.csv)  ",
         f"**Dataset:** `{meta['dataset_path']}`  ",
         "**Figuras:** [`Imagenes/`](Imagenes/)", "",
         "> Todas las cifras se leen de esos archivos. Las etiquetas del corpus base coinciden, noticia por noticia, con las del corpus ampliado "
         "vigente (la corrección de Omdena del 2026-10-24 no afecta a este corpus, que no contiene noticias de Omdena).", "",
         "## 1. Metodología", "",
         "### 1.1 Tarea y convención de etiquetas", "",
         "Clasificación binaria con la noticia falsa como clase positiva (0 = real, 1 = falsa). Cada modelo produce $P(\\text{Falsa} \\mid x)$ y la "
         "predicción usa una regla común: $\\hat{y} = 1$ si $P(\\text{Falsa} \\mid x) \\ge 0{,}5$. Los textos se truncan a "
         f"{meta['truncamiento_tokens']} tokens.", "",
         "### 1.2 Corpus", "",
         "| Clase | Etiqueta | N | % |", "|:---|:---:|---:|---:|",
         f"| Noticia real | 0 | {n(nr)} | {pct(nr / N)} |",
         f"| Noticia falsa | 1 | {n(nf)} | {pct(nf / N)} |",
         f"| **Total** | | **{n(N)}** | **100,00 %** |", "",
         "| Repositorio | Región | N | Reales (0) | Falsas (1) |", "|:---|:---|---:|---:|---:|"]
    for s, e in J['por_fuente'].items():
        L.append(f"| {s} | {e['Region'].replace('Sin region', 'Sin región')} | {n(e['N'])} | {n(e['Real_0'])} | {n(e['Falsa_1'])} |")
    L += ["", "### 1.3 Modelos evaluados", "",
          "Los tres modelos se usan en inferencia, sin ajuste adicional, con la revisión fijada en `config_proyecto.py`.", "",
          "| | " + " | ".join(NUM[k] for k in ORDEN) + " |", "|:---|:---|:---|:---|",
          "| Identificador | " + " | ".join(f"`{M[k]['ID']}`" for k in ORDEN) + " |",
          "| Revisión | " + " | ".join(f"`{M[k]['Revision'][:10]}`" for k in ORDEN) + " |",
          "| Nombre en este reporte | " + " | ".join(M[k]['Etiqueta_corta'] for k in ORDEN) + " |",
          "| Arquitectura | " + " | ".join(M[k]['Ficha_Tecnica']['Arquitectura'] for k in ORDEN) + " |",
          "| Modelo base | " + " | ".join(f"`{M[k]['Ficha_Tecnica']['Modelo_base']}`" for k in ORDEN) + " |",
          "| Capas / dimensión / cabezas | " + " | ".join(f"{M[k]['Ficha_Tecnica']['Capas']} / {n(M[k]['Ficha_Tecnica']['Dim_oculta'])} / "
                                                         f"{M[k]['Ficha_Tecnica']['Cabezas']}" for k in ORDEN) + " |",
          "| Vocabulario | " + " | ".join(n(M[k]['Ficha_Tecnica']['Vocabulario']) for k in ORDEN) + " |",
          "| Datos de ajuste declarados en la tarjeta | " + " | ".join(M[k]['Ficha_Tecnica']['Datos_ajuste_declarados'] for k in ORDEN) + " |", "",
          "> **Corrección de nomenclatura.** M1 aparecía en versiones anteriores como «BETO Narrativaai». Según su `config.json`, es un RoBERTa-large "
          "preentrenado con el corpus de la Biblioteca Nacional de España (BNE), no un modelo BETO.", "",
          "## 2. Correspondencia entre las salidas de cada modelo y las clases", "",
          "Se verificó con el `config.json` de cada modelo (consulta del 2026-10-24) qué posición de la salida softmax corresponde a la clase falsa.", "",
          "| Modelo | `id2label` | Índice usado como P(Falsa) | Verificación |", "|:---|:---|:---:|:---|"]
    for k in ORDEN:
        ver = VERIFICACION.get(k, f"Empírica: con el índice {M[k]['Indice_Clase_Falsa']}, ROC-AUC = {f(M[k]['AUC'])}; con el otro índice sería {f(1 - M[k]['AUC'])}")
        L.append(f"| {nombre(J, k)} | {ID2LABEL[k]} | {M[k]['Indice_Clase_Falsa']} | {ver} |")
    L += ["", "## 3. Resultados globales", "",
          f"![Métricas globales](Imagenes/{FIGURAS[0]})", "",
          f"### 3.1 Métricas de la clase falsa (N = {n(N)})", "",
          "| Modelo | Exactitud | Precisión | Exhaustividad | F1 | ROC-AUC | AP (PR-AUC) | Tiempo total | Latencia |",
          "|:---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in por_auc:
        m = M[k]
        L.append(f"| {nombre(J, k)} | {f(m['Accuracy'])} | {f(m['Precision'])} | {f(m['Recall'])} | {f(m['F1-Score'])} | {f(m['AUC'])} | {f(m['AP'])} | "
                 f"{f(m['Tiempo Total (s)'], 2)} s | {f(m['Latencia (ms)'], 2)} ms |")
    L += ["", "### 3.2 Métricas promedio por clase (macro)", "",
          "| Modelo | Precisión macro | Exhaustividad macro | F1 macro |", "|:---|---:|---:|---:|"]
    for k in por_auc:
        L.append(f"| {nombre(J, k)} | {f(M[k]['Precision_Macro'])} | {f(M[k]['Recall_Macro'])} | {f(M[k]['F1_Macro'])} |")
    L += ["", "### 3.3 Matriz de confusión y errores", "",
          f"![Matrices de confusión](Imagenes/{FIGURAS[1]})", "",
          "| Modelo | TN | FP | FN | TP | Especificidad | Tasa de falsos negativos |", "|:---|---:|---:|---:|---:|---:|---:|"]
    for k in por_auc:
        (tn, fp), (fn, tp) = M[k]['CM']
        L.append(f"| {nombre(J, k)} | {n(tn)} | {n(fp)} | {n(fn)} | {n(tp)} | {pct(tn / (tn + fp))} | {pct(fn / (fn + tp))} |")
    L += ["", f"![Errores por modelo](Imagenes/{FIGURAS[4]})", "",
          "La figura separa los dos tipos de error: noticias reales marcadas como falsas (falsos positivos) y noticias falsas que pasan como reales "
          "(falsos negativos), con el conteo entre paréntesis.", "",
          "### 3.4 Curvas ROC y precisión-exhaustividad", "",
          f"![Curvas ROC y PR](Imagenes/{FIGURAS[2]})", "",
          f"La línea punteada de la curva PR es la prevalencia de la clase falsa ({n(nf)} / {n(N)} = {f(nf / N)}), la precisión esperada de un clasificador al azar.", "",
          "### 3.5 Distribución de las probabilidades", "",
          f"![Distribución de P(falsa)](Imagenes/{FIGURAS[3]})", "",
          "Un modelo que separa bien las clases concentra las reales cerca de 0 y las falsas cerca de 1. La posición de cada distribución respecto "
          "del umbral 0,5 explica el perfil de error de la sección anterior.", "",
          "## 4. Interpretación de los resultados globales", ""]
    (tn2, fp2), (fn2, tp2) = M[mejor]['CM']
    (tn3, fp3), (fn3, tp3) = M[mas_exh]['CM']
    (tnp, fpp), (fnp, tpp) = M[peor]['CM']
    L += [f"1. **{M[mejor]['Etiqueta_corta']} obtiene la mejor capacidad de ordenamiento** (ROC-AUC {f(M[mejor]['AUC'])}; AP {f(M[mejor]['AP'])}) "
          f"y una exactitud de {pct(M[mejor]['Accuracy'])}. Su perfil de error es asimétrico: produce {n(fp2)} falsos positivos sobre {n(nr)} noticias "
          f"reales (especificidad {pct(esp(mejor))}), pero no detecta {n(fn2)} de las {n(nf)} noticias falsas (exhaustividad {pct(M[mejor]['Recall'])}).",
          f"2. **{M[mas_exh]['Etiqueta_corta']} prioriza la detección de la clase falsa.** Encuentra {n(tp3)} de las {n(nf)} noticias falsas "
          f"(exhaustividad {pct(M[mas_exh]['Recall'])}) a costa de {n(fp3)} falsos positivos sobre noticias reales (especificidad {pct(esp(mas_exh))}). "
          f"Su ROC-AUC de {f(M[mas_exh]['AUC'])} indica que un umbral distinto de 0,5 podría equilibrar mejor ambos errores.",
          f"3. **{M[peor]['Etiqueta_corta']} tiene el desempeño global más bajo** (ROC-AUC {f(M[peor]['AUC'])}), con {n(fpp)} falsos positivos y "
          f"{n(fnp)} falsos negativos.", ""]

    L += ["## 5. Desglose por región y por repositorio de origen", "",
          "### 5.1 Por región", "",
          "| Región (N) | " + " | ".join(nombre(J, k) for k in ORDEN) + " |", "|:---|:---|:---|:---|"]
    for r, e in J['por_region'].items():
        if e['N'] < 30:
            continue
        L.append(f"| {r} ({n(e['N'])}) | " + " | ".join(
            f"Exact. {f(e['modelos'][k]['Accuracy'])} · F1 macro {f(e['modelos'][k]['F1_Macro'])} · AUC {f(e['modelos'][k]['ROC_AUC'])}" for k in ORDEN) + " |")
    L += ["", "La única fuente latinoamericana del corpus base es FakeDeS; la comparación regional en un corpus más amplio está en "
          "[`../../SABERT_Evaluacion/`](../../SABERT_Evaluacion/).", "",
          "### 5.2 Por repositorio", "",
          f"![Desempeño por fuente](Imagenes/{FIGURAS[5]})", "",
          "Exactitud / F1 macro / ROC-AUC / tasa de falsos negativos de la clase falsa:", "",
          "| Repositorio (N) | " + " | ".join(nombre(J, k) for k in ORDEN) + " |", "|:---|:---|:---|:---|"]
    F = J['por_fuente']
    for s, e in F.items():
        if e['N'] < 30:
            continue
        L.append(f"| {s}, {e['Region']} ({n(e['N'])}) | " + " | ".join(
            f"{f(e['modelos'][k]['Accuracy'])} / {f(e['modelos'][k]['F1_Macro'])} / {f(e['modelos'][k]['ROC_AUC'])} / {pct(e['modelos'][k]['FNR_Falsa'])}"
            for k in ORDEN) + " |")
    pequenas = {s: e for s, e in F.items() if e['N'] < 30}
    for s, e in pequenas.items():
        aciertos = ", ".join(f"{NUM[k]} acierta {int(np.trace(np.array(e['modelos'][k]['CM'])))}" for k in ORDEN)
        L += ["", f"Los {n(e['N'])} registros de {s} ({n(e['Falsa_1'])} falsos) no permiten calcular métricas interpretables por fuente ({aciertos})."]
    L += ["", "Matrices de confusión por repositorio `[TN, FP, FN, TP]`:", "",
          "| Repositorio | " + " | ".join(nombre(J, k) for k in ORDEN) + " |", "|:---|:---|:---|:---|"]
    for s, e in F.items():
        if e['N'] < 30:
            continue
        L.append(f"| {s} | " + " | ".join(", ".join(n(v) for fila in e['modelos'][k]['CM'] for v in fila) for k in ORDEN) + " |")

    fk = next(s for s in F if s.startswith('FakeDeS'))
    fr = next(s for s in F if s.startswith('Freiren'))
    sab = 'SaBERT_Verificado'
    m1 = 'RoBERTa_BNE_Narrativaai'
    cm_s = F[fk]['modelos'][sab]['CM']
    acc_rango = [F[s]['modelos']['Spanish_FakeNews_Juanillaberia']['Accuracy'] for s in F if F[s]['N'] >= 30]
    L += ["", "### 5.3 Lectura del desglose", "",
          f"1. **SaBERT pierde desempeño en FakeDeS**, la única fuente latinoamericana del corpus base: su exactitud baja de "
          f"{pct(F[fr]['modelos'][sab]['Accuracy'])} en Freiren a {pct(F[fk]['modelos'][sab]['Accuracy'])}, y deja de detectar {n(cm_s[1][0])} de las "
          f"{n(F[fk]['Falsa_1'])} noticias falsas (tasa de falsos negativos {pct(F[fk]['modelos'][sab]['FNR_Falsa'])}). Conserva una especificidad alta "
          f"({n(cm_s[0][1])} falsos positivos sobre {n(F[fk]['Real_0'])} noticias reales).",
          f"2. **M3 mantiene un desempeño más estable entre fuentes** (exactitud entre {pct(min(acc_rango))} y {pct(max(acc_rango))}), aunque en Freiren "
          f"clasifica como falsas {n(F[fr]['modelos']['Spanish_FakeNews_Juanillaberia']['CM'][0][1])} de las {n(F[fr]['Real_0'])} noticias reales.",
          f"3. **El resultado de M1 en FakeDeS no debe leerse como robustez regional.** Según su tarjeta, M1 se ajustó con FakeDeS, por lo que su exactitud "
          f"en esa fuente ({pct(F[fk]['modelos'][m1]['Accuracy'])}) puede reflejar exposición previa a esos datos.",
          f"4. **Tamaño muestral.** FakeDeS aporta solo {n(F[fk]['N'])} noticias, el {pct(F[fk]['N'] / N)} del corpus base. Esta limitación motivó la "
          "auditoría sobre el corpus ampliado (N = 4.418) documentada en [`../../SABERT_Evaluacion/`](../../SABERT_Evaluacion/).", "",
          "### 5.4 Hipótesis explicativas (no verificadas en este experimento)", "",
          "La caída de SaBERT en FakeDeS es compatible con dos explicaciones que este benchmark **no permite distinguir**:", "",
          "- **Cambio de dominio:** diferencias léxicas, temáticas y de registro entre la prensa española y la latinoamericana.",
          "- **Aprendizaje de atajos:** dependencia de entidades o temas sobrerrepresentados en sus datos de entrenamiento, cuya composición exacta no "
          "está documentada públicamente.", "",
          "Para contrastarlas hace falta el control por enmascaramiento y la evaluación sobre el corpus ampliado.", "",
          "## 6. Costo de inferencia", "",
          f"![Latencia](Imagenes/{FIGURAS[6]})", ""]
    t2 = M[sab]['Tiempo Total (s)']
    L += ["## 7. Limitaciones y advertencias", "",
          "1. **Posible contaminación de datos de entrenamiento.** M1 declara FakeDeS como dato de ajuste, y la composición de los datos de M2 y M3 no se "
          "conoce con detalle, por lo que pueden solaparse con el corpus base. Las métricas sobre este corpus deben interpretarse como una cota optimista.",
          f"2. **Duplicados.** El corpus base contiene {n(grupos_dup)} grupos de textos duplicados ({n(meta['textos_unicos'])} textos únicos). Las métricas "
          f"de este reporte se calculan sobre los {n(N)} registros sin deduplicar.",
          "3. **Tiempos de inferencia.** Provienen de la corrida que generó la caché `predictions_cache_2604.npz` y dependen del hardware de esa ejecución, "
          "que no quedó registrado." + (f" El tiempo de M2 ({f(t2, 2)} s) aparece en la caché con solo dos decimales, a diferencia de los otros dos; "
                                        "conviene confirmarlo en una nueva ejecución." if round(t2, 2) == t2 else ""),
          "4. **Revisión de la caché.** La caché no guarda la revisión del modelo con que se generó; se asume la fijada en `config_proyecto.py`.",
          "5. **Umbral fijo.** Todas las métricas puntuales usan el umbral 0,5. ROC-AUC y AP son independientes del umbral.", "",
          "## 8. Conclusiones", "",
          f"1. En el corpus base, {M[mejor]['Etiqueta_corta']} presenta el mejor desempeño global (exactitud {f(M[mejor]['Accuracy'])}; F1 de la clase "
          f"falsa {f(M[mejor]['F1-Score'])}; F1 macro {f(M[mejor]['F1_Macro'])}; ROC-AUC {f(M[mejor]['AUC'])}), con alta especificidad y una exhaustividad "
          "moderada para la clase falsa.",
          f"2. {M[mas_exh]['Etiqueta_corta']} alcanza la mayor exhaustividad para la clase falsa ({f(M[mas_exh]['Recall'])}) a costa de una especificidad "
          f"baja ({f(esp(mas_exh))}).",
          f"3. {M[peor]['Etiqueta_corta']} obtiene el ROC-AUC más bajo de los tres ({f(M[peor]['AUC'])}).",
          f"4. El desempeño de SaBERT cae en la única fuente latinoamericana del corpus base (exactitud {f(F[fk]['modelos'][sab]['Accuracy'])}). Por su "
          "tamaño reducido, esta evidencia se complementa con la auditoría sobre el corpus ampliado en `SABERT_Evaluacion/`.", "",
          "## 9. Reproducibilidad", "",
          "```bash",
          "# Desde la raíz del repositorio, con el entorno virtual activado",
          "python Modelos_Individuales/Clasificar_fake/Modelos_Literatura/Codigos/run_3_modelos_fakenews.py   # métricas + reporte",
          "python Modelos_Individuales/Clasificar_fake/Modelos_Literatura/Codigos/generar_reporte_3_modelos.py  # solo reporte y figuras",
          "```", "",
          "- Si existe `Codigos/predictions_cache_2604.npz`, `run_3_modelos_fakenews.py` recalcula las métricas sin ejecutar los modelos (no requiere `torch` "
          "ni `transformers`). Si se borra la caché, descarga los tres modelos en la revisión fijada en `config_proyecto.py` y ejecuta la inferencia.",
          "- Para copiar las figuras a la carpeta de la tesis, defina la variable de entorno `TESIS_IMG_DIR`.", ""]
    (REP / 'Reporte_3_Modelos_FakeNews.md').write_text("\n".join(L), encoding='utf-8')
    cfg.exportar_figuras_tesis([IMG / x for x in FIGURAS], "Fakenews3model")
    print(f"[OK] {REP / 'Reporte_3_Modelos_FakeNews.md'}")


if __name__ == '__main__':
    main()
