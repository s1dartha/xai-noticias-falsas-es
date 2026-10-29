"""
Genera Reportes/Reporte_Evaluacion_SaBERT_Domain_Shift.md y sus figuras.

Entradas:
  Dataset/predicciones_sabert_ampliado_4418.csv      P(falsa) de SaBERT por noticia (umbral 0,5)
  Reportes/auditoria_censura_vs_domain_shift.json    máscaras y contrafácticos (auditoria_censura_vs_domain_shift.py)
Salidas:
  Reportes/metricas_sabert_dataset_ampliado_4418.json métricas globales, por región, por corpus de origen y por fuente
  Reportes/Imagenes/*.png y el reporte .md
Todas las cifras se calculan de esos archivos; no hay valores escritos a mano.
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config_proyecto.py").exists())))
import config_proyecto as cfg
import graficos_comunes as gc
from auditoria_censura_vs_domain_shift import metricas

REP = cfg.SABERT_REPORTES_DIR
IMG = cfg.SABERT_IMG_DIR
METRICAS_JSON = REP / 'metricas_sabert_dataset_ampliado_4418.json'
AUDITORIA_JSON = REP / 'auditoria_censura_vs_domain_shift.json'
N_BOOT = 1000
CORTO = {'Freiren Unified Corpus (España)': 'Unified (España)', 'Edds Fixed (España)': 'Edds (España)',
         'Omdena LATAM (Politics Fake News Detector)': 'Omdena (Am. Latina)', 'MEX-A3T (UNAM / IPN)': 'MEX-A3T (México)',
         'FakeDeS 2021 (LatAm)': 'FakeDeS 2021 (Am. Latina)'}
GRUPO = {'global': 'Global', 'espana': 'España', 'america_latina': 'América Latina', 'omdena': 'Omdena (sin máscaras)',
         'mex_fakedes': 'MEX-A3T + FakeDeS', 'mex_fakedes_con_mascara': 'MEX-A3T + FakeDeS, textos con máscara',
         'mex_fakedes_sin_mascara': 'MEX-A3T + FakeDeS, textos sin máscara'}


def f(x, d=3):
    return '—' if x is None else f"{x:.{d}f}".replace('.', ',')


def pct(x, d=1):
    return '—' if x is None else f"{100 * x:.{d}f}".replace('.', ',') + " %"


def pp(x, d=1):
    return f"{100 * x:+.{d}f}".replace('.', ',') + " pp"


def mil(n):
    return f"{n:,}".replace(',', '.')


def ic_f1(y, p, rng):
    from sklearn.metrics import f1_score
    pred = (p >= 0.5).astype(int)
    b = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        b.append(f1_score(y[i], pred[i], average='macro', zero_division=0))
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def calcular_metricas(df):
    rng = np.random.default_rng(cfg.SEED)
    y, p = df['clase_num'].values.astype(int), df['prob_fake'].values.astype(float)

    def m(mask):
        r = metricas(y[mask], p[mask])
        r['f1_macro_ic95'] = ic_f1(y[mask], p[mask], rng)
        return r

    todo = np.ones(len(df), bool)
    out = {'metadata': {'modelo': f"{cfg.MODELOS_HF['M2_SaBERT']['id']}@{cfg.MODELOS_HF['M2_SaBERT']['revision']}",
                        'umbral': 0.5, 'n_bootstrap': N_BOOT, 'predicciones': cfg.PREDICCIONES_SABERT_4418_CSV.name},
           'global': m(todo),
           'regiones': {r: m((df['region'] == r).values) for r in ['España', 'América Latina']},
           'corpus_origen': {d: {'region': g['region'].iloc[0], **m((df['dataset_origen'] == d).values)}
                             for d, g in df.groupby('dataset_origen')},
           'fuentes': {s: m((df['Fuente'] == s).values) for s in df['Fuente'].unique()}}
    METRICAS_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    return out


def figuras(df, M, A):
    from sklearn.metrics import roc_curve, precision_recall_curve
    plt = gc.pyplot()
    IMG.mkdir(parents=True, exist_ok=True)
    fig_out = {}
    orden = sorted(M['corpus_origen'], key=lambda d: (M['corpus_origen'][d]['region'] != 'España', -M['corpus_origen'][d]['f1_macro']))
    col_reg = {'España': gc.SERIE[0], 'América Latina': gc.SERIE[1]}

    # 1. F1 macro e IC por región y por corpus de origen
    filas = [('Global', M['global'], gc.TINTA_2)] + [(r, M['regiones'][r], col_reg[r]) for r in ['España', 'América Latina']] + \
            [(CORTO.get(d, d), M['corpus_origen'][d], col_reg[M['corpus_origen'][d]['region']]) for d in orden]
    # posiciones: Global | España, América Latina | corpus de origen (separados por un hueco)
    pos, yy = [], 0.0
    for i in range(len(filas)):
        if i in (1, 3):
            yy += 0.7
        pos.append(-yy)
        yy += 1
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for i, (n, r, c) in enumerate(filas):
        yv = pos[i]
        lo, hi = r['f1_macro_ic95']
        ax.plot([lo, hi], [yv, yv], color=c, lw=2, solid_capstyle='round')
        ax.plot(r['f1_macro'], yv, 'o', color=c, ms=8, mec='white', mew=1.5)
        ax.text(hi + 0.012, yv, f"{f(r['f1_macro'])}  (n = {mil(r['n'])})", va='center', fontsize=8.5, color=gc.TINTA_2)
        ax.text(-0.01, yv, n, va='center', ha='right', fontsize=9, color=gc.TINTA, transform=ax.get_yaxis_transform())
    ax.set_yticks([])
    ax.axvline(0.5, color=gc.TINTA_SUAVE, ls=':', lw=1)
    ax.set_xlim(0.3, 1.08)
    ax.set_xlabel('F1 macro con umbral 0,5 (punto) e IC 95 % bootstrap (línea)')
    ax.set_title('SaBERT por región y corpus de origen')
    gc.coma(ax, 'x', 1)
    ax.grid(axis='y', visible=False)
    fig_out['f1'] = gc.guardar(fig, IMG / 'f1_macro_region_corpus.png')

    # 2. Matrices de confusión por región (% por fila)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for ax, r in zip(axes, ['España', 'América Latina']):
        x = M['regiones'][r]
        cm = np.array([[x['tn'], x['fp']], [x['fn'], x['tp']]])
        fr = cm / cm.sum(1, keepdims=True)
        ax.imshow(fr, cmap=gc.secuencial(), vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mil(int(cm[i, j]))}\n({pct(fr[i, j], 0)})", ha='center', va='center', fontsize=10,
                        color='white' if fr[i, j] > 0.55 else gc.TINTA)
        ax.set_xticks([0, 1], ['Predicha real', 'Predicha falsa'])
        ax.set_yticks([0, 1], ['Real', 'Falsa'])
        ax.set_title(f"{r} (n = {mil(x['n'])})")
        ax.grid(False)
    fig.tight_layout()
    fig_out['cm'] = gc.guardar(fig, IMG / 'matrices_confusion_region.png')

    # 3. Curvas ROC y PR por corpus de origen
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    estilos = {'España': '-', 'América Latina': '--'}
    for k, d in enumerate(orden):
        g = df[df['dataset_origen'] == d]
        r = M['corpus_origen'][d]
        fpr, tpr, _ = roc_curve(g['clase_num'], g['prob_fake'])
        pr, rc, _ = precision_recall_curve(g['clase_num'], g['prob_fake'])
        c = gc.SERIE[k]
        axes[0].plot(fpr, tpr, color=c, ls=estilos[r['region']], label=f"{CORTO.get(d, d)}: AUC {f(r['roc_auc'])}")
        axes[1].plot(rc, pr, color=c, ls=estilos[r['region']], label=f"{CORTO.get(d, d)}: AP {f(r['ap'])}")
    axes[0].plot([0, 1], [0, 1], color=gc.TINTA_SUAVE, ls=':', lw=1)
    axes[0].set_xlabel('Tasa de falsos positivos'); axes[0].set_ylabel('Tasa de verdaderos positivos (recall de falsas)')
    axes[0].set_title('Curvas ROC')
    axes[1].set_xlabel('Recall de falsas'); axes[1].set_ylabel('Precisión de falsas')
    axes[1].set_title('Curvas precisión-recall')
    for ax in axes:
        ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        gc.coma(ax, 'x', 1); gc.coma(ax, 'y', 1)
    fig.suptitle('Línea continua: corpus de España · discontinua: corpus de América Latina', x=0.01, ha='left', fontsize=9, color=gc.TINTA_2)
    fig.tight_layout()
    fig_out['roc'] = gc.guardar(fig, IMG / 'curvas_roc_pr_corpus.png')

    # 4. Distribución de P(falsa) por clase en cada corpus de origen
    fig, axes = plt.subplots(1, len(orden), figsize=(13, 3.5), sharey=False)
    bins = np.linspace(0, 1, 21)
    for ax, d in zip(axes, orden):
        g = df[df['dataset_origen'] == d]
        for cl, nom in [(0, 'Reales'), (1, 'Falsas')]:
            v = g.loc[g['clase_num'] == cl, 'prob_fake']
            ax.hist(v, bins=bins, color=gc.CLASE[cl], alpha=0.55, label=nom, edgecolor='white', linewidth=0.5)
        ax.axvline(0.5, color=gc.TINTA, ls=':', lw=1)
        nr, nf = int((g['clase_num'] == 0).sum()), int((g['clase_num'] == 1).sum())
        ax.set_title(f"{CORTO.get(d, d)}\n{mil(nr)} reales · {mil(nf)} falsas", fontsize=9.5)
        ax.set_xlabel('P(falsa) según SaBERT', fontsize=8.5)
        gc.coma(ax, 'x', 1)
    axes[0].set_ylabel('Noticias')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fig_out['dist'] = gc.guardar(fig, IMG / 'distribucion_probabilidad_por_corpus.png')

    # 5. Tasas de falsos negativos y falsos positivos por corpus
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(orden))
    fn = [M['corpus_origen'][d]['tasa_falsos_negativos'] for d in orden]
    fp = [M['corpus_origen'][d]['tasa_falsos_positivos'] for d in orden]
    b1 = ax.bar(x - 0.2, fn, 0.38, color=gc.CLASE[1], label='Falsos negativos (falsas que pasan como reales)')
    b2 = ax.bar(x + 0.2, fp, 0.38, color=gc.CLASE[0], label='Falsos positivos (reales marcadas como falsas)')
    for b, v in list(zip(b1, fn)) + list(zip(b2, fp)):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, pct(v, 0), ha='center', fontsize=8, color=gc.TINTA_2)
    ax.set_xticks(x, [CORTO.get(d, d) for d in orden], fontsize=8.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Tasa')
    gc.coma(ax, 'y', 1)
    ax.legend(fontsize=8.5, loc='upper left')
    ax.set_title('Errores de SaBERT por corpus de origen (umbral 0,5)')
    fig_out['err'] = gc.guardar(fig, IMG / 'falsos_negativos_positivos_corpus.png')

    # 6. Máscaras de anonimización por corpus
    mc = A['mascaras_por_corpus']
    om = sorted(mc, key=lambda d: -mc[d]['pct_con_mascara'])
    fig, ax = plt.subplots(figsize=(8, 3.0))
    yv = np.arange(len(om))[::-1]
    ax.barh(yv, [mc[d]['pct_con_mascara'] for d in om], color=[col_reg[mc[d]['region']] for d in om], height=0.55)
    for yy, d in zip(yv, om):
        ax.text(mc[d]['pct_con_mascara'] + 0.01, yy, f"{pct(mc[d]['pct_con_mascara'], 0)}  ({mil(mc[d]['ocurrencias_number'])} *NUMBER*)",
                va='center', fontsize=8.5, color=gc.TINTA_2)
    ax.set_yticks(yv, [CORTO.get(d, d) for d in om])
    ax.set_xlim(0, 1.15)
    ax.set_xlabel('Proporción de noticias con al menos una máscara (*NUMBER*, *URL*, *PHONE*)')
    gc.coma(ax, 'x', 1)
    ax.grid(axis='y', visible=False)
    fig_out['masc'] = gc.guardar(fig, IMG / 'mascaras_por_corpus.png')

    # 7. Contrafácticos: original frente a perturbado
    S = A['metricas_estratificadas']
    esc = [('España\noriginal', S['espana'], gc.SERIE[0]), ('España\ncon máscaras', A['contrafactico_espana_enmascarada'], gc.SERIE[0]),
           ('MEX-A3T + FakeDeS\noriginal', S['mex_fakedes'], gc.SERIE[1]), ('MEX-A3T + FakeDeS\nsin máscaras', A['contrafactico_mex_fakedes_restaurado'], gc.SERIE[1]),
           ('Omdena\n(nunca tuvo máscaras)', S['omdena'], gc.SERIE[2])]
    mets = [('exactitud', 'Exactitud'), ('f1_macro', 'F1 macro'), ('recall_falsa', 'Recall de falsas'), ('roc_auc', 'ROC-AUC')]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), sharey=True)
    for ax, (k, nom) in zip(axes, mets):
        v = [e[1][k] for e in esc]
        bars = ax.barh(np.arange(len(esc))[::-1], v, color=[e[2] for e in esc], height=0.6)
        for i, b in enumerate(bars):
            if i in (1, 3):
                b.set_hatch('///'); b.set_edgecolor('white')
            ax.text(v[i] + 0.01, b.get_y() + b.get_height() / 2, f(v[i], 2), va='center', fontsize=8, color=gc.TINTA_2)
        ax.set_xlim(0, 1.12)
        ax.set_title(nom)
        gc.coma(ax, 'x', 1)
        ax.grid(axis='y', visible=False)
    axes[0].set_yticks(np.arange(len(esc))[::-1], [e[0] for e in esc], fontsize=8.5)
    fig.suptitle('Contrafácticos: introducir o quitar máscaras sin cambiar el dominio (barras rayadas = texto modificado)', x=0.01, ha='left', fontsize=10.5)
    fig.tight_layout()
    fig_out['cf'] = gc.guardar(fig, IMG / 'contrafacticos_mascaras_vs_dominio.png')
    return {k: 'Imagenes/' + Path(v).name for k, v in fig_out.items()}


def fila_metricas(nombre, r):
    return (f"| {nombre} | {mil(r['n'])} | {pct(r['n_falsas'] / r['n'])} | {pct(r['exactitud'])} | {f(r['f1_macro'])} "
            f"[{f(r['f1_macro_ic95'][0])}; {f(r['f1_macro_ic95'][1])}] | {f(r['precision_falsa'])} | {pct(r['recall_falsa'])} | "
            f"{f(r['roc_auc'])} | {f(r['ap'])} | {pct(r['tasa_falsos_negativos'])} | {pct(r['tasa_falsos_positivos'])} |")


ENC = ("| Grupo | n | % falsas | Exactitud | F1 macro [IC 95 %] | Precisión (falsa) | Recall (falsa) | ROC-AUC | AP | Tasa FN | Tasa FP |\n"
       "|:---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|")


def main():
    df = pd.read_csv(cfg.PREDICCIONES_SABERT_4418_CSV)
    M = calcular_metricas(df)
    A = json.loads(AUDITORIA_JSON.read_text(encoding='utf-8'))
    figs = figuras(df, M, A)
    S, D, V = A['metricas_estratificadas'], A['deltas'], A['verificacion_procedencia']
    es, la = M['regiones']['España'], M['regiones']['América Latina']
    co = M['corpus_origen']
    orden = sorted(co, key=lambda d: (co[d]['region'] != 'España', -co[d]['f1_macro']))
    peor = min(co, key=lambda d: co[d]['f1_macro'])
    mx = A['mascaras_por_corpus']
    dA, dB, dO = D['enmascarar_espana'], D['restaurar_mex_fakedes'], D['omdena_vs_espana']
    con_masc = [d for d in mx if mx[d]['pct_con_mascara'] >= 0.05]
    pocas = [d for d in mx if 0 < mx[d]['pct_con_mascara'] < 0.05]
    txt_con = ', '.join(f"{CORTO.get(d, d)} ({mil(mx[d]['n_con_mascara'])} de {mil(mx[d]['n'])} noticias)" for d in con_masc)
    txt_pocas = ', '.join(f"{CORTO.get(d, d)} tiene {mil(mx[d]['n_con_mascara'])} de {mil(mx[d]['n'])}" for d in pocas) or 'ningún otro corpus tiene máscaras'
    mascaras_irrelevantes =max(abs(dA['delta_roc_auc']), abs(dB['delta_roc_auc'])) < 0.02 and abs(dO['delta_roc_auc']) > 0.1

    L = ["# SaBERT en España y América Latina: región, corpus de origen y máscaras de anonimización", "",
         "**Generado por:** `Codigos/generar_reporte_sabert.py` a partir de `Dataset/predicciones_sabert_ampliado_4418.csv` y "
         "`Reportes/auditoria_censura_vs_domain_shift.json` (`Codigos/auditoria_censura_vs_domain_shift.py`)  ",
         f"**Modelo:** `{M['metadata']['modelo']}` (usado tal como se publicó, sin reentrenar; P(falsa) con umbral 0,5)  ",
         f"**Corpus:** corpus ampliado de {mil(M['global']['n'])} noticias ({mil(es['n'])} de España y {mil(la['n'])} de América Latina), "
         "con las etiquetas vigentes (Omdena con `Corrected_label`). Métricas en `Reportes/metricas_sabert_dataset_ampliado_4418.json`.", "",
         "## 1. Resumen", "",
         f"- En **España** SaBERT obtiene F1 macro {f(es['f1_macro'])} y ROC-AUC {f(es['roc_auc'])}; en **América Latina**, F1 macro {f(la['f1_macro'])} "
         f"y ROC-AUC {f(la['roc_auc'])}. La exactitud cae {pp(la['exactitud'] - es['exactitud']).lstrip('+')} y el recall de noticias falsas pasa de "
         f"{pct(es['recall_falsa'])} a {pct(la['recall_falsa'])}: en América Latina deja pasar como reales el {pct(la['tasa_falsos_negativos'])} de las falsas.",
         f"- La caída no es uniforme: el peor corpus es **{CORTO.get(peor, peor)}** (F1 macro {f(co[peor]['f1_macro'])}, ROC-AUC {f(co[peor]['roc_auc'])}, "
         f"recall de falsas {pct(co[peor]['recall_falsa'])}).",
         ("- **Las máscaras de anonimización no explican la caída.** " if mascaras_irrelevantes else "- **Efecto de las máscaras:** ") +
         f"Introducir máscaras en España cambia el ROC-AUC en {f(dA['delta_roc_auc'])} y quitarlas en MEX-A3T + FakeDeS lo cambia en {f(dB['delta_roc_auc'])}, "
         f"mientras que Omdena, que nunca tuvo máscaras, queda {f(abs(dO['delta_roc_auc']))} por debajo de España en ROC-AUC.",
         f"- Las probabilidades guardadas se verificaron recalculando SaBERT (revisión fija) en {V['n_muestra']} noticias: diferencia máxima "
         f"{V['dif_abs_max']:.1e}".replace('.', ',') + f"; {V['conclusion']}.", "",
         "## 2. Resultados por región", "", ENC,
         fila_metricas('**Global**', M['global']), fila_metricas('España', es), fila_metricas('América Latina', la), "",
         f"![Matrices de confusión por región]({figs['cm']})", "",
         "Las filas suman 100 %: la fila «Falsa» muestra qué parte de las noticias falsas detecta SaBERT y qué parte deja pasar como reales.", "",
         "## 3. Resultados por corpus de origen", "",
         "La región mezcla corpus muy distintos (temas, época, forma de anotar). El desglose muestra dónde se concentra la caída.", "", ENC]
    for d in orden:
        L.append(fila_metricas(f"{CORTO.get(d, d)}", co[d]))
    L += ["", f"![F1 macro por región y corpus]({figs['f1']})", "",
          f"![Curvas ROC y PR por corpus]({figs['roc']})", "",
          f"![Distribución de P(falsa) por corpus]({figs['dist']})", "",
          "En los corpus donde SaBERT funciona, las noticias falsas se concentran a la derecha del umbral y las reales a la izquierda. "
          "Donde falla, las falsas se quedan a la izquierda junto con las reales: el modelo no las reconoce como falsas, no es solo un problema de umbral "
          "(si lo fuera, el ROC-AUC seguiría alto).", "",
          f"![Falsos negativos y positivos por corpus]({figs['err']})", "",
          "## 4. ¿Máscaras de anonimización o cambio de dominio?", "",
          f"Algunos corpus reemplazan cifras, enlaces y teléfonos por `*NUMBER*`, `*URL*` y `*PHONE*`. Las máscaras están casi solo en "
          f"**{txt_con}**; {txt_pocas}; el resto no tiene ninguna. Como ese corpus está en América Latina, la caída regional podría deberse a las "
          "máscaras y no a la región. Se separan ambos efectos con tres pruebas.", "",
          "| Corpus | Región | n | Noticias con máscara | *NUMBER* | *URL* | *PHONE* |", "|:---|:---|---:|---:|---:|---:|---:|"]
    for d in sorted(mx, key=lambda d: -mx[d]['pct_con_mascara']):
        x = mx[d]
        L.append(f"| {CORTO.get(d, d)} | {x['region']} | {mil(x['n'])} | {mil(x['n_con_mascara'])} ({pct(x['pct_con_mascara'])}) | "
                 f"{mil(x['ocurrencias_number'])} | {mil(x['ocurrencias_url'])} | {mil(x['ocurrencias_phone'])} |")
    cA, cB = A['contrafactico_espana_enmascarada'], A['contrafactico_mex_fakedes_restaurado']
    L += ["", f"![Máscaras por corpus]({figs['masc']})", "",
          "| Escenario | n | Exactitud | F1 macro | Recall (falsa) | ROC-AUC | AP |", "|:---|---:|---:|---:|---:|---:|---:|"]
    for nom, r in [('España original', S['espana']), ('España con máscaras sintéticas (contrafáctico A)', cA),
                   ('MEX-A3T + FakeDeS original', S['mex_fakedes']), ('MEX-A3T + FakeDeS sin máscaras (contrafáctico B)', cB),
                   ('MEX-A3T, solo textos con máscara', S['mex_a3t_con_mascara']),
                   ('MEX-A3T, solo textos sin máscara', S['mex_a3t_sin_mascara']),
                   ('Omdena (sin máscaras)', S['omdena'])]:
        L.append(f"| {nom} | {mil(r['n'])} | {pct(r['exactitud'])} | {f(r['f1_macro'])} | {pct(r['recall_falsa'])} | {f(r['roc_auc'])} | {f(r['ap'])} |")
    L += ["", f"![Contrafácticos]({figs['cf']})", "",
          f"1. **Contrafáctico A (introducir máscaras sin cambiar de dominio).** En las {mil(S['espana']['n'])} noticias de España se reemplazaron "
          f"{mil(cA['cifras_enmascaradas'])} cifras y los enlaces por `*NUMBER*` y `*URL*` ({pct(cA['pct_textos_modificados'])} de los textos cambió). "
          f"La exactitud cambia {pp(dA['delta_exactitud'])}, el F1 macro {f(dA['delta_f1_macro'])} y el ROC-AUC {f(dA['delta_roc_auc'])}.",
          f"2. **Contrafáctico B (quitar máscaras sin cambiar de dominio).** En MEX-A3T + FakeDeS las máscaras se reemplazaron por valores plausibles "
          f"({mil(cB['n_textos_modificados'])} textos modificados). La exactitud cambia {pp(dB['delta_exactitud'])}, el F1 macro {f(dB['delta_f1_macro'])} "
          f"y el ROC-AUC {f(dB['delta_roc_auc'])}.",
          f"3. **Comparación dentro de América Latina.** Omdena no tiene ninguna máscara y es donde SaBERT rinde peor (ROC-AUC {f(S['omdena']['roc_auc'])}); "
          f"MEX-A3T + FakeDeS, con máscaras, alcanza ROC-AUC {f(S['mex_fakedes']['roc_auc'])}. Dentro de MEX-A3T, los textos con máscara obtienen "
          f"ROC-AUC {f(S['mex_a3t_con_mascara']['roc_auc'])} y los textos sin máscara {f(S['mex_a3t_sin_mascara']['roc_auc'])}: "
          + ("tener máscaras no empeora el resultado. " if S['mex_a3t_con_mascara']['roc_auc'] >= S['mex_a3t_sin_mascara']['roc_auc']
             else "los textos con máscara rinden peor. ") + "Esta comparación es observacional (los textos con y sin máscara son distintos); la prueba causal son los contrafácticos 1 y 2.", "",
          ("**Conclusión:** las máscaras mueven las métricas muy poco en ambas direcciones, mientras que el corpus latinoamericano sin máscaras (Omdena) es el "
           "que más cae. La caída de SaBERT en América Latina se debe al cambio de dominio (países, entidades, temas y forma de redactar y de anotar de cada "
           "corpus), no a la anonimización."
           if mascaras_irrelevantes else
           "**Conclusión:** los contrafácticos muestran un efecto apreciable de las máscaras; la caída regional no puede atribuirse solo al cambio de dominio."),
          "", "## 5. Interpretación y límites", "",
          "- **Por qué cae en América Latina (hipótesis, no demostrada).** La ficha pública de SaBERT no declara con qué corpus se ajustó, de modo que no se puede "
          "afirmar que se entrenó con prensa o verificadores españoles. La explicación más compatible con los datos es un cambio de dominio: los corpus de España "
          "se parecen más a los datos con que se ajustó el modelo que los de América Latina.",
          "- **Posible solapamiento con el entrenamiento.** El 93 % del corpus Unified proviene de Blanco-Fernández et al. (2024), un corpus amplio de noticias en "
          "español publicado en abierto. Si SaBERT vio parte de esas noticias al ajustarse, su rendimiento en España estaría inflado. No se puede comprobar sin "
          "conocer sus datos de entrenamiento; por eso el resultado de España se toma como cota superior, no como desempeño esperado en noticias nuevas.",
          "- **El corpus no es solo la región.** Cada corpus tiene su propio periodo, temas y criterio de anotación; parte de las diferencias entre corpus "
          "puede deberse a la anotación y no al modelo. Omdena, en particular, reúne noticias políticas latinoamericanas etiquetadas por el propio proyecto "
          "(columna `Corrected_label`).",
          "- **Umbral fijo.** Todas las cifras usan el umbral 0,5. El ROC-AUC, que no depende del umbral, muestra que en los corpus donde SaBERT falla el problema "
          "no se resuelve moviendo el umbral.",
          "- **Implicación para la tesis.** Un clasificador de veracidad ajustado en un dominio no se transfiere de forma fiable a otro. Esto motiva un "
          "instrumento complementario basado en rasgos de estilo (Modelo sobre Variables), interpretable y evaluado sin fuga en ambas regiones, y explicaciones "
          "por oración (MexGen) que permitan revisar en qué se apoya cada decisión.", "",
          "## 6. Qué cambió respecto de la versión anterior de este reporte", "",
          "- Las cifras se recalcularon con las etiquetas vigentes del corpus (Omdena con `Corrected_label`); la versión anterior usaba la etiqueta `Prediction` "
          "de Omdena, que era la salida de un clasificador.",
          "- Se retiró la afirmación de que SaBERT se ajustó con notas de Newtral, Maldita y EFE Verifica, porque su ficha no lo declara.",
          f"- La versión anterior comparaba textos con y sin máscara en MEX-A3T + FakeDeS juntos; como FakeDeS casi no tiene máscaras y SaBERT rinde peor en "
          f"él, esa comparación favorecía a los textos con máscara. La comparación se hace ahora dentro de MEX-A3T (ROC-AUC {f(S['mex_a3t_con_mascara']['roc_auc'])} "
          f"con máscara frente a {f(S['mex_a3t_sin_mascara']['roc_auc'])} sin máscara).",
          f"- La versión anterior citaba en el texto un número de cifras y de textos modificados en el contrafáctico A que no coincidía con su propio JSON; "
          f"el valor correcto es {mil(cA['cifras_enmascaradas'])} cifras en {mil(cA['n_textos_modificados'])} textos.",
          "- Se retiró la propuesta de una arquitectura en cascada con latencias y umbrales que no se evaluaron.",
          "- Todas las cifras y figuras se generan desde los archivos de resultados; no quedan valores escritos a mano.", "",
          "## Archivos", "",
          "| Archivo | Contenido |", "|:---|:---|",
          "| `Codigos/auditoria_censura_vs_domain_shift.py` | Máscaras, métricas estratificadas, contrafácticos A y B, verificación de procedencia |",
          "| `Codigos/generar_reporte_sabert.py` | Métricas por región, corpus y fuente; figuras; este reporte |",
          "| `Reportes/auditoria_censura_vs_domain_shift.json` | Resultados de la auditoría |",
          "| `Reportes/metricas_sabert_dataset_ampliado_4418.json` | Métricas con IC 95 % bootstrap del F1 macro |",
          "| `Reportes/datos/` | P(falsa) de los textos contrafácticos y de la muestra de verificación (caché) |",
          "| `../Dataset/predicciones_sabert_ampliado_4418.csv` | P(falsa) de SaBERT para las 4.418 noticias |", ""]
    salida = REP / 'Reporte_Evaluacion_SaBERT_Domain_Shift.md'
    salida.write_text("\n".join(L) + "\n", encoding='utf-8')
    print(f"[OK] {salida}")


if __name__ == '__main__':
    main()
