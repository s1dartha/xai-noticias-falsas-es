"""
Selección del clasificador del Modelo sobre Variables entre tres familias respaldadas por la literatura:
regresión logística, bosque aleatorio y gradient boosting.

Protocolo (idéntico al de evaluar_modelo_sobre_variables_sin_fuga.py): textos únicos, StratifiedGroupKFold(5),
umbral anidado que maximiza el F1 macro, hiperparámetros fijos. Los tres modelos usan las 43 variables (S+R+E+O+L).

Regla de decisión, fijada antes de ver resultados: se elige el modelo de mayor F1 macro global en validación cruzada.
Si su diferencia con la regresión logística no es significativa (el IC 95 % del bootstrap pareado incluye el 0), se
elige la regresión logística por parsimonia e interpretabilidad (Rudin, 2019).

Salidas: Reportes/metricas_seleccion_modelo_sobre_variables.json y Reportes/Imagenes/seleccion_modelo/*.png
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config_proyecto.py").exists())))
import config_proyecto as cfg
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluar_modelo_sobre_variables_sin_fuga as ev

TIPOS = ['lr', 'rf', 'gbdt']
# Paleta categórica de referencia (slots 1-3, validados para todos los pares) y gris neutro para SaBERT
COLOR = {'lr': '#2a78d6', 'rf': '#eb6834', 'gbdt': '#1baf7a', 'sabert': '#898781'}
TINTA, TINTA_2, REJILLA = '#0b0b0b', '#52514e', '#e1e0d9'
IMG = cfg.MODELO_VARIABLES_IMG_DIR / 'seleccion_modelo'


def estilo(ax):
    ax.grid(axis='y', color=REJILLA, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color('#c3c2b7')
    ax.tick_params(colors=TINTA_2, labelsize=9)


def main():
    t0 = time.time()
    df, n_noticias, meta_tabla = ev.cargar_datos()
    G = meta_tabla['grupos']
    variables = G['S'] + G['R'] + G['E'] + G['O'] + G['L']
    y, g = df['label_num'].values, df['id_texto'].values
    X = df[variables].values.astype(float)
    externo = ev.pliegues_externos(df)
    region = {'España': (df['region'] == 'España').values, 'América Latina': (df['region'] == 'América Latina').values}
    res = {'metadata': {
        'n_textos_unicos': int(len(df)), 'n_variables': len(variables), 'variables': variables,
        'validacion': f'StratifiedGroupKFold({ev.N_PLIEGUES}), umbral anidado (F1 macro)',
        'modelos': {t: {'nombre': ev.NOMBRES_MODELO[t], 'configuracion': ev.DESCRIPCION_MODELO[t]} for t in TIPOS},
        'regla': 'mayor F1 macro global en validación cruzada; si la diferencia con la regresión logística no es significativa '
                 '(IC 95 % pareado incluye 0), se elige la regresión logística por parsimonia',
    }, 'modelos': {}}
    oof, preds = {}, {}

    print("[1/3] Validación cruzada", flush=True)
    for t in TIPOS:
        t1 = time.time()
        p, pred, th = ev.validacion_cruzada(df, X, t, externo)
        oof[t], preds[t] = p, pred
        res['modelos'][t] = {'validacion_cruzada': {'umbrales_por_pliegue': th, **ev.desglose(df, y, p, pred)},
                             'segundos_validacion_cruzada': time.time() - t1}
        m = res['modelos'][t]['validacion_cruzada']['Global']
        print(f"  {ev.NOMBRES_MODELO[t]:<20} F1m={m['f1_macro']:.3f} FP={m['FP']} FN={m['FN']} AUC={m['roc_auc']:.3f} ({time.time() - t1:.0f} s)", flush=True)
    p_sab = df['sabert_p_falsa'].values
    p_s, pred_s, th_s = ev.validacion_cruzada(df, p_sab[:, None], 'sabert', externo)
    res['sabert_referencia'] = {'validacion_cruzada': {'umbrales_por_pliegue': th_s, **ev.desglose(df, y, p_s, pred_s)}}

    res['comparaciones_pareadas'] = {}
    for t in ['rf', 'gbdt']:
        for r, m in [('Global', np.ones(len(y), bool)), *region.items()]:
            res['comparaciones_pareadas'][f'{ev.NOMBRES_MODELO[t]} − Regresión logística · {r}'] = ev.diferencia_pareada(y[m], preds[t][m], preds['lr'][m])

    print("[2/3] Transferencia entre regiones y exclusión de una fuente", flush=True)
    escenarios = {f'{a} → {b}': (region[a], region[b]) for a, b in [('España', 'América Latina'), ('América Latina', 'España')]}
    for f in sorted(df['Fuente'].unique()):
        m = (df['Fuente'] == f).values
        escenarios[f'Sin {ev.nombre_fuente(f)}'] = (~m, m)
    res['escenarios_externos'] = {}
    for esc, (m_tr, m_te) in escenarios.items():
        r = {}
        for t in TIPOS + ['sabert']:
            Xe = p_sab[:, None] if t == 'sabert' else X
            p, th = ev.ajustar_y_predecir(Xe[m_tr], y[m_tr], g[m_tr], Xe[m_te], t)
            r[t] = {'umbral': th, **ev.metricas(y[m_te], p, (p >= th).astype(int))}
        res['escenarios_externos'][esc] = r
        print(f"  {esc:<55} " + " | ".join(f"{t}: {v['f1_macro']:.3f}" for t, v in r.items()), flush=True)
    loso = [e for e in escenarios if e.startswith('Sin ')]
    res['resumen_exclusion_fuente'] = {t: float(np.mean([res['escenarios_externos'][e][t]['f1_macro'] for e in loso])) for t in TIPOS + ['sabert']}

    # Decisión
    f1 = {t: res['modelos'][t]['validacion_cruzada']['Global']['f1_macro'] for t in TIPOS}
    mejor = max(f1, key=f1.get)
    elegido, motivo = mejor, f'{ev.NOMBRES_MODELO[mejor]} tiene el mayor F1 macro global'
    if mejor != 'lr':
        d = res['comparaciones_pareadas'][f'{ev.NOMBRES_MODELO[mejor]} − Regresión logística · Global']
        if d['ic95'][0] <= 0:
            elegido = 'lr'
            motivo = (f'{ev.NOMBRES_MODELO[mejor]} tiene el mayor F1 macro, pero su ventaja sobre la regresión logística no es significativa '
                      f'(IC 95 % [{d["ic95"][0]:.3f}; {d["ic95"][1]:.3f}]); se elige la regresión logística por parsimonia')
    else:
        motivo = 'La regresión logística tiene el mayor F1 macro global'
    res['modelo_elegido'], res['motivo'] = elegido, motivo
    print(f"\nElegido: {ev.NOMBRES_MODELO[elegido]} — {motivo}", flush=True)

    # Coeficientes de la regresión logística (descriptivos: ajuste con todos los textos únicos)
    lr = ev.modelo('lr').fit(X, y)
    coef = lr[-1].coef_[0]
    grupo_de = {v: k for k, vs in G.items() for v in vs}
    res['coeficientes_regresion_logistica'] = sorted(
        [{'variable': v, 'grupo': grupo_de[v], 'coeficiente_estandarizado': float(c), 'odds_ratio_por_1_DE': float(np.exp(c))}
         for v, c in zip(variables, coef)], key=lambda r: -abs(r['coeficiente_estandarizado']))

    # ---------------- Figuras ----------------
    print("[3/3] Figuras", flush=True)
    IMG.mkdir(parents=True, exist_ok=True)
    figuras = {}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.dpi': 150, 'savefig.dpi': 200})

    # 1. F1 macro por escenario
    esc_cv = [('Validación cruzada\nGlobal', lambda t: res['modelos'][t]['validacion_cruzada']['Global'] if t != 'sabert' else res['sabert_referencia']['validacion_cruzada']['Global']),
              ('Validación cruzada\nEspaña', lambda t: res['modelos'][t]['validacion_cruzada']['España'] if t != 'sabert' else res['sabert_referencia']['validacion_cruzada']['España']),
              ('Validación cruzada\nAmérica Latina', lambda t: res['modelos'][t]['validacion_cruzada']['América Latina'] if t != 'sabert' else res['sabert_referencia']['validacion_cruzada']['América Latina']),
              ('España →\nAmérica Latina', lambda t: res['escenarios_externos']['España → América Latina'][t]),
              ('América Latina →\nEspaña', lambda t: res['escenarios_externos']['América Latina → España'][t])]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    estilo(ax)
    ancho = 0.19
    serie = TIPOS + ['sabert']
    for i, t in enumerate(serie):
        vals = [fn(t) for _, fn in esc_cv]
        x = np.arange(len(esc_cv)) + (i - 1.5) * (ancho + 0.01)
        err = np.array([[v['f1_macro'] - v['f1_macro_ic95'][0], v['f1_macro_ic95'][1] - v['f1_macro']] for v in vals]).T
        ax.bar(x, [v['f1_macro'] for v in vals], ancho, color=COLOR[t], edgecolor='white', linewidth=1,
               label=ev.NOMBRES_MODELO.get(t, 'SaBERT (referencia)'), hatch='//' if t == 'sabert' else None)
        ax.errorbar(x, [v['f1_macro'] for v in vals], yerr=err, fmt='none', ecolor=TINTA_2, elinewidth=0.8, capsize=2)
    ax.axhline(0.5, color='#c3c2b7', linewidth=0.8, linestyle=':')
    ax.text(len(esc_cv) - 0.45, 0.505, 'azar', color=TINTA_2, fontsize=8, ha='right', va='bottom')
    ax.set_xticks(np.arange(len(esc_cv)))
    ax.set_xticklabels([e for e, _ in esc_cv], fontsize=9, color=TINTA)
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel('F1 macro (IC 95 %)', color=TINTA, fontsize=10)
    ax.set_title('F1 macro de los tres clasificadores y de SaBERT por escenario', color=TINTA, fontsize=11, loc='left')
    ax.legend(frameon=False, fontsize=9, ncol=4, loc='upper left')
    fig.tight_layout()
    figuras['f1_por_escenario'] = IMG / 'f1_macro_por_escenario.png'
    fig.savefig(figuras['f1_por_escenario'])
    plt.close(fig)

    # 2. Exclusión de una fuente
    fig, ax = plt.subplots(figsize=(10, 4.2))
    estilo(ax)
    for i, t in enumerate(serie):
        vals = [res['escenarios_externos'][e][t]['f1_macro'] for e in loso]
        x = np.arange(len(loso)) + (i - 1.5) * (ancho + 0.01)
        ax.bar(x, vals, ancho, color=COLOR[t], edgecolor='white', linewidth=1, label=ev.NOMBRES_MODELO.get(t, 'SaBERT (referencia)'),
               hatch='//' if t == 'sabert' else None)
    ax.axhline(0.5, color='#c3c2b7', linewidth=0.8, linestyle=':')
    ax.set_xticks(np.arange(len(loso)))
    ax.set_xticklabels([e.replace('Sin ', 'Prueba en\n') for e in loso], fontsize=8.5, color=TINTA)
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel('F1 macro', color=TINTA, fontsize=10)
    ax.set_title('Fuente no vista en el entrenamiento: F1 macro en la fuente excluida', color=TINTA, fontsize=11, loc='left')
    ax.legend(frameon=False, fontsize=9, ncol=4, loc='upper left')
    fig.tight_layout()
    figuras['exclusion_fuente'] = IMG / 'f1_macro_exclusion_fuente.png'
    fig.savefig(figuras['exclusion_fuente'])
    plt.close(fig)

    # 3. Falsos positivos y falsos negativos (validación cruzada global)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=False)
    for ax, clave, titulo in [(axes[0], 'FP', 'Falsos positivos (reales marcadas como falsas)'),
                              (axes[1], 'FN', 'Falsos negativos (falsas no detectadas)')]:
        estilo(ax)
        vals = [res['modelos'][t]['validacion_cruzada']['Global'][clave] for t in TIPOS] + [res['sabert_referencia']['validacion_cruzada']['Global'][clave]]
        b = ax.bar(range(4), vals, 0.6, color=[COLOR[t] for t in serie], edgecolor='white', linewidth=1)
        b[3].set_hatch('//')
        for xi, v in enumerate(vals):
            ax.text(xi, v + max(vals) * 0.015, f'{v}', ha='center', va='bottom', fontsize=9, color=TINTA)
        ax.set_xticks(range(4))
        ax.set_xticklabels(['Regresión\nlogística', 'Bosque\naleatorio', 'Gradient\nboosting', 'SaBERT\n(referencia)'], fontsize=8.5, color=TINTA)
        ax.set_title(titulo, fontsize=10, color=TINTA, loc='left')
    fig.suptitle('Errores en validación cruzada (4.100 textos únicos, umbral anidado)', fontsize=11, color=TINTA, x=0.01, ha='left')
    fig.tight_layout()
    figuras['fp_fn'] = IMG / 'falsos_positivos_negativos.png'
    fig.savefig(figuras['fp_fn'])
    plt.close(fig)

    # 4. Curvas ROC y PR (predicciones fuera de pliegue)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for t in serie:
        p = p_s if t == 'sabert' else oof[t]
        auc = (res['sabert_referencia'] if t == 'sabert' else res['modelos'][t])['validacion_cruzada']['Global']
        fpr, tpr, _ = roc_curve(y, p)
        pr, rc, _ = precision_recall_curve(y, p)
        nombre = ev.NOMBRES_MODELO.get(t, 'SaBERT (referencia)')
        estilo_l = '--' if t == 'sabert' else '-'
        axes[0].plot(fpr, tpr, color=COLOR[t], linewidth=2, linestyle=estilo_l, label=f"{nombre} (AUC {auc['roc_auc']:.3f})".replace('.', ','))
        axes[1].plot(rc, pr, color=COLOR[t], linewidth=2, linestyle=estilo_l, label=f"{nombre} (AP {auc['pr_auc']:.3f})".replace('.', ','))
    axes[0].plot([0, 1], [0, 1], color='#c3c2b7', linewidth=0.8, linestyle=':')
    axes[1].axhline(y.mean(), color='#c3c2b7', linewidth=0.8, linestyle=':')
    for ax, (xl, yl, ti) in zip(axes, [('Tasa de falsos positivos', 'Recall (falsas)', 'Curva ROC'), ('Recall (falsas)', 'Precisión (falsas)', 'Curva precisión-recall')]):
        estilo(ax)
        ax.grid(axis='x', color=REJILLA, linewidth=0.6)
        ax.set_xlabel(xl, fontsize=9.5, color=TINTA)
        ax.set_ylabel(yl, fontsize=9.5, color=TINTA)
        ax.set_title(ti, fontsize=10.5, color=TINTA, loc='left')
        ax.legend(frameon=False, fontsize=8.5, loc='lower right' if ti == 'Curva ROC' else 'upper right')
    fig.tight_layout()
    figuras['roc_pr'] = IMG / 'curvas_roc_pr.png'
    fig.savefig(figuras['roc_pr'])
    plt.close(fig)

    # 5. Coeficientes de la regresión logística (20 de mayor magnitud)
    top = res['coeficientes_regresion_logistica'][:20][::-1]
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.grid(axis='x', color=REJILLA, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    vals = [r['coeficiente_estandarizado'] for r in top]
    ax.barh(range(len(top)), vals, 0.65, color=['#e34948' if v > 0 else '#2a78d6' for v in vals], edgecolor='white', linewidth=1)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"{r['variable']} ({r['grupo']})" for r in top], fontsize=8.5, color=TINTA)
    ax.axvline(0, color='#c3c2b7', linewidth=0.8)
    ax.set_xlabel('Coeficiente estandarizado (log-odds por 1 desviación estándar)', fontsize=9.5, color=TINTA)
    ax.set_title('Regresión logística: 20 variables de mayor peso\nrojo = empuja hacia «falsa»; azul = empuja hacia «real»', fontsize=10.5, color=TINTA, loc='left')
    ax.tick_params(colors=TINTA_2, labelsize=8.5)
    fig.tight_layout()
    figuras['coeficientes_lr'] = IMG / 'coeficientes_regresion_logistica.png'
    fig.savefig(figuras['coeficientes_lr'])
    plt.close(fig)

    res['figuras'] = {k: str(v.relative_to(cfg.MODELO_VARIABLES_REPORTES_DIR)).replace('\\', '/') for k, v in figuras.items()}
    res['metadata']['duracion_s'] = time.time() - t0
    with open(cfg.SELECCION_MODELO_JSON, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False, default=float)
    print(f"[OK] {cfg.SELECCION_MODELO_JSON} ({res['metadata']['duracion_s'] / 60:.1f} min)")


if __name__ == '__main__':
    main()
