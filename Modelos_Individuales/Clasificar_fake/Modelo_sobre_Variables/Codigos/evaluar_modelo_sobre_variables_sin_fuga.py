"""
Fase 4b: evaluación del modelo sobre variables sin fuga de información.

Diseño
------
- Unidad de evaluación: textos únicos. Las 4.418 noticias se deduplican por texto normalizado (se conserva la primera
  aparición); así ningún texto aparece a la vez en entrenamiento y prueba ni se cuenta dos veces.
- Validación: StratifiedGroupKFold de 5 pliegues (grupo = id del texto).
- Todo lo que depende de los datos se aprende dentro del pliegue de entrenamiento: el modelo, el tratamiento de los NaN
  y el umbral de decisión. No hay cortes calculados sobre el corpus completo; la única imputación previa es la de
  red_consecutivas_media y red_consecutivas_min en las noticias de una sola oración, heredada de la extracción
  (ver construir_tabla_variables.py y el análisis de sensibilidad de la tesis, Tabla 4.15).
- Los textos casi idénticos (TF-IDF >= 0,90) no se agrupan aquí; su efecto se mide en
  Analisis_complementarios/sensibilidad_clasificador.py.
- Umbral (único punto de operación): validación anidada. Dentro del entrenamiento de cada pliegue se obtienen
  predicciones fuera de pliegue con otra StratifiedGroupKFold de 5 pliegues y se elige el θ que maximiza el F1 macro;
  ese θ se aplica al pliegue de prueba. SaBERT recibe exactamente el mismo procedimiento (θ elegido en el
  entrenamiento) y también se informa con su umbral nativo de 0,5.
- Hiperparámetros fijos, decididos antes de ver los resultados (no se ajustan).

Experimentos
------------
  1. Validación cruzada en el corpus completo, con ablación por grupos de variables (S, R, E, O, L).
  2. Transferencia entre regiones: entrenar en España y evaluar en América Latina, y al revés.
  3. Exclusión de una fuente por vez: entrenar con cuatro fuentes y evaluar en la quinta.
Métricas: F1 macro (principal) con IC 95 %, exactitud, precisión, recall, especificidad, FP, FN, tasas de FP y FN,
ROC-AUC y PR-AUC; desglose por región, por fuente y Omdena frente al resto dentro de América Latina; diferencias
pareadas por bootstrap frente a SaBERT.
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config_proyecto.py").exists())))
import config_proyecto as cfg

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

N_PLIEGUES = 5
N_BOOT = 1000
REJILLA_UMBRAL = np.round(np.arange(0.05, 0.951, 0.01), 2)
OMDENA = 'Fake-news-latam-omdena'


NOMBRES_MODELO = {'lr': 'Regresión logística', 'rf': 'Bosque aleatorio', 'gbdt': 'Gradient boosting'}
DESCRIPCION_MODELO = {
    'lr': 'imputación por mediana + estandarización (ajustadas en el pliegue) + LogisticRegression(C=1, L2)',
    'rf': 'imputación por mediana (ajustada en el pliegue) + RandomForestClassifier(500 árboles, min_samples_leaf=5, max_features="sqrt")',
    'gbdt': 'HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0)',
}


def modelo(tipo):
    if tipo == 'gbdt':
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                              l2_regularization=1.0, random_state=cfg.SEED)
    if tipo == 'rf':
        return make_pipeline(SimpleImputer(strategy='median'),
                             RandomForestClassifier(n_estimators=500, min_samples_leaf=5, max_features='sqrt', n_jobs=-1, random_state=cfg.SEED))
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))


# ---------------------------------------------------------------------------
# MÉTRICAS
# ---------------------------------------------------------------------------
def _f1_macro(y, pred):
    tp = np.sum((pred == 1) & (y == 1)); tn = np.sum((pred == 0) & (y == 0))
    fp = np.sum((pred == 1) & (y == 0)); fn = np.sum((pred == 0) & (y == 1))
    f1_pos = 2 * tp / max(2 * tp + fp + fn, 1)
    f1_neg = 2 * tn / max(2 * tn + fn + fp, 1)
    return (f1_pos + f1_neg) / 2


def umbral_f1_macro(y, p):
    return float(REJILLA_UMBRAL[int(np.argmax([_f1_macro(y, (p >= t).astype(int)) for t in REJILLA_UMBRAL]))])


def metricas(y, p, pred):
    y, p, pred = np.asarray(y), np.asarray(p), np.asarray(pred)
    tp = int(np.sum((pred == 1) & (y == 1))); tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0))); fn = int(np.sum((pred == 0) & (y == 1)))
    rng = np.random.default_rng(cfg.SEED)
    boots = [_f1_macro(y[i], pred[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(N_BOOT))]
    dos_clases = len(np.unique(y)) == 2
    return {
        'n': int(len(y)), 'falsas': int(y.sum()), 'reales': int((y == 0).sum()),
        'f1_macro': float(_f1_macro(y, pred)), 'f1_macro_ic95': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        'exactitud': (tp + tn) / len(y),
        'precision_falsa': tp / max(tp + fp, 1), 'recall_falsa': tp / max(tp + fn, 1),
        'especificidad': tn / max(tn + fp, 1), 'f1_falsa': 2 * tp / max(2 * tp + fp + fn, 1),
        'VP': tp, 'VN': tn, 'FP': fp, 'FN': fn,
        'tasa_FP': fp / max(fp + tn, 1), 'tasa_FN': fn / max(fn + tp, 1),
        'roc_auc': float(roc_auc_score(y, p)) if dos_clases else None,
        'pr_auc': float(average_precision_score(y, p)) if dos_clases else None,
    }


def diferencia_pareada(y, pred_a, pred_b):
    """Bootstrap pareado de F1 macro (A − B), FP (A − B) y FN (A − B)."""
    rng = np.random.default_rng(cfg.SEED)
    d = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        d.append(_f1_macro(y[i], pred_a[i]) - _f1_macro(y[i], pred_b[i]))
    fp = lambda pr: int(np.sum((pr == 1) & (y == 0)))
    fn = lambda pr: int(np.sum((pr == 0) & (y == 1)))
    return {'dif_f1_macro': float(_f1_macro(y, pred_a) - _f1_macro(y, pred_b)),
            'ic95': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            'p_bootstrap_dif_menor_o_igual_0': float(np.mean(np.array(d) <= 0)),
            'dif_FP': fp(pred_a) - fp(pred_b), 'dif_FN': fn(pred_a) - fn(pred_b)}


# ---------------------------------------------------------------------------
# ENTRENAMIENTO CON UMBRAL ANIDADO
# ---------------------------------------------------------------------------
def ajustar_y_predecir(X_tr, y_tr, g_tr, X_te, tipo):
    """Devuelve (P(falsa) en prueba, θ elegido solo con el entrenamiento)."""
    if tipo == 'sabert':
        return X_te[:, 0], umbral_f1_macro(y_tr, X_tr[:, 0])
    oof = np.zeros(len(y_tr))
    interno = StratifiedGroupKFold(n_splits=N_PLIEGUES, shuffle=True, random_state=cfg.SEED + 1)
    for a, b in interno.split(X_tr, y_tr, g_tr):
        oof[b] = modelo(tipo).fit(X_tr[a], y_tr[a]).predict_proba(X_tr[b])[:, 1]
    theta = umbral_f1_macro(y_tr, oof)
    return modelo(tipo).fit(X_tr, y_tr).predict_proba(X_te)[:, 1], theta


def desglose(df, y, p, pred):
    out = {'Global': metricas(y, p, pred)}
    for r in ['España', 'América Latina']:
        m = (df['region'] == r).values
        out[r] = metricas(y[m], p[m], pred[m])
    lat = (df['region'] == 'América Latina').values
    om = df['Fuente'].str.contains(OMDENA).values
    out['América Latina: Omdena'] = metricas(y[lat & om], p[lat & om], pred[lat & om])
    out['América Latina: sin Omdena'] = metricas(y[lat & ~om], p[lat & ~om], pred[lat & ~om])
    out['por_fuente'] = {nombre_fuente(f): metricas(y[(df['Fuente'] == f).values], p[(df['Fuente'] == f).values], pred[(df['Fuente'] == f).values])
                         for f in sorted(df['Fuente'].unique())}
    return out


def nombre_fuente(url):
    return url.rstrip('/').split('/')[-1].split(' (')[0] if '/' in url else url


def cargar_datos():
    """Tabla de variables deduplicada por texto, con la probabilidad de SaBERT alineada."""
    meta_tabla = json.loads(cfg.TABLA_VARIABLES_META_JSON.read_text(encoding='utf-8'))
    tabla = pd.read_csv(cfg.TABLA_VARIABLES_CSV)
    sab = pd.read_csv(cfg.PREDICCIONES_SABERT_4418_CSV)
    assert (sab['Text'].astype(str).values == tabla['Text'].astype(str).values).all(), "SaBERT desalineado con la tabla"
    assert (sab['clase_num'].map(cfg.etiqueta_binaria).values == tabla['label_num'].values).all()
    tabla['sabert_p_falsa'] = sab['prob_fake'].values
    df = tabla.drop_duplicates('id_texto', keep='first').reset_index(drop=True)
    return df, len(tabla), meta_tabla


def pliegues_externos(df):
    return list(StratifiedGroupKFold(n_splits=N_PLIEGUES, shuffle=True, random_state=cfg.SEED).split(df, df['label_num'].values, df['id_texto'].values))


def validacion_cruzada(df, X, tipo, externo):
    """Predicciones fuera de pliegue con umbral anidado: (P(falsa), predicción, umbrales por pliegue)."""
    y, g = df['label_num'].values, df['id_texto'].values
    p, pred, thetas = np.zeros(len(y)), np.zeros(len(y), int), []
    for tr, te in externo:
        p[te], th = ajustar_y_predecir(X[tr], y[tr], g[tr], X[te], tipo)
        pred[te] = (p[te] >= th).astype(int)
        thetas.append(th)
    return p, pred, thetas


def main():
    t0 = time.time()
    df, n_noticias, meta_tabla = cargar_datos()
    G = meta_tabla['grupos']
    y, g = df['label_num'].values, df['id_texto'].values
    print(f"Noticias: {n_noticias} | textos únicos evaluados: {len(df)} | falsas: {int(y.sum())}", flush=True)
    sel = json.loads(cfg.SELECCION_MODELO_JSON.read_text(encoding='utf-8'))
    M = sel['modelo_elegido']
    print(f"Modelo elegido en la selección: {NOMBRES_MODELO[M]}", flush=True)

    todas = G['S'] + G['R'] + G['E'] + G['O'] + G['L']
    CONFIGS = {
        'SaBERT': (['sabert_p_falsa'], 'sabert'),
        'S (sensacionalismo)': (G['S'], M),
        'R (redundancia)': (G['R'], M),
        'E (emociones)': (G['E'], M),
        'O (ortotipográficos)': (G['O'], M),
        'L (lingüísticos)': (G['L'], M),
        'S+R+E (tres pilares)': (G['S'] + G['R'] + G['E'], M),
        'S+R+E+O': (G['S'] + G['R'] + G['E'] + G['O'], M),
        'Completo (S+R+E+O+L)': (todas, M),
        'Completo + SaBERT': (todas + ['sabert_p_falsa'], M),
    }
    PRINCIPAL = 'Completo (S+R+E+O+L)'
    res = {'metadata': {
        'n_noticias': int(n_noticias), 'n_textos_unicos': int(len(df)), 'falsas': int(y.sum()), 'reales': int((y == 0).sum()),
        'validacion': f'StratifiedGroupKFold({N_PLIEGUES}), grupo = texto; umbral anidado con StratifiedGroupKFold({N_PLIEGUES}) interna maximizando F1 macro',
        'modelo_principal': PRINCIPAL, 'tipo_modelo': M, 'nombre_modelo': NOMBRES_MODELO[M], 'descripcion_modelo': DESCRIPCION_MODELO[M],
        'seleccion': 'ver Reporte_Seleccion_Modelo_sobre_Variables.md',
        'grupos_variables': G, 'nombres_grupos': meta_tabla['nombres_grupos'],
        'configuraciones': {k: {'variables': v[0], 'modelo': v[1]} for k, v in CONFIGS.items()},
    }}

    # 1. Validación cruzada
    print("\n[1/3] Validación cruzada con ablación", flush=True)
    externo = pliegues_externos(df)
    preds = pd.DataFrame({'id_texto': df['id_texto'], 'label_num': y, 'region': df['region'], 'Fuente': df['Fuente']})
    res['validacion_cruzada'] = {}
    pred_cv = {}
    for nombre, (vars_, tipo) in CONFIGS.items():
        p, pred, thetas = validacion_cruzada(df, df[vars_].values.astype(float), tipo, externo)
        pred_cv[nombre] = pred
        preds[f'p__{nombre}'] = p
        preds[f'pred__{nombre}'] = pred
        res['validacion_cruzada'][nombre] = {'umbrales_por_pliegue': thetas, **desglose(df, y, p, pred)}
        gm = res['validacion_cruzada'][nombre]['Global']
        print(f"  {nombre:<32} F1m={gm['f1_macro']:.3f} rec={gm['recall_falsa']:.3f} FP={gm['FP']} FN={gm['FN']} AUC={gm['roc_auc']:.3f} θ={np.round(thetas, 2).tolist()}", flush=True)
    sab_05 = (df['sabert_p_falsa'].values >= 0.5).astype(int)
    res['validacion_cruzada']['SaBERT (umbral nativo 0,5)'] = {'umbrales_por_pliegue': [0.5] * N_PLIEGUES, **desglose(df, y, df['sabert_p_falsa'].values, sab_05)}

    res['comparaciones_pareadas'] = {}
    for a in [PRINCIPAL, 'S+R+E (tres pilares)', 'Completo + SaBERT']:
        for region, m in [('Global', np.ones(len(y), bool)), ('España', (df['region'] == 'España').values),
                          ('América Latina', (df['region'] == 'América Latina').values)]:
            res['comparaciones_pareadas'][f'{a} vs SaBERT | {region}'] = diferencia_pareada(y[m], pred_cv[a][m], pred_cv['SaBERT'][m])
    for a in ['S+R+E+O', PRINCIPAL]:
        res['comparaciones_pareadas'][f'{a} vs S+R+E (tres pilares) | Global'] = diferencia_pareada(y, pred_cv[a], pred_cv['S+R+E (tres pilares)'])

    # 2. Transferencia entre regiones y 3. exclusión de una fuente
    CONFIGS_EXTERNAS = ['SaBERT', 'S+R+E (tres pilares)', PRINCIPAL, 'Completo + SaBERT']

    def evaluar_particion(m_tr, m_te):
        out = {}
        for nombre in CONFIGS_EXTERNAS:
            vars_, tipo = CONFIGS[nombre]
            X = df[vars_].values.astype(float)
            p, th = ajustar_y_predecir(X[m_tr], y[m_tr], g[m_tr], X[m_te], tipo)
            pred = (p >= th).astype(int)
            out[nombre] = {'umbral': th, **metricas(y[m_te], p, pred)}
            out[nombre]['_pred'] = pred
        for nombre in CONFIGS_EXTERNAS[1:]:
            out[nombre]['vs_SaBERT'] = diferencia_pareada(y[m_te], out[nombre]['_pred'], out['SaBERT']['_pred'])
        for nombre in CONFIGS_EXTERNAS:
            del out[nombre]['_pred']
        return out

    print("\n[2/3] Transferencia entre regiones", flush=True)
    res['transferencia'] = {}
    for origen, destino in [('España', 'América Latina'), ('América Latina', 'España')]:
        r = evaluar_particion((df['region'] == origen).values, (df['region'] == destino).values)
        res['transferencia'][f'{origen} → {destino}'] = r
        print(f"  {origen} → {destino}: " + " | ".join(f"{k}: F1m={v['f1_macro']:.3f} FN={v['FN']} FP={v['FP']}" for k, v in r.items()), flush=True)

    print("\n[3/3] Exclusión de una fuente por vez", flush=True)
    res['exclusion_fuente'] = {}
    for f in sorted(df['Fuente'].unique()):
        m = (df['Fuente'] == f).values
        r = evaluar_particion(~m, m)
        res['exclusion_fuente'][nombre_fuente(f)] = r
        print(f"  sin {nombre_fuente(f)}: " + " | ".join(f"{k}: F1m={v['f1_macro']:.3f}" for k, v in r.items()), flush=True)

    res['metadata']['duracion_s'] = time.time() - t0
    cfg.MODELO_VARIABLES_REPORTES_DIR.mkdir(parents=True, exist_ok=True)
    with open(cfg.METRICAS_MODELO_VARIABLES_SIN_FUGA_JSON, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False, default=float)
    preds.to_csv(cfg.PREDICCIONES_MODELO_VARIABLES_SIN_FUGA_CSV, index=False)
    print(f"\n[OK] {cfg.METRICAS_MODELO_VARIABLES_SIN_FUGA_JSON} ({res['metadata']['duracion_s'] / 60:.1f} min)")


if __name__ == '__main__':
    main()
