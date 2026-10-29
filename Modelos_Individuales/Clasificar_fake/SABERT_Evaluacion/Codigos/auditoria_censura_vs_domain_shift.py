"""
Auditoría de SaBERT: ¿cae en América Latina por las máscaras de anonimización (*NUMBER*, *URL*, *PHONE*)
o por el cambio de dominio (países, entidades y estilo)?

Experimentos (sobre el corpus ampliado de 4.418 noticias y las predicciones guardadas en
Dataset/predicciones_sabert_ampliado_4418.csv):
  1. Conteo de máscaras por corpus de origen y por clase.
  2. Métricas estratificadas con las predicciones originales: España, América Latina, Omdena (sin máscaras),
     MEX-A3T + FakeDeS y, dentro de MEX-A3T (que concentra casi todas las máscaras), textos con y sin máscara.
  3. Contrafáctico A: en las noticias de España se reemplazan cifras y URL por *NUMBER* y *URL*
     (se fija el dominio y se introduce solo la máscara).
  4. Contrafáctico B: en MEX-A3T + FakeDeS se reemplazan las máscaras por valores plausibles
     (se fija el dominio y se quita solo la máscara).
  5. Diferencias (deltas) de cada contrafáctico frente a su original y de Omdena frente a España.
  6. Verificación de procedencia: se recalcula SaBERT (revisión fija de config_proyecto) en una muestra
     estratificada y se compara con las probabilidades guardadas.

Las probabilidades de los contrafácticos se guardan en Reportes/datos/ para no repetir la inferencia.
Salida: Reportes/auditoria_censura_vs_domain_shift.json. Las figuras y el reporte los genera generar_reporte_sabert.py.

Uso: python auditoria_censura_vs_domain_shift.py [--recalcular]
"""

import re
import sys
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config_proyecto.py").exists())))
import config_proyecto as cfg

DATOS = cfg.SABERT_REPORTES_DIR / 'datos'
CACHE_A = DATOS / 'probs_contrafactico_espana_enmascarada.csv'
CACHE_B = DATOS / 'probs_contrafactico_mex_fakedes_restaurado.csv'
CACHE_VERIF = DATOS / 'probs_verificacion_muestra.csv'
SALIDA_JSON = cfg.SABERT_REPORTES_DIR / 'auditoria_censura_vs_domain_shift.json'
N_VERIFICACION = 60
BATCH = 16

PAT_NUMBER = re.compile(r"\*(?:NUMBER|\d+)\*", re.IGNORECASE)
PAT_URL = re.compile(r"\*URL\*", re.IGNORECASE)
PAT_PHONE = re.compile(r"\*PHONE\*", re.IGNORECASE)
PAT_MASCARA = re.compile(r"\*(?:NUMBER|URL|PHONE|\d+)\*", re.IGNORECASE)
PAT_URL_REAL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
PAT_NUM_REAL = re.compile(r"\b\d+(?:[\.,]\d+)?\b")


def metricas(y, p_fake, umbral=0.5):
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                                 average_precision_score, confusion_matrix)
    y = np.asarray(y, int)
    p_fake = np.asarray(p_fake, float)
    pred = (p_fake >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        'n': int(len(y)), 'n_real': int((y == 0).sum()), 'n_falsas': int((y == 1).sum()),
        'exactitud': float(accuracy_score(y, pred)),
        'f1_macro': float(f1_score(y, pred, average='macro', zero_division=0)),
        'precision_falsa': float(precision_score(y, pred, pos_label=1, zero_division=0)),
        'recall_falsa': float(recall_score(y, pred, pos_label=1, zero_division=0)),
        'f1_falsa': float(f1_score(y, pred, pos_label=1, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, p_fake)) if len(np.unique(y)) > 1 else None,
        'ap': float(average_precision_score(y, p_fake)) if len(np.unique(y)) > 1 else None,
        'p_falsa_media': float(p_fake.mean()),
        'tasa_falsos_negativos': float(fn / (fn + tp)) if fn + tp else None,
        'tasa_falsos_positivos': float(fp / (fp + tn)) if fp + tn else None,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


class SaBERT:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        m = cfg.MODELOS_HF['M2_SaBERT']
        self.torch, self.idx = torch, m['idx_positivo']
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tok = AutoTokenizer.from_pretrained(m['id'], revision=m['revision'])
        self.model = AutoModelForSequenceClassification.from_pretrained(m['id'], revision=m['revision']).eval().to(self.dev)
        self.ref = f"{m['id']}@{m['revision']}"

    def p_falsa(self, textos):
        textos = [unicodedata.normalize('NFC', str(t)) for t in textos]  # tildes NFD de Omdena (ver Analisis_complementarios/corregir_nfc.py)
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(textos), BATCH):
                enc = self.tok(textos[i:i + BATCH], padding=True, truncation=True, max_length=256, return_tensors='pt').to(self.dev)
                out.append(self.torch.softmax(self.model(**enc).logits.float(), -1)[:, self.idx].cpu().numpy())
                if (i // BATCH) % 20 == 0:
                    print(f"    {i + len(textos[i:i + BATCH])}/{len(textos)}", flush=True)
        return np.concatenate(out)


def con_cache(ruta, ids, textos, modelo, recalcular):
    if ruta.exists() and not recalcular:
        c = pd.read_csv(ruta)
        if len(c) == len(textos) and (c['texto'].astype(str).values == np.array(textos, dtype=str)).all():
            return c['p_falsa'].values
    p = modelo().p_falsa(textos)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'fila': ids, 'texto': textos, 'p_falsa': p}).to_csv(ruta, index=False)
    return p


def enmascarar(t):
    t2 = PAT_URL_REAL.sub('*URL*', t)
    n = len(PAT_NUM_REAL.findall(t2))
    return PAT_NUM_REAL.sub('*NUMBER*', t2), n


def restaurar(t):
    return PAT_NUMBER.sub('20', PAT_PHONE.sub('5512345678', PAT_URL.sub('https://www.portalnoticias.com', t)))


def delta(a, b, desc):
    return {'descripcion': desc, **{f'delta_{k}': (b[k] - a[k]) for k in ('exactitud', 'f1_macro', 'recall_falsa', 'f1_falsa', 'roc_auc', 'ap',
                                                                         'tasa_falsos_negativos', 'tasa_falsos_positivos')}}


def main():
    recalcular = '--recalcular' in sys.argv
    df = pd.read_csv(cfg.PREDICCIONES_SABERT_4418_CSV)
    corpus = pd.read_csv(cfg.CORPUS_AMPLIADO_CSV)
    assert (df['Text'].astype(str).values == corpus['Text'].astype(str).values).all(), "Predicciones desalineadas con el corpus"
    assert (df['clase_num'].values == corpus['clase_num'].values).all(), "Etiquetas de las predicciones distintas a las del corpus vigente"
    y, p = df['clase_num'].astype(int).values, df['prob_fake'].astype(float).values
    txt = df['Text'].astype(str)

    df['n_number'] = txt.apply(lambda s: len(PAT_NUMBER.findall(s)))
    df['n_url'] = txt.apply(lambda s: len(PAT_URL.findall(s)))
    df['n_phone'] = txt.apply(lambda s: len(PAT_PHONE.findall(s)))
    df['con_mascara'] = txt.apply(lambda s: bool(PAT_MASCARA.search(s)))

    mascaras = {}
    for fuente, g in df.groupby('dataset_origen'):
        mascaras[fuente] = {
            'region': g['region'].iloc[0], 'n': int(len(g)), 'n_con_mascara': int(g['con_mascara'].sum()),
            'pct_con_mascara': float(g['con_mascara'].mean()),
            'ocurrencias_number': int(g['n_number'].sum()), 'ocurrencias_url': int(g['n_url'].sum()), 'ocurrencias_phone': int(g['n_phone'].sum()),
            'pct_reales_con_mascara': float(g.loc[g['clase_num'] == 0, 'con_mascara'].mean()) if (g['clase_num'] == 0).any() else None,
            'pct_falsas_con_mascara': float(g.loc[g['clase_num'] == 1, 'con_mascara'].mean()) if (g['clase_num'] == 1).any() else None,
        }

    es = (df['region'] == 'España').values
    lat = (df['region'] == 'América Latina').values
    omd = df['dataset_origen'].str.contains('Omdena', case=False).values
    mex = df['dataset_origen'].str.contains('MEX-A3T|FakeDeS', case=False).values
    con = df['con_mascara'].values
    a3t = df['dataset_origen'].str.contains('MEX-A3T', case=False).values
    grupos = {'global': np.ones(len(df), bool), 'espana': es, 'america_latina': lat, 'omdena': omd,
              'mex_fakedes': mex, 'mex_fakedes_con_mascara': mex & con, 'mex_fakedes_sin_mascara': mex & ~con,
              # Las máscaras están casi solo en MEX-A3T: la comparación limpia es dentro de ese corpus
              'mex_a3t_con_mascara': a3t & con, 'mex_a3t_sin_mascara': a3t & ~con}
    estrat = {k: metricas(y[m], p[m]) for k, m in grupos.items()}

    modelo_cargado = {}

    def modelo():
        if 'm' not in modelo_cargado:
            print(f"[SaBERT] cargando {cfg.MODELOS_HF['M2_SaBERT']['id']}", flush=True)
            modelo_cargado['m'] = SaBERT()
        return modelo_cargado['m']

    # Contrafáctico A: España con máscaras sintéticas
    idx_es = np.where(es)[0]
    pares = [enmascarar(t) for t in txt.values[idx_es]]
    t_es_m = [a for a, _ in pares]
    print("[A] España enmascarada", flush=True)
    p_es_m = con_cache(CACHE_A, idx_es, t_es_m, modelo, recalcular)
    m_es_m = metricas(y[es], p_es_m)
    m_es_m.update({'n_textos_modificados': int(sum(a != b for a, b in zip(t_es_m, txt.values[idx_es]))),
                   'cifras_enmascaradas': int(sum(n for _, n in pares))})
    m_es_m['pct_textos_modificados'] = m_es_m['n_textos_modificados'] / len(idx_es)

    # Contrafáctico B: MEX-A3T + FakeDeS sin máscaras
    idx_mx = np.where(mex)[0]
    t_mx_r = [restaurar(t) for t in txt.values[idx_mx]]
    print("[B] MEX-A3T + FakeDeS restaurado", flush=True)
    p_mx_r = con_cache(CACHE_B, idx_mx, t_mx_r, modelo, recalcular)
    m_mx_r = metricas(y[mex], p_mx_r)
    m_mx_r['n_textos_modificados'] = int(sum(a != b for a, b in zip(t_mx_r, txt.values[idx_mx])))

    # Verificación de procedencia de las probabilidades guardadas
    rng = np.random.default_rng(cfg.SEED)
    idx_v = np.sort(np.concatenate([rng.choice(np.where(df['dataset_origen'] == d)[0], N_VERIFICACION // 5, replace=False)
                                    for d in df['dataset_origen'].unique()]))
    print("[Verificación] muestra de las probabilidades guardadas", flush=True)
    p_v = con_cache(CACHE_VERIF, idx_v, list(txt.values[idx_v]), modelo, recalcular)
    dif = np.abs(p_v - p[idx_v])
    verif = {'modelo': f"{cfg.MODELOS_HF['M2_SaBERT']['id']}@{cfg.MODELOS_HF['M2_SaBERT']['revision']}",
             'n_muestra': int(len(idx_v)), 'dif_abs_max': float(dif.max()), 'dif_abs_media': float(dif.mean()),
             'misma_prediccion': float(np.mean((p_v >= 0.5) == (p[idx_v] >= 0.5))),
             'conclusion': ('las probabilidades guardadas coinciden con SaBERT en su revisión fija'
                            if dif.max() < 1e-3 else 'las probabilidades guardadas NO coinciden exactamente con SaBERT en su revisión fija')}

    res = {
        'metadata': {'corpus': str(cfg.CORPUS_AMPLIADO_CSV.name), 'predicciones': str(cfg.PREDICCIONES_SABERT_4418_CSV.name),
                     'modelo': verif['modelo'], 'umbral': 0.5, 'clase_positiva': '1 = falsa (P(falsa) = salida idx_positivo de SaBERT)',
                     'max_len': 256},
        'verificacion_procedencia': verif,
        'mascaras_por_corpus': mascaras,
        'metricas_estratificadas': estrat,
        'contrafactico_espana_enmascarada': m_es_m,
        'contrafactico_mex_fakedes_restaurado': m_mx_r,
        'deltas': {
            'enmascarar_espana': delta(estrat['espana'], m_es_m, 'España enmascarada − España original (efecto de introducir máscaras)'),
            'restaurar_mex_fakedes': delta(estrat['mex_fakedes'], m_mx_r, 'MEX-A3T + FakeDeS restaurado − original (efecto de quitar máscaras)'),
            'omdena_vs_espana': delta(estrat['espana'], estrat['omdena'], 'Omdena (sin máscaras) − España (cambio de dominio sin máscaras)'),
            'america_latina_vs_espana': delta(estrat['espana'], estrat['america_latina'], 'América Latina − España (caída total)'),
            'mex_a3t_con_vs_sin_mascara': delta(estrat['mex_a3t_sin_mascara'], estrat['mex_a3t_con_mascara'],
                                                'MEX-A3T con máscara − sin máscara (mismo corpus; no es experimental: los textos difieren)'),
        },
    }
    SALIDA_JSON.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"[OK] {SALIDA_JSON}")
    print(json.dumps(verif, ensure_ascii=False))


if __name__ == '__main__':
    main()
