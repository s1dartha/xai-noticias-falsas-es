"""
Evaluación comparativa de modelos de sensacionalismo / clickbait en español (sin ajuste fino).

Protocolo
---------
- Clase positiva (1) = sensacionalista / clickbait.
- Todos los modelos se usan tal como fueron publicados (revisión fija). No hay ajuste fino.
- Umbral de decisión: se reporta con 0,5 y con un umbral global elegido en los conjuntos de
  CALIBRACIÓN (TA1C validation + mitad de Salud), maximizando el F1 macro. Los conjuntos de
  PRUEBA no intervienen en la elección del umbral.
- Criterio de aceptación: F1 macro >= 0,65 en los conjuntos de prueba con etiquetas humanas.

Conjuntos
---------
  TA1C (Mordecki y Moncecchi, 2025, arXiv:2507.09777; tarea de IberLEF 2025): clickbait, 3 anotadores (validation = calibración, test = prueba),
      en dos entradas: solo titular y titular + cuerpo.
  Salud: 2.200 titulares de salud de medios mexicanos anotados manualmente por Díaz Román, Mendoza-Carreón y
      Florencia Juárez (repositorio gonzalezheber/Text-classifier-of-sensationalist-headlines-in-Spanish-using-BERT-based-models),
      partición 50/50 estratificada (calibración / prueba).
  Amarillismo humano: 114 noticias reales del repositorio EchuCompa/Sensationalism-Detector (solo prueba):
      trainset_realnews.csv (14) + trainset_paper_and_corpus.csv (59) + externos/echucompa/data_testset_realnews.csv (41).
      Se excluyen las noticias generadas con GPT (trainset_gptnews.csv del repositorio de origen).
  IA sintético: 70 textos generados con IA (control de cordura, solo prueba).
  NoticIA (García-Ferrero y Altuna, 2024): 100 noticias clickbait (solo exhaustividad).

Salidas: reportes/metricas_modelos_sensacionalismo.json y cache/*.npy (probabilidades por modelo y conjunto) y cache/*__meta.json.
"""

import sys
import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, precision_score,
                             recall_score, accuracy_score, confusion_matrix)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config_proyecto.py").exists())))
import config_proyecto as cfg

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

DIR = Path(__file__).resolve().parent
# Modo de prueba rápida (variable de entorno SMOKE=1): 8 textos por conjunto, caché y salida aparte
SMOKE = os.environ.get('SMOKE') == '1'
CACHE = DIR / ("cache_smoke" if SMOKE else "cache")
CACHE.mkdir(exist_ok=True)
REPORTE_JSON = (DIR / "smoke.json") if SMOKE else cfg.METRICAS_SENSACIONALISMO_JSON
EXT = cfg.SENSACIONALISMO_DATASET_DIR / "externos"
MAX_LEN_TITULAR = 128
MAX_LEN_CUERPO = 256   # misma longitud que usa el pipeline de explicabilidad para P_full
N_BOOT = 1000
CRITERIO_F1_MACRO = 0.65

# ---------------------------------------------------------------------------
# MODELOS (revisión fija). 'positiva' = nombre de la etiqueta positiva en id2label, o None si el
# config no la nombra (LABEL_0/LABEL_1): en ese caso el índice se determina SOLO con la partición de
# calibración de TA1C (el índice con ROC-AUC > 0,5), nunca con datos de prueba.
# ---------------------------------------------------------------------------
MODELOS = [
    {'clave': 'jjneila', 'id': 'JJNeila/bert-spanish-sensationalism-oss', 'revision': '1b8140efe548c876cfc03357c86100ede9118440',
     'tipo': 'clf', 'indice': 1, 'constructo': 'sensacionalismo (etiquetas probablemente generadas con LLM)'},
    {'clave': 'ggcastle_beto_clickbait', 'id': 'ggcastle/beto-clickbait-es', 'revision': None,
     'tipo': 'clf', 'positiva': 'clickbait', 'constructo': 'clickbait (entrenado con TA1C train)'},
    {'clave': 'dcere_ta1c_base', 'id': 'dcere/ta1c-Clickbait-Detector-es', 'revision': None,
     'tipo': 'clf', 'positiva': None, 'constructo': 'clickbait (IberLEF 2025 TA1C)'},
    {'clave': 'dcere_ta1c_large', 'id': 'dcere/ta1c-Clickbait-Detector-es-large', 'revision': None,
     'tipo': 'clf', 'positiva': None, 'constructo': 'clickbait (IberLEF 2025 TA1C, RoBERTa-large-BNE)'},
    {'clave': 'christinacdl_xlmr_clickbait', 'id': 'christinacdl/XLM_RoBERTa-Multilingual-Clickbait-Detection', 'revision': None,
     'tipo': 'clf', 'positiva': 'clickbait', 'constructo': 'clickbait (titulares en inglés traducidos)'},
]


def revision_actual(model_id):
    """Obtiene el sha del modelo en Hugging Face para fijarlo en el reporte."""
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(model_id).sha
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CONJUNTOS DE EVALUACIÓN
# ---------------------------------------------------------------------------
def unir(*partes):
    return '. '.join(str(p).strip() for p in partes if isinstance(p, str) and p.strip() and p.strip().lower() != 'nan')


def cargar_conjuntos():
    c = {}
    for part, rol in [('validation', 'calibracion'), ('test', 'prueba')]:
        t = pd.read_csv(EXT / 'ta1c' / f'TA1C_dataset_{part}_complete.csv')
        y = (t['Tag Value'] == 'Clickbait').astype(int).values
        c[f'ta1c_{part}_titular'] = {'textos': t['Article Title'].astype(str).tolist(), 'y': y, 'rol': rol,
                                     'unidad': 'titular', 'max_len': MAX_LEN_TITULAR, 'constructo': 'clickbait'}
        c[f'ta1c_{part}_cuerpo'] = {'textos': [unir(a, b) for a, b in zip(t['Article Title'], t['Article Cleaned Text'])], 'y': y,
                                    'rol': rol, 'unidad': 'titular+cuerpo', 'max_len': MAX_LEN_CUERPO, 'constructo': 'clickbait'}

    s = pd.read_csv(cfg.SENSACIONALISMO_DATASET_DIR / 'dataset_Salud.csv')
    s_cal, s_test = train_test_split(s, test_size=0.5, stratify=s['clase'], random_state=cfg.SEED)
    for nombre, d, rol in [('salud_calibracion', s_cal, 'calibracion'), ('salud_prueba', s_test, 'prueba')]:
        c[nombre] = {'textos': d['encabezado'].astype(str).tolist(), 'y': d['clase'].astype(int).values, 'rol': rol,
                     'unidad': 'titular', 'max_len': MAX_LEN_TITULAR, 'constructo': 'sensacionalismo (procedencia no documentada)'}

    D = cfg.SENSACIONALISMO_DATASET_DIR
    humanos = pd.concat([
        pd.read_csv(D / 'trainset_realnews.csv').rename(columns={'Copete/Resumen': 'Copete'}),
        pd.read_csv(D / 'trainset_paper_and_corpus.csv'),
        pd.read_csv(EXT / 'echucompa' / 'data_testset_realnews.csv'),
    ], ignore_index=True)
    y_h = humanos['Amarillismo'].map({'Amarillista': 1, 'No Amarillista': 0})
    assert y_h.notna().all()
    c['amarillismo_humano_titular'] = {'textos': humanos['Titular'].astype(str).tolist(), 'y': y_h.astype(int).values, 'rol': 'prueba',
                                       'unidad': 'titular', 'max_len': MAX_LEN_TITULAR, 'constructo': 'sensacionalismo / amarillismo'}
    c['amarillismo_humano_cuerpo'] = {'textos': [unir(a, b, d) for a, b, d in zip(humanos['Titular'], humanos['Copete'], humanos['Cuerpo'])],
                                      'y': y_h.astype(int).values, 'rol': 'prueba', 'unidad': 'titular+copete+cuerpo',
                                      'max_len': MAX_LEN_CUERPO, 'constructo': 'sensacionalismo / amarillismo'}

    ia = pd.read_csv(D / 'dataset_IA_sintetico_70.csv')
    c['ia_sintetico'] = {'textos': ia['Texto'].astype(str).tolist(), 'y': (ia['Clase'] == 'Sensacionalista').astype(int).values,
                         'rol': 'prueba', 'unidad': 'texto breve', 'max_len': MAX_LEN_TITULAR, 'constructo': 'sensacionalismo (sintético, control)'}

    n = pd.read_parquet(EXT / 'noticia' / 'test-00000-of-00001.parquet')
    col_t = 'web_headline' if 'web_headline' in n.columns else n.columns[0]
    col_c = 'web_text' if 'web_text' in n.columns else n.columns[1]
    c['noticia_clickbait'] = {'textos': [unir(a, b) for a, b in zip(n[col_t], n[col_c])], 'y': np.ones(len(n), dtype=int),
                              'rol': 'prueba_solo_recall', 'unidad': 'titular+cuerpo', 'max_len': MAX_LEN_CUERPO, 'constructo': 'clickbait (todos positivos)'}
    return c


# ---------------------------------------------------------------------------
# INFERENCIA
# ---------------------------------------------------------------------------
def inferir_clf(tok, model, textos, max_len, batch=16):
    import torch
    salida = []
    orden = np.argsort([len(t) for t in textos])
    with torch.inference_mode():
        for i in range(0, len(textos), batch):
            lote = [textos[j] for j in orden[i:i + batch]]
            enc = tok(lote, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
            salida.append(torch.softmax(model(**enc).logits.float(), -1).numpy())
    probs = np.concatenate(salida)
    inv = np.empty_like(orden)
    inv[orden] = np.arange(len(orden))
    return probs[inv]


def predecir(m, conjuntos):
    """Devuelve {conjunto: P(positiva)} usando caché por (modelo, conjunto)."""
    import torch
    torch.set_num_threads(os.cpu_count() or 4)
    pendientes = [n for n in conjuntos if not (CACHE / f"{m['clave']}__{n}.npy").exists()]
    meta_path = CACHE / f"{m['clave']}__meta.json"
    if pendientes:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        rev = m['revision'] or revision_actual(m['id'])
        print(f"  cargando {m['id']} (revision {rev})", flush=True)
        tok = AutoTokenizer.from_pretrained(m['id'], revision=rev)
        model = AutoModelForSequenceClassification.from_pretrained(m['id'], revision=rev).eval()
        id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
        tiempos = json.loads(meta_path.read_text(encoding='utf-8'))['tiempos_s'] if meta_path.exists() else {}
        for n in pendientes:
            d = conjuntos[n]
            t0 = time.time()
            p = inferir_clf(tok, model, d['textos'], d['max_len'])
            tiempos[n] = time.time() - t0
            np.save(CACHE / f"{m['clave']}__{n}.npy", p)
            print(f"    {n:<30} {len(d['textos']):>5} textos en {tiempos[n]:7.1f} s", flush=True)
        meta_path.write_text(json.dumps({'revision': rev, 'id2label': id2label, 'tiempos_s': tiempos}, ensure_ascii=False, indent=1), encoding='utf-8')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    crudas = {n: np.load(CACHE / f"{m['clave']}__{n}.npy") for n in conjuntos}
    return crudas, meta


def prob_positiva(m, crudas, meta, conjuntos):
    """Convierte las salidas a P(positiva) y documenta cómo se eligió el índice."""
    if 'indice' in m:
        idx, regla = m['indice'], f"índice {m['indice']} fijado (tarjeta del modelo)"
    elif m.get('positiva'):
        cand = [int(k) for k, v in meta['id2label'].items() if v.strip().lower() == m['positiva']]
        assert len(cand) == 1, (m['id'], meta['id2label'])
        idx, regla = cand[0], f"id2label[{cand[0]}] = {meta['id2label'][str(cand[0])]}"
    else:
        y = conjuntos['ta1c_validation_cuerpo']['y']
        aucs = {k: roc_auc_score(y, crudas['ta1c_validation_cuerpo'][:, k]) for k in range(crudas['ta1c_validation_cuerpo'].shape[1])}
        idx = max(aucs, key=aucs.get)
        regla = f"id2label sin nombres; índice {idx} elegido con TA1C validation (AUC por índice: { {k: round(v, 3) for k, v in aucs.items()} })"
    return {n: v[:, idx] for n, v in crudas.items()}, regla


# ---------------------------------------------------------------------------
# MÉTRICAS
# ---------------------------------------------------------------------------
def metricas(y, p, tau):
    pred = (p >= tau).astype(int)
    out = {'n': int(len(y)), 'positivos': int(y.sum()), 'umbral': float(tau),
           'recall': float(recall_score(y, pred, zero_division=0)),
           'cm': confusion_matrix(y, pred, labels=[0, 1]).tolist()}
    if len(np.unique(y)) == 2:
        rng = np.random.default_rng(cfg.SEED)
        boots = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(y), len(y))
            if len(np.unique(y[i])) == 2:
                boots.append(f1_score(y[i], pred[i], average='macro', zero_division=0))
        out.update({
            'accuracy': float(accuracy_score(y, pred)),
            'precision': float(precision_score(y, pred, zero_division=0)),
            'f1_positiva': float(f1_score(y, pred, zero_division=0)),
            'f1_macro': float(f1_score(y, pred, average='macro', zero_division=0)),
            'f1_macro_ic95': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            'roc_auc': float(roc_auc_score(y, p)),
            'pr_auc': float(average_precision_score(y, p)),
            'especificidad': float(((pred == 0) & (y == 0)).sum() / max(1, (y == 0).sum())),
        })
    return out


def umbral_f1_macro(y, p):
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    puntajes = [f1_score(y, (p >= t).astype(int), average='macro', zero_division=0) for t in grid]
    return float(grid[int(np.argmax(puntajes))])


def main():
    conjuntos = cargar_conjuntos()
    if SMOKE:
        for d in conjuntos.values():
            idx = np.r_[np.where(d['y'] == 1)[0][:4], np.where(d['y'] == 0)[0][:4]]
            d['textos'] = [d['textos'][i] for i in idx]
            d['y'] = d['y'][idx]
    print("Conjuntos:", {n: (len(d['y']), int(d['y'].sum()), d['rol']) for n, d in conjuntos.items()}, flush=True)
    calib = ['ta1c_validation_cuerpo', 'salud_calibracion']
    principales = ['ta1c_test_cuerpo', 'salud_prueba', 'amarillismo_humano_cuerpo']
    resultados = {'metadata': {
        'clase_positiva': '1 = sensacionalista / clickbait', 'sin_ajuste_fino': True,
        'umbral': 'global, maximiza F1 macro en ' + ' + '.join(calib) + ' (los conjuntos de prueba no se usan)',
        'criterio_aceptacion': f'F1 macro >= {CRITERIO_F1_MACRO} en {principales}',
        'max_len': {'titular': MAX_LEN_TITULAR, 'cuerpo': MAX_LEN_CUERPO},
        'conjuntos': {n: {'n': int(len(d['y'])), 'positivos': int(d['y'].sum()), 'rol': d['rol'], 'unidad': d['unidad'],
                          'constructo': d['constructo']} for n, d in conjuntos.items()},
    }, 'modelos': {}}
    for m in MODELOS:
        print(f"\n=== {m['clave']} ({m['id']}) ===", flush=True)
        try:
            crudas, meta = predecir(m, conjuntos)
        except Exception as e:
            print(f"  [ERROR] {e}", flush=True)
            resultados['modelos'][m['clave']] = {'id': m['id'], 'error': str(e)}
            continue
        probs, regla = prob_positiva(m, crudas, meta, conjuntos)
        y_cal = np.concatenate([conjuntos[n]['y'] for n in calib])
        p_cal = np.concatenate([probs[n] for n in calib])
        tau = umbral_f1_macro(y_cal, p_cal)
        r = {'id': m['id'], 'revision': meta['revision'], 'constructo': m['constructo'], 'id2label': meta['id2label'],
             'regla_indice': regla, 'umbral_calibrado': tau, 'tiempos_s': meta['tiempos_s'], 'conjuntos': {}}
        for n, d in conjuntos.items():
            r['conjuntos'][n] = {'umbral_0_5': metricas(d['y'], probs[n], 0.5), 'umbral_calibrado': metricas(d['y'], probs[n], tau)}
        f1s = [r['conjuntos'][n]['umbral_calibrado']['f1_macro'] for n in principales]
        r['f1_macro_medio_principales'] = float(np.mean(f1s))
        r['cumple_criterio'] = bool(all(f >= CRITERIO_F1_MACRO for f in f1s))
        resultados['modelos'][m['clave']] = r
        print(f"  tau={tau:.2f} | " + " | ".join(
            f"{n}: F1m={r['conjuntos'][n]['umbral_calibrado']['f1_macro']:.3f} R={r['conjuntos'][n]['umbral_calibrado']['recall']:.3f}"
            for n in principales), flush=True)
        with open(REPORTE_JSON, 'w', encoding='utf-8') as f:
            json.dump(resultados, f, indent=1, ensure_ascii=False)
    print(f"\n[OK] {REPORTE_JSON}")


if __name__ == '__main__':
    main()
