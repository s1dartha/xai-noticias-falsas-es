"""
Evaluación comparativa de modelos de emociones en español (sin ajuste fino).

Protocolo
---------
- 7 clases: las 6 emociones de Ekman (alegría, tristeza, ira, sorpresa, asco, miedo) + «others» (sin emoción dominante).
- Todos los modelos se usan tal como fueron publicados (revisión fija). No hay ajuste fino ni umbral: la clase
  predicha es la de mayor probabilidad.
- Las salidas de cada modelo se reordenan por NOMBRE de etiqueta (id2label) a un orden común, porque el orden de
  los índices difiere entre modelos.
- Cada modelo recibe el preprocesamiento con el que fue entrenado (pysentimiento o XLM-T).
- Criterio de referencia: F1 macro >= 0,65 (el mismo que en sensacionalismo).

Conjuntos
---------
  EmoEvent / EmoEvalEs (Plaza-del-Arco et al., LREC 2020; IberLEF 2021): tuits en español de la partición de
      prueba, anotados por personas. En splits/es/test.tsv las etiquetas están desplazadas 30 filas respecto de los
      textos; la etiqueta se toma del corpus completo emoevent_es.csv (ver etiquetas_test_corregidas). Los tres modelos se entrenaron con las particiones train/dev de este corpus,
      por lo que la de prueba no se vio en el entrenamiento.
  Corpus de noticias (dominio de la tesis): muestra estratificada por región y clase de unas 300 noticias del corpus
      ampliado (301 por el redondeo de cada estrato), sin etiquetas de emoción.
      Sirve para medir el acuerdo entre modelos (kappa de Cohen) y la distribución de emociones predichas.

Salidas: Reportes/metricas_modelos_emociones.json y cache/*.npy (probabilidades por modelo y conjunto).
"""

import sys
import os
import re
import json
import time
import importlib.util
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (f1_score, precision_recall_fscore_support, accuracy_score, confusion_matrix,
                             roc_auc_score, cohen_kappa_score)

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config_proyecto.py").exists())))
import config_proyecto as cfg

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

DIR = Path(__file__).resolve().parent
CACHE = DIR / "cache"
CACHE.mkdir(exist_ok=True)
EMOEVENT_DIR = cfg.EMOCIONES_DATASET_DIR / "externos" / "emoevent"
MAX_LEN_TUIT = 128
MAX_LEN_NOTICIA = 256
N_MUESTRA_NOTICIAS = 300
N_BOOT = 1000
CRITERIO_F1_MACRO = 0.65

EMOCIONES = ['others', 'joy', 'sadness', 'anger', 'surprise', 'disgust', 'fear']
NOMBRES_ES = {'others': 'sin emoción dominante', 'joy': 'alegría', 'sadness': 'tristeza', 'anger': 'ira',
              'surprise': 'sorpresa', 'disgust': 'asco', 'fear': 'miedo'}

MODELOS = [
    {'clave': 'robertuito_pysentimiento', 'id': 'pysentimiento/robertuito-emotion-analysis',
     'revision': 'bcd6835f4d1ab1a061bd7437c9d762623c8437ad', 'preproceso': 'pysentimiento',
     'referencia': 'Pérez et al. (2021, 2023), pysentimiento; RoBERTuito (Pérez et al., LREC 2022)'},
    {'clave': 'beto_pysentimiento', 'id': 'finiteautomata/beto-emotion-analysis',
     'revision': '9b628b0bd91471ad9bd709c10522c379ce09c32a', 'preproceso': 'pysentimiento',
     'referencia': 'Pérez et al. (2021), pysentimiento; BETO (Cañete et al., 2020)'},
    {'clave': 'xlmr_emoevales_gsiupm', 'id': 'daveni/twitter-xlm-roberta-emotion-es',
     'revision': 'ab57a1137b2eb1f6c90fc77b0a4c4ced7dbd4d60', 'preproceso': 'xlmt',
     'referencia': 'Vera et al. (2021), GSI-UPM, primer lugar en EmoEvalEs (IberLEF 2021); XLM-T (Barbieri et al., LREC 2022)'},
]


# ---------------------------------------------------------------------------
# PREPROCESAMIENTO
# ---------------------------------------------------------------------------
def _cargar_preprocess_pysentimiento():
    # Se carga solo pysentimiento/preprocessing.py: el paquete completo exige dependencias de entrenamiento (datasets)
    spec = importlib.util.find_spec('pysentimiento')
    ruta = Path(spec.submodule_search_locations[0]) / 'preprocessing.py'
    s = importlib.util.spec_from_file_location('pysentimiento_preprocessing', ruta)
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    return lambda t: mod.preprocess_tweet(t, lang='es')


def _preprocess_xlmt(t):
    # Preprocesamiento de XLM-T (Barbieri et al., 2022): menciones -> @user, enlaces -> http
    return ' '.join('@user' if w.startswith('@') and len(w) > 1 else 'http' if w.startswith('http') else w for w in t.split(' '))


PREPROCESOS = {'pysentimiento': _cargar_preprocess_pysentimiento(), 'xlmt': _preprocess_xlmt}


# ---------------------------------------------------------------------------
# CONJUNTOS
# ---------------------------------------------------------------------------
def clave_texto(x):
    """Clave para cruzar un tuit de las particiones con el corpus completo, que reemplaza enlaces, hashtags y
    usuarios por URL / HASHTAG / USER: se eliminan esos elementos, la puntuación y los espacios."""
    x = re.sub(r'https?://\S+|#\w+|@\w+|\b(?:URL|HASHTAG|USER)\b', ' ', str(x))
    return re.sub(r'[^0-9a-záéíóúüñ]+', '', unicodedata.normalize('NFKC', x).lower())


def etiquetas_test_corregidas():
    """La columna `emotion` de splits/es/test.tsv está desplazada 30 filas respecto de los textos (el archivo
    termina con 30 filas que solo tienen etiqueta). La etiqueta de cada tuit de prueba se toma del corpus completo
    publicado (emoevent_es.csv) cruzando por texto; la misma comprobación en train y dev da coincidencia total."""
    full = pd.read_csv(EMOEVENT_DIR / 'emoevent_es.csv', sep='\t', quoting=3)
    full['k'] = full['tweet'].map(clave_texto)
    ambiguas = full.groupby('k')['emotion'].nunique()
    ambiguas = set(ambiguas[ambiguas > 1].index)
    lk = full[~full['k'].isin(ambiguas)].drop_duplicates('k').set_index('k')['emotion']
    diag = {'fuente_etiquetas': 'emoevent_es.csv (corpus completo), cruce por texto normalizado'}
    for part in ['train', 'dev', 'test']:
        s = pd.read_csv(EMOEVENT_DIR / f'{part}.tsv', sep='\t', quoting=3)
        sin_texto = int(s['tweet'].isna().sum())
        s = s[s['tweet'].notna()].copy()
        s['emotion_corpus'] = s['tweet'].map(clave_texto).map(lk)
        enc = s['emotion_corpus'].notna()
        diag[part] = {'filas': int(len(s) + sin_texto), 'sin_texto': sin_texto, 'encontradas_en_corpus': int(enc.sum()),
                      'coincidencia_etiqueta_archivo_vs_corpus': float((s.loc[enc, 'emotion'] == s.loc[enc, 'emotion_corpus']).mean())}
        if part == 'test':
            t = s[enc].rename(columns={'emotion': 'emotion_archivo', 'emotion_corpus': 'emotion'})
    # En train hay 1 tuit de 5.683 con etiqueta distinta entre archivo y corpus; el cruce se valida con >= 99,9 %
    assert diag['train']['coincidencia_etiqueta_archivo_vs_corpus'] >= 0.999 and diag['dev']['coincidencia_etiqueta_archivo_vs_corpus'] >= 0.999, diag
    assert set(t['emotion']) <= set(EMOCIONES), set(t['emotion'])
    return t, diag


def cargar_conjuntos():
    t, diagnostico = etiquetas_test_corregidas()
    c = {'emoevent_test': {'textos': t['tweet'].astype(str).tolist(), 'y': np.array([EMOCIONES.index(e) for e in t['emotion']]),
                           'max_len': MAX_LEN_TUIT, 'dominio': 'tuits (EmoEvent, prueba)', 'etiquetado': True,
                           'diagnostico_etiquetas': diagnostico}}

    n = pd.read_csv(cfg.CORPUS_AMPLIADO_CSV).drop_duplicates(subset='Text')
    muestra = n.groupby(['region', 'clase_num']).sample(frac=N_MUESTRA_NOTICIAS / len(n), random_state=cfg.SEED)
    c['noticias_muestra'] = {'textos': muestra['Text'].astype(str).tolist(), 'y': None, 'max_len': MAX_LEN_NOTICIA,
                             'dominio': 'noticias del corpus de la tesis (sin etiquetas de emoción)', 'etiquetado': False,
                             'estratos': {f'{r} / {cfg.NOMBRES_CLASE[int(k)]}': int(v) for (r, k), v in muestra.groupby(['region', 'clase_num']).size().items()}}
    return c


# ---------------------------------------------------------------------------
# INFERENCIA
# ---------------------------------------------------------------------------
def inferir(tok, model, textos, max_len, batch=16):
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
    """Devuelve {conjunto: probabilidades (n, 7) en el orden EMOCIONES}, con caché por (modelo, conjunto)."""
    import torch
    torch.set_num_threads(os.cpu_count() or 4)
    meta_path = CACHE / f"{m['clave']}__meta.json"
    pendientes = [n for n in conjuntos if not (CACHE / f"{m['clave']}__{n}.npy").exists()]
    if pendientes:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        print(f"  cargando {m['id']} (revision {m['revision']})", flush=True)
        tok = AutoTokenizer.from_pretrained(m['id'], revision=m['revision'])
        model = AutoModelForSequenceClassification.from_pretrained(m['id'], revision=m['revision']).eval()
        id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
        assert sorted(id2label.values()) == sorted(EMOCIONES), id2label
        orden = [next(k for k, v in id2label.items() if v == e) for e in EMOCIONES]
        previo = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
        tiempos, max_len_efectivo = previo.get('tiempos_s', {}), previo.get('max_len_efectivo', {})
        # RoBERTuito admite como máximo 128 tokens (max_position_embeddings = 130): la longitud se acota por modelo
        tope = model.config.max_position_embeddings - 2
        pre = PREPROCESOS[m['preproceso']]
        for n in pendientes:
            d = conjuntos[n]
            max_len_efectivo[n] = min(d['max_len'], tope)
            t0 = time.time()
            p = inferir(tok, model, [pre(t) for t in d['textos']], max_len_efectivo[n])[:, orden]
            tiempos[n] = time.time() - t0
            np.save(CACHE / f"{m['clave']}__{n}.npy", p)
            print(f"    {n:<20} {len(d['textos']):>5} textos en {tiempos[n]:7.1f} s", flush=True)
        meta_path.write_text(json.dumps({'revision': m['revision'], 'id2label': id2label, 'orden_reindexado': orden,
                                         'max_len_efectivo': max_len_efectivo, 'tiempos_s': tiempos}, ensure_ascii=False, indent=1), encoding='utf-8')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    return {n: np.load(CACHE / f"{m['clave']}__{n}.npy") for n in conjuntos}, meta


# ---------------------------------------------------------------------------
# MÉTRICAS
# ---------------------------------------------------------------------------
def metricas(y, p):
    pred = p.argmax(1)
    k = list(range(len(EMOCIONES)))
    rng = np.random.default_rng(cfg.SEED)
    boots = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        boots.append(f1_score(y[i], pred[i], labels=k, average='macro', zero_division=0))
    pr, rc, f1, sop = precision_recall_fscore_support(y, pred, labels=k, zero_division=0)
    return {
        'n': int(len(y)),
        'accuracy': float(accuracy_score(y, pred)),
        'f1_macro': float(f1_score(y, pred, labels=k, average='macro', zero_division=0)),
        'f1_macro_ic95': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        'f1_macro_6_emociones': float(f1_score(y, pred, labels=k[1:], average='macro', zero_division=0)),
        'f1_ponderado': float(f1_score(y, pred, labels=k, average='weighted', zero_division=0)),
        'recall_macro': float(np.mean(rc)),
        'roc_auc_macro_ovr': float(roc_auc_score(y, p, multi_class='ovr', average='macro', labels=k)),
        'por_clase': {e: {'precision': float(pr[j]), 'recall': float(rc[j]), 'f1': float(f1[j]), 'soporte': int(sop[j])}
                      for j, e in enumerate(EMOCIONES)},
        'matriz_confusion': confusion_matrix(y, pred, labels=k).tolist(),
    }


def main():
    conjuntos = cargar_conjuntos()
    print("Conjuntos:", {n: len(d['textos']) for n, d in conjuntos.items()}, flush=True)
    res = {'metadata': {
        'clases': EMOCIONES, 'nombres_es': NOMBRES_ES, 'sin_ajuste_fino': True,
        'decision': 'argmax de las 7 probabilidades (sin umbral)',
        'criterio_referencia': f'F1 macro >= {CRITERIO_F1_MACRO} en emoevent_test',
        'max_len': {'tuit': MAX_LEN_TUIT, 'noticia': MAX_LEN_NOTICIA},
        'conjuntos': {n: {'n': len(d['textos']), 'dominio': d['dominio'], 'etiquetado': d['etiquetado'],
                          **({'distribucion': {e: int((d['y'] == j).sum()) for j, e in enumerate(EMOCIONES)}} if d['etiquetado'] else {}),
                          **({'estratos': d['estratos']} if 'estratos' in d else {}),
                          **({'diagnostico_etiquetas': d['diagnostico_etiquetas']} if 'diagnostico_etiquetas' in d else {})} for n, d in conjuntos.items()},
    }, 'modelos': {}, 'acuerdo_en_noticias': {}}
    preds_noticias = {}
    for m in MODELOS:
        print(f"\n=== {m['clave']} ({m['id']}) ===", flush=True)
        probs, meta = predecir(m, conjuntos)
        r = {'id': m['id'], 'revision': meta['revision'], 'referencia': m['referencia'], 'preproceso': m['preproceso'],
             'id2label': meta['id2label'], 'max_len_efectivo': meta['max_len_efectivo'], 'tiempos_s': meta['tiempos_s'],
             'ms_por_noticia': 1000 * meta['tiempos_s']['noticias_muestra'] / len(conjuntos['noticias_muestra']['textos'])}
        r['emoevent_test'] = metricas(conjuntos['emoevent_test']['y'], probs['emoevent_test'])
        pn = probs['noticias_muestra']
        preds_noticias[m['clave']] = pn.argmax(1)
        r['noticias_muestra'] = {
            'distribucion_predicha': {e: float((pn.argmax(1) == j).mean()) for j, e in enumerate(EMOCIONES)},
            'probabilidad_media': {e: float(pn[:, j].mean()) for j, e in enumerate(EMOCIONES)},
            'intensidad_emocional_media': float((1 - pn[:, 0]).mean()),
        }
        r['cumple_criterio'] = r['emoevent_test']['f1_macro'] >= CRITERIO_F1_MACRO
        res['modelos'][m['clave']] = r
        e = r['emoevent_test']
        print(f"  EmoEvent test: F1m={e['f1_macro']:.3f} {np.round(e['f1_macro_ic95'], 3)} Acc={e['accuracy']:.3f} "
              f"F1m6={e['f1_macro_6_emociones']:.3f} AUC={e['roc_auc_macro_ovr']:.3f} | noticias 'others'={r['noticias_muestra']['distribucion_predicha']['others']:.2f}", flush=True)
    claves = list(preds_noticias)
    for i in range(len(claves)):
        for j in range(i + 1, len(claves)):
            a, b = preds_noticias[claves[i]], preds_noticias[claves[j]]
            res['acuerdo_en_noticias'][f'{claves[i]} vs {claves[j]}'] = {
                'kappa_cohen': float(cohen_kappa_score(a, b)), 'acuerdo_bruto': float((a == b).mean())}
    cfg.EMOCIONES_REPORTES_DIR.mkdir(parents=True, exist_ok=True)
    with open(cfg.METRICAS_EMOCIONES_JSON, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"\n[OK] {cfg.METRICAS_EMOCIONES_JSON}")


if __name__ == '__main__':
    main()
