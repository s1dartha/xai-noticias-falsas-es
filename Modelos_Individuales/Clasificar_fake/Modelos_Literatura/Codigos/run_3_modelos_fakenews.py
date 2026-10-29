"""
Benchmark de 3 clasificadores de noticias falsas en español sobre el corpus base (N = 2.604).

Modelos evaluados (inferencia sin ajuste adicional):
  M1: Narrativaai/fake-news-detection-spanish
  M2: VerificadoProfesional/SaBERT-Spanish-Fake-News
  M3: Juanillaberia/spanish-fake-news-classifier

Las predicciones se guardan en una caché (.npz). Si la caché existe y corresponde al
corpus, el script recalcula las métricas sin volver a ejecutar los modelos; en ese
caso no requiere torch ni transformers.

Salidas (en Modelos_Literatura/Reportes):
  metricas_3_modelos_fakenews.json y predicciones_3_modelos_2604.csv
Al final llama a generar_reporte_3_modelos.py, que escribe Reporte_3_Modelos_FakeNews.md y sus figuras.
"""

import sys
import gc
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    average_precision_score
)

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config_proyecto.py").exists())))
import config_proyecto as cfg

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

print("=" * 80, flush=True)
print(" SISTEMA DE EVALUACION EXPERIMENTAL: 3 MODELOS PARA DETECCION DE FAKE NEWS", flush=True)
print("=" * 80, flush=True)

# =========================================================================
# 1. CARGA DEL CORPUS BASE Y ETIQUETAS CANONICAS (FAKE = 1, REAL = 0)
# =========================================================================
dataset_path = cfg.CORPUS_BASE_XLSX
if not dataset_path.exists():
    raise FileNotFoundError(f"No se encontro el corpus base en {dataset_path}")
print(f"[INFO] Dataset localizado en: {dataset_path}", flush=True)

df_raw = pd.read_excel(dataset_path)
col_text = 'Text' if 'Text' in df_raw.columns else 'Texto'

if 'label_num' in df_raw.columns:
    y_true = df_raw['label_num'].astype(int).values
else:
    y_true = df_raw['class'].apply(cfg.etiqueta_binaria).astype(int).values

# Control de consistencia: label_num debe coincidir con la columna booleana `class`
if 'class' in df_raw.columns:
    y_check = df_raw['class'].apply(cfg.etiqueta_binaria).astype(int).values
    if not np.array_equal(y_true, y_check):
        raise ValueError("Inconsistencia entre label_num y class en el corpus base.")

texts = df_raw[col_text].astype(str).tolist()
counts = np.bincount(y_true, minlength=2)
print(f"[INFO] Dataset: {len(df_raw)} registros. Clase 0 (Real): {counts[cfg.REAL]} | Clase 1 (Falsa): {counts[cfg.FAKE]}", flush=True)

# =========================================================================
# 2. CACHE DE PREDICCIONES
# =========================================================================
cache_file = cfg.CACHE_PREDICCIONES_3_MODELOS_NPZ
CLAVES_MODELOS = ['m1', 'm2', 'm3']
cached_data = {}
if cache_file.exists():
    try:
        cached_data = dict(np.load(cache_file, allow_pickle=True))
        if 'y_true' in cached_data and np.array_equal(cached_data['y_true'], y_true):
            print(f"[INFO] Cargada cache previa con claves: {sorted(cached_data.keys())}", flush=True)
        else:
            print("[WARN] La cache no corresponde al corpus actual; se descarta.", flush=True)
            cached_data = {}
    except Exception as e:
        print(f"[WARN] No se pudo leer cache previa: {e}", flush=True)
        cached_data = {}


def inferir_clasificador(modelo_cfg, textos, batch_size=16, max_length=256):
    """Ejecuta un clasificador de Hugging Face y devuelve P(clase positiva) y el tiempo total."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    torch.set_num_threads(max(1, __import__('os').cpu_count() or 4))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Dispositivo: {device}", flush=True)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(modelo_cfg['id'], revision=modelo_cfg['revision'])
    model = AutoModelForSequenceClassification.from_pretrained(modelo_cfg['id'], revision=modelo_cfg['revision'], dtype=torch.float32)
    model.to(device)
    model.eval()

    order = np.argsort([len(t) for t in textos])
    inv_order = np.argsort(order)
    sorted_texts = [textos[idx] for idx in order]
    probs_sorted = []
    with torch.inference_mode():
        for i in range(0, len(sorted_texts), batch_size):
            batch = sorted_texts[i:i + batch_size]
            inputs = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors='pt').to(device)
            logits = model(**inputs).logits.float()
            probs = torch.softmax(logits, dim=-1)
            probs_sorted.extend(probs[:, modelo_cfg['idx_positivo']].cpu().tolist())
            if (i // batch_size) % 25 == 0 or (i + batch_size >= len(sorted_texts)):
                print(f"  [{modelo_cfg['id']}] Procesados {min(i + batch_size, len(sorted_texts))}/{len(sorted_texts)}...", flush=True)

    prob = np.array([probs_sorted[idx] for idx in inv_order])
    tiempo = time.time() - t0
    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return prob, tiempo


# =========================================================================
# 3. EVALUACION DE LOS 3 MODELOS
# =========================================================================
MODELOS = [
    # clave, entrada en config, nombre de reporte, etiqueta corta, sufijo de archivos
    # M1 es RoBERTa-large-BNE (PlanTL) ajustado con FakeDeS, no BETO (ver config.json del modelo).
    ('m1', 'M1_Narrativaai', 'RoBERTa-large-BNE Fake News (Narrativaai)', 'RoBERTa-BNE (Narrativaai)', 'roberta_bne'),
    ('m2', 'M2_SaBERT', 'SaBERT (VerificadoProfesional)', 'SaBERT (VerificadoProf)', 'sabert'),
    ('m3', 'M3_Juanillaberia', 'Spanish Fake News (Juanillaberia)', 'BERT Seq (Juanillaberia)', 'juanillaberia'),
]
CLAVES_JSON = {'m1': 'RoBERTa_BNE_Narrativaai', 'm2': 'SaBERT_Verificado', 'm3': 'Spanish_FakeNews_Juanillaberia'}

# Ficha técnica tomada del config.json y la tarjeta de cada modelo en Hugging Face (consulta 2026-10-24)
FICHA_TECNICA = {
    'm1': {'Arquitectura': 'RoBERTa-large', 'Modelo_base': 'PlanTL-GOB-ES/roberta-large-bne', 'Capas': 24, 'Dim_oculta': 1024,
           'Cabezas': 16, 'Vocabulario': 50262, 'Datos_ajuste_declarados': 'FakeDeS (segun la tarjeta del modelo)'},
    'm2': {'Arquitectura': 'BERT-base', 'Modelo_base': 'dccuchile/bert-base-spanish-wwm-uncased', 'Capas': 12, 'Dim_oculta': 768,
           'Cabezas': 12, 'Vocabulario': 31002, 'Datos_ajuste_declarados': '125.000 noticias en español de varias regiones (UBA)'},
    'm3': {'Arquitectura': 'BERT-base', 'Modelo_base': 'dccuchile/bert-base-spanish-wwm-uncased', 'Capas': 12, 'Dim_oculta': 768,
           'Cabezas': 12, 'Vocabulario': 31002, 'Datos_ajuste_declarados': 'Ajuste secuencial: ~57k noticias cortas y ~2k noticias largas'},
}

# Nombre corto y region de cada repositorio del corpus base (la columna Fuente contiene la URL completa)
REPOSITORIOS = [
    ('Freiren/Unified-and-Balanced-Spanish-Fake-News-Corpus', 'Freiren', 'España'),
    ('Edds/spanish-fake-news-fixed', 'Edds', 'España'),
    ('mariagrandury/fake_news_corpus_spanish', 'FakeDeS (mariagrandury)', 'América Latina'),
    ('arseniitretiakov', 'Kaggle (arseniitretiakov)', 'Sin region asignada'),
]


def identificar_repositorio(url_fuente):
    for patron, nombre, region in REPOSITORIOS:
        if patron in url_fuente:
            return nombre, region
    raise ValueError(f"Fuente no reconocida en el corpus base: {url_fuente}")


def metricas_subconjunto(yt, pr, pb):
    tn, fp, fn, tp = confusion_matrix(yt, pr, labels=[cfg.REAL, cfg.FAKE]).ravel()
    dos_clases = len(np.unique(yt)) == 2
    return {
        'Accuracy': float(accuracy_score(yt, pr)),
        'F1_Macro': float(precision_recall_fscore_support(yt, pr, average='macro', zero_division=0)[2]),
        'Recall_Falsa': float(tp / (tp + fn)) if (tp + fn) else None,
        'FNR_Falsa': float(fn / (tp + fn)) if (tp + fn) else None,
        'Especificidad': float(tn / (tn + fp)) if (tn + fp) else None,
        'ROC_AUC': float(auc(*roc_curve(yt, pb, pos_label=cfg.FAKE)[:2])) if dos_clases else None,
        'AP': float(average_precision_score(yt, pb, pos_label=cfg.FAKE)) if dos_clases else None,
        'CM': [[int(tn), int(fp)], [int(fn), int(tp)]]
    }


results = {}
probs, preds = {}, {}

for n, (clave, cfg_key, nombre, etiqueta, sufijo) in enumerate(MODELOS, start=1):
    modelo_cfg = cfg.MODELOS_HF[cfg_key]
    print("\n" + "=" * 80, flush=True)
    print(f">>> [{n}/3] EVALUANDO MODELO {n}: {nombre} ({modelo_cfg['id']})...", flush=True)
    print("=" * 80, flush=True)

    if f'prob_{clave}' in cached_data and f'time_{clave}' in cached_data:
        print(f"[INFO] Cargando predicciones de {clave.upper()} desde cache...", flush=True)
        prob = np.asarray(cached_data[f'prob_{clave}'], dtype=float)
        tiempo = float(cached_data[f'time_{clave}'])
    else:
        prob, tiempo = inferir_clasificador(modelo_cfg, texts, batch_size=32 if clave == 'm3' else 16)
        cached_data.update({f'prob_{clave}': prob, f'time_{clave}': tiempo, 'y_true': y_true})

    # Regla de decision unica para los 3 modelos: clase positiva si P(FAKE) >= 0,5
    pred = (prob >= 0.5).astype(int)
    cached_data[f'pred_{clave}'] = pred
    probs[clave], preds[clave] = prob, pred

    lat = (tiempo / len(texts)) * 1000
    acc = accuracy_score(y_true, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", pos_label=cfg.FAKE, zero_division=0)
    prec_mac, rec_mac, f1_mac, _ = precision_recall_fscore_support(y_true, pred, average="macro", zero_division=0)
    fpr, tpr, _ = roc_curve(y_true, prob, pos_label=cfg.FAKE)
    auc_roc = auc(fpr, tpr)
    ap = average_precision_score(y_true, prob, pos_label=cfg.FAKE)
    cm = confusion_matrix(y_true, pred, labels=[cfg.REAL, cfg.FAKE])

    # Controles de consistencia de la matriz de confusion
    assert cm.sum() == len(y_true)
    assert cm[cfg.FAKE].sum() == counts[cfg.FAKE] and cm[cfg.REAL].sum() == counts[cfg.REAL]

    results[CLAVES_JSON[clave]] = {
        'Nombre': nombre,
        'Etiqueta_corta': etiqueta,
        'Sufijo': sufijo,
        'ID': modelo_cfg['id'],
        'Revision': modelo_cfg['revision'],
        'Indice_Clase_Falsa': modelo_cfg['idx_positivo'],
        'Ficha_Tecnica': FICHA_TECNICA[clave],
        'Accuracy': float(acc), 'Precision': float(prec), 'Recall': float(rec), 'F1-Score': float(f1),
        'Precision_Macro': float(prec_mac), 'Recall_Macro': float(rec_mac), 'F1_Macro': float(f1_mac),
        'AUC': float(auc_roc), 'AP': float(ap),
        'Tiempo Total (s)': float(tiempo), 'Latencia (ms)': float(lat),
        'CM': cm.tolist(),
        'CM_orden': '[[TN, FP], [FN, TP]] con filas = etiqueta real (0 Real, 1 Falsa)'
    }

    print(f"[OK] {clave.upper()} evaluado: {tiempo:.2f}s ({lat:.2f} ms/muestra)", flush=True)
    print(f"  Accuracy: {acc:.4f} | F1-Fake: {f1:.4f} | AUC-ROC: {auc_roc:.4f} | AP: {ap:.4f}", flush=True)

# Guardar cache solo con los 3 modelos evaluados
np.savez_compressed(
    cache_file,
    y_true=y_true,
    **{f'{tipo}_{c}': cached_data[f'{tipo}_{c}'] for c in CLAVES_MODELOS for tipo in ('prob', 'pred', 'time')}
)
print(f"[INFO] Cache de predicciones (3 modelos) guardada en: {cache_file}", flush=True)

df_preds_out = pd.DataFrame({'id': np.arange(len(y_true)), 'y_true': y_true})
for clave, _, _, _, sufijo in MODELOS:
    df_preds_out[f'pred_{clave}_{sufijo}'] = preds[clave]
    df_preds_out[f'prob_{clave}_{sufijo}'] = probs[clave]
df_preds_out.to_csv(cfg.PREDICCIONES_3_MODELOS_CSV, index=False)
print(f"[INFO] Predicciones y probabilidades individuales guardadas en: {cfg.PREDICCIONES_3_MODELOS_CSV}", flush=True)

# =========================================================================
# 4. DESGLOSE POR FUENTE Y POR REGION
# =========================================================================
urls_fuente = df_raw['Fuente'].astype(str).values
fuentes = np.array([identificar_repositorio(u)[0] for u in urls_fuente])
regiones = np.array([identificar_repositorio(u)[1] for u in urls_fuente])


def desglose(etiquetas, orden, url=False):
    out = {}
    for grupo in [g for g in orden if g in set(etiquetas)]:
        mask = etiquetas == grupo
        yt = y_true[mask]
        entrada = {'N': int(mask.sum()), 'Real_0': int((yt == cfg.REAL).sum()), 'Falsa_1': int((yt == cfg.FAKE).sum()), 'modelos': {}}
        if url:
            entrada = {'URL': str(urls_fuente[mask][0]), 'Region': next(r for _, nm, r in REPOSITORIOS if nm == grupo), **entrada}
        for clave in CLAVES_MODELOS:
            entrada['modelos'][CLAVES_JSON[clave]] = metricas_subconjunto(yt, preds[clave][mask], probs[clave][mask])
        out[grupo] = entrada
    return out


por_fuente = desglose(fuentes, [nm for _, nm, _ in REPOSITORIOS], url=True)
por_region = desglose(regiones, ['España', 'América Latina', 'Sin region asignada'])

print("\n=== DESGLOSE POR FUENTE (Accuracy / Recall Falsa) ===", flush=True)
for fuente, e in por_fuente.items():
    resumen = " | ".join(f"{k}: {v['Accuracy']:.4f} / {v['Recall_Falsa'] if v['Recall_Falsa'] is None else round(v['Recall_Falsa'], 4)}"
                         for k, v in e['modelos'].items())
    print(f"  {fuente} (N={e['N']}, {e['Region']}): {resumen}", flush=True)

# =========================================================================
# 5. JSON ESTRUCTURADO
# =========================================================================
with open(cfg.METRICAS_3_MODELOS_JSON, 'w', encoding='utf-8') as f:
    json.dump({
        'metadata': {
            'script': 'Modelos_Literatura/Codigos/run_3_modelos_fakenews.py',
            'dataset_path': str(dataset_path.relative_to(cfg.ROOT)).replace('\\', '/'),
            'total_registros': len(y_true),
            'textos_unicos': int(pd.Series(texts).nunique()),
            'clase_real_0': int(counts[cfg.REAL]),
            'clase_falsa_1': int(counts[cfg.FAKE]),
            'clase_positiva': 'Falsa (1)',
            'regla_decision': 'P(Falsa) >= 0.5',
            'truncamiento_tokens': 256,
            'origen_predicciones': 'cache predictions_cache_2604.npz (inferencia previa) o inferencia nueva',
            'nota_tiempos': 'Los tiempos y latencias provienen de la corrida que genero la cache y dependen del hardware.',
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        },
        'modelos': results,
        'por_region': por_region,
        'por_fuente': por_fuente
    }, f, indent=4, ensure_ascii=False)

print(f"[INFO] Metricas serializadas en: {cfg.METRICAS_3_MODELOS_JSON}", flush=True)

# =========================================================================
# 6. REPORTE Y FIGURAS
# =========================================================================
sys.path.insert(0, str(Path(__file__).resolve().parent))
import generar_reporte_3_modelos
generar_reporte_3_modelos.main()

print("\n" + "=" * 80, flush=True)
print(">>> EVALUACION COMPLETADA EXITOSAMENTE PARA LOS 3 MODELOS.", flush=True)
print("=" * 80, flush=True)
