"""
Fase 4a: tabla única de variables por noticia (corpus ampliado, N = 4.418).

La usan el modelo sobre variables (evaluar_modelo_sobre_variables_sin_fuga.py), el IML y los estilos discursivos. Ninguna variable se calcula
con información de la etiqueta. Los valores faltantes se dejan como NaN y se tratan dentro de cada pliegue, con una
excepción heredada de la extracción de origen: red_consecutivas_media y red_consecutivas_min ya vienen imputadas con la
mediana del corpus completo en las noticias de una sola oración (extraer_features_*.py). Su efecto se examina en el
análisis de sensibilidad de la tesis (Tabla 4.15; Analisis_complementarios/sensibilidad_clasificador.py).

Pilares y grupos de variables
-----------------------------
  S  Sensacionalismo (JJNeila, índice 1): probabilidad del documento y de sus oraciones.
  R  Redundancia (hiiamsid): similitud coseno entre oraciones, sin umbrales.
  E  Emociones (RoBERTuito, pysentimiento): probabilidades por oración agregadas por documento.
  O  Marcadores ortotipográficos: mayúsculas, exclamaciones, interrogaciones, puntos suspensivos, comillas.
  L  Rasgos lingüísticos: marcadores epistémicos, legibilidad, riqueza léxica, morfosintaxis, anclajes, longitud.

S, R, O y L se toman de features_modelo_sobre_variables_ampliado_avanzado_4418.csv tras comprobar que coinciden con un recálculo
independiente (revisión fija de cada modelo). E se calcula aquí, con caché por oración.
"""

import sys
import os
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config_proyecto.py").exists())))
import config_proyecto as cfg

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

CACHE = cfg.MODELO_VARIABLES_DIR / "cache"
CACHE.mkdir(exist_ok=True)
N_VERIFICACION = 40

# Nombre en la tabla -> columna de origen en features_modelo_sobre_variables_ampliado_avanzado_4418.csv
VARIABLES_REUTILIZADAS = {
    'S': {'sens_p_documento': 'P_full', 'sens_p_media_oraciones': 'P_mean', 'sens_p_max_oraciones': 'P_max',
          'sens_sd_oraciones': 'sigma_sens'},
    'R': {'red_sim_max': 'max_intra_similarity', 'red_sim_media': 'mean_intra_similarity',
          'red_sim_top3': 'top3_mean_intra_similarity', 'red_sim_varianza': 'var_intra_similarity',
          'red_consecutivas_media': 'consec_sim_mean', 'red_consecutivas_min': 'consec_sim_min'},
    'O': {'orto_mayusculas_sostenidas': 'all_caps_ratio', 'orto_proporcion_mayusculas': 'upper_chars_ratio',
          'orto_exclamaciones': 'excl_density', 'orto_interrogaciones': 'quest_density',
          'orto_puntos_suspensivos': 'ellipsis_density', 'orto_comillas': 'quotes_density'},
    'L': {'ling_atenuadores': 'hedges_density', 'ling_intensificadores': 'boosters_density',
          'ling_verbos_dicendi': 'dicendi_density', 'ling_flesch_szigriszt': 'flesch_szigriszt',
          'ling_guiraud': 'guiraud_ttr', 'ling_hapax': 'hapax_ratio', 'ling_adjetivos_sustantivos': 'adj_noun_ratio',
          'ling_pronombres_1p': 'pron_1p_density', 'ling_pronombres_3p': 'pron_3p_density', 'ling_adverbios': 'adv_density',
          'ling_cifras': 'numbers_density', 'ling_porcentajes': 'percent_count', 'ling_referencias_temporales': 'temporal_density',
          'ling_palabras': 'conteo_palabras_text', 'ling_oraciones': 'num_sentences'},
}
EMOCIONES_USADAS = {'anger': 'ira', 'fear': 'miedo', 'sadness': 'tristeza', 'joy': 'alegria', 'surprise': 'sorpresa'}


def segmentar(texto):
    # Misma segmentación que la extracción de redundancia y sensacionalismo
    import nltk
    sents = [s.strip() for s in nltk.sent_tokenize(str(texto), language='spanish') if len(s.strip()) > 8]
    return sents if sents else [str(texto).strip()[:200]]


def id_texto(texto):
    return hashlib.sha1(' '.join(str(texto).lower().split()).encode('utf-8')).hexdigest()[:16]


def inferir(tok, model, textos, max_len, batch=32):
    import torch
    salida = []
    orden = np.argsort([len(t) for t in textos])
    with torch.inference_mode():
        for i in range(0, len(textos), batch):
            lote = [textos[j] for j in orden[i:i + batch]]
            enc = tok(lote, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
            salida.append(torch.softmax(model(**enc).logits.float(), -1).numpy())
            if (i // batch) % 100 == 0:
                print(f"    {i:>6}/{len(textos)}", flush=True)
    probs = np.concatenate(salida)
    inv = np.empty_like(orden)
    inv[orden] = np.arange(len(orden))
    return probs[inv]


def verificar_reutilizadas(df, oraciones):
    """Recalcula S y R en una muestra con revisiones fijas y compara con los valores reutilizados."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(os.cpu_count() or 4)
    idx = np.random.default_rng(cfg.SEED).choice(len(df), N_VERIFICACION, replace=False)
    ms = cfg.MODELOS_HF['Sensacionalismo_JJNeila']
    tok = AutoTokenizer.from_pretrained(ms['id'], revision=ms['revision'])
    mod = AutoModelForSequenceClassification.from_pretrained(ms['id'], revision=ms['revision']).eval()
    textos = [str(df['Text'].iloc[i])[:1000] for i in idx]
    p_doc = inferir(tok, mod, textos, 256)[:, ms['idx_positivo']]
    p_media = [inferir(tok, mod, oraciones[i], 128)[:, ms['idx_positivo']].mean() for i in idx]
    mr = cfg.MODELOS_HF['SBERT_Redundancia']
    sbert = SentenceTransformer(mr['id'], revision=mr['revision'], device='cpu')
    sim_max = []
    for i in idx:
        if len(oraciones[i]) < 2:
            sim_max.append(np.nan)
            continue
        e = sbert.encode(oraciones[i], normalize_embeddings=True)
        m = e @ e.T
        sim_max.append(m[np.triu_indices(len(e), 1)].max())
    dif = {
        'sens_p_documento': float(np.nanmax(np.abs(p_doc - df['P_full'].values[idx]))),
        'sens_p_media_oraciones': float(np.nanmax(np.abs(np.array(p_media) - df['P_mean'].values[idx]))),
        'red_sim_max': float(np.nanmax(np.abs(np.array(sim_max) - df['max_intra_similarity'].values[idx]))),
    }
    print(f"  diferencia máxima con el recálculo ({N_VERIFICACION} noticias): {dif}", flush=True)
    assert all(v < 1e-3 for v in dif.values()), f"Las variables reutilizadas no coinciden con el recálculo: {dif}"
    return {'n_noticias': N_VERIFICACION, 'diferencia_absoluta_maxima': dif}


def emociones_por_oracion(oraciones_planas):
    """Probabilidades de RoBERTuito para cada oración (orden: others, joy, sadness, anger, surprise, disgust, fear)."""
    ruta = CACHE / "emociones_oraciones_robertuito.npy"
    if ruta.exists():
        p = np.load(ruta)
        if len(p) == len(oraciones_planas):
            return p
    import torch
    import importlib.util
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.set_num_threads(os.cpu_count() or 4)
    spec = importlib.util.find_spec('pysentimiento')
    s = importlib.util.spec_from_file_location('pysentimiento_preprocessing', Path(spec.submodule_search_locations[0]) / 'preprocessing.py')
    pre = importlib.util.module_from_spec(s)
    s.loader.exec_module(pre)
    me = cfg.MODELOS_HF['Emociones_RoBERTuito']
    tok = AutoTokenizer.from_pretrained(me['id'], revision=me['revision'])
    mod = AutoModelForSequenceClassification.from_pretrained(me['id'], revision=me['revision']).eval()
    id2label = {int(k): v.lower() for k, v in mod.config.id2label.items()}
    orden = [next(k for k, v in id2label.items() if v == e) for e in ['others', 'joy', 'sadness', 'anger', 'surprise', 'disgust', 'fear']]
    print(f"  emociones: {len(oraciones_planas)} oraciones", flush=True)
    p = inferir(tok, mod, [pre.preprocess_tweet(o, lang='es') for o in oraciones_planas], me['max_len'])[:, orden]
    np.save(ruta, p)
    return p


def main():
    corpus = pd.read_csv(cfg.CORPUS_AMPLIADO_CSV)
    base = pd.read_csv(cfg.FEATURES_AVANZADO_4418_CSV)
    assert len(base) == len(corpus) and (base['Text'].astype(str).values == corpus['Text'].astype(str).values).all(), "Filas desalineadas"

    tabla = pd.DataFrame({
        'id_texto': corpus['Text'].map(id_texto),
        'Text': corpus['Text'],
        'label_num': corpus['clase_num'].map(cfg.etiqueta_binaria).astype(int),
        'region': corpus['region'], 'Fuente': corpus['Fuente'], 'subfuente_medio': corpus['subfuente_medio'],
    })
    oraciones = [segmentar(t) for t in corpus['Text']]
    assert (np.array([len(o) for o in oraciones]) == base['num_sentences'].values).all(), "La segmentación no coincide con la de la extracción previa"

    print("[1/3] Verificando las variables reutilizadas (S y R) con un recálculo independiente...", flush=True)
    verificacion = verificar_reutilizadas(base, oraciones)
    grupos = {}
    for g, mapa in VARIABLES_REUTILIZADAS.items():
        for nuevo, viejo in mapa.items():
            tabla[nuevo] = base[viejo].values
        grupos[g] = list(mapa)

    print("[2/3] Emociones por oración con RoBERTuito...", flush=True)
    planas = [o for doc in oraciones for o in doc]
    p = emociones_por_oracion(planas)
    cortes = np.cumsum([0] + [len(o) for o in oraciones])
    col = {'others': 0, 'joy': 1, 'sadness': 2, 'anger': 3, 'surprise': 4, 'disgust': 5, 'fear': 6}
    filas = []
    for i in range(len(oraciones)):
        q = p[cortes[i]:cortes[i + 1]]
        intensidad = 1 - q[:, col['others']]
        f = {'emo_intensidad_media': intensidad.mean(), 'emo_intensidad_max': intensidad.max()}
        for en, es in EMOCIONES_USADAS.items():
            f[f'emo_{es}_media'] = q[:, col[en]].mean()
            f[f'emo_{es}_max'] = q[:, col[en]].max()
        filas.append(f)
    emo = pd.DataFrame(filas)
    for c in emo.columns:
        tabla[c] = emo[c].values
    grupos['E'] = list(emo.columns)

    print("[3/3] Guardando...", flush=True)
    salida = cfg.TABLA_VARIABLES_CSV
    salida.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(salida, index=False)
    meta = {
        'n_noticias': int(len(tabla)), 'n_textos_unicos': int(tabla['id_texto'].nunique()),
        'n_oraciones': int(len(planas)), 'etiqueta': '1 = falsa, 0 = real',
        'grupos': grupos,
        'nombres_grupos': {'S': 'Sensacionalismo', 'R': 'Redundancia', 'E': 'Emociones', 'O': 'Marcadores ortotipográficos', 'L': 'Rasgos lingüísticos'},
        'modelos': {k: {'id': cfg.MODELOS_HF[k]['id'], 'revision': cfg.MODELOS_HF[k]['revision']}
                    for k in ['Sensacionalismo_JJNeila', 'SBERT_Redundancia', 'Emociones_RoBERTuito']},
        'verificacion_variables_reutilizadas': verificacion,
        'faltantes': {c: int(tabla[c].isna().sum()) for g in grupos.values() for c in g if tabla[c].isna().any()},
        'nota': 'Sin imputación: los NaN (noticias con una sola oración) se tratan dentro de cada pliegue.',
    }
    cfg.TABLA_VARIABLES_META_JSON.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f"[OK] {salida} ({len(tabla)} filas, {sum(len(g) for g in grupos.values())} variables)")


if __name__ == '__main__':
    main()
