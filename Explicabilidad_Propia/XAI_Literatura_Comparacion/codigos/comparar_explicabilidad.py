"""
Fase 6: comparación de la explicabilidad propia (MexGen por oraciones) con seis métodos de la literatura sobre cuatro
modelos: JJNeila (sensacionalismo), SaBERT (veracidad), hiiamsid (redundancia) y el Modelo sobre Variables.

Uso: python comparar_explicabilidad.py [jjneila sabert redundancia modelo_variables]
Cada documento se guarda en cache/<objetivo>/<id>.json, de modo que la ejecución puede reanudarse.
Al final, consolidar() reúne todo en reportes/datos/metricas_comparacion_xai.json.
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config_proyecto.py").exists())))
import config_proyecto as cfg
sys.path.insert(0, str(Path(__file__).resolve().parent))
import xai_metodos as xm

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

DIR = Path(__file__).resolve().parent
CACHE = DIR / 'cache'
MUESTRA_CSV = cfg.XAI_COMPARACION_DATOS_DIR / 'muestra_comparacion_xai.csv'
N_POR_ESTRATO = 25          # 4 estratos (región × clase) = 100 noticias
N_MODELO_VARIABLES = 100    # misma muestra (con caché por texto, cada noticia tarda ~20 s)
N_ADEBAYO = 15
MAX_PALABRAS = 200          # para que la noticia completa quepa en 256 tokens y todos los métodos vean el mismo texto
METODOS_PALABRA = ['IG', 'IxG', 'Rollout', 'LRP-Chefer', 'LIME-palabra', 'SHAP-palabra']
METODOS_ORACION = ['MexGen', 'LIME-oración', 'SHAP-oración']


def construir_muestra():
    if MUESTRA_CSV.exists():
        return pd.read_csv(MUESTRA_CSV)
    import nltk
    t = pd.read_csv(cfg.TABLA_VARIABLES_CSV).drop_duplicates('id_texto')
    t['n_palabras'] = t['Text'].str.split().str.len()
    t['n_oraciones_punkt'] = t['Text'].map(lambda x: len(nltk.sent_tokenize(x, language='spanish')))
    elegibles = t[(t['n_palabras'] <= MAX_PALABRAS) & (t['n_oraciones_punkt'] >= 3)]
    m = (elegibles.groupby(['region', 'label_num'], group_keys=False)
         .sample(n=N_POR_ESTRATO, random_state=cfg.SEED)[['id_texto', 'Text', 'label_num', 'region', 'Fuente', 'n_palabras', 'n_oraciones_punkt']])
    m = m.sample(frac=1, random_state=cfg.SEED).reset_index(drop=True)   # mezcla: los primeros N quedan estratificados
    MUESTRA_CSV.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(MUESTRA_CSV, index=False)
    return m


def crear_objetivo(clave):
    if clave in ('jjneila', 'sabert'):
        return xm.Clasificador(clave)
    if clave == 'redundancia':
        return xm.Redundancia()
    return xm.ModeloVariables()


def atribuciones(obj, texto, sp_w, sp_s, sp_c):
    """Devuelve {método: (vector, granularidad, segundos)}."""
    A = {}
    if obj.tiene_gradientes:
        t0 = time.perf_counter()
        g, off = obj.ig_ixg(texto)
        dt = (time.perf_counter() - t0) / 2
        for k, v in g.items():
            A[k] = (obj.a_palabras(v, off, texto), 'palabra', dt)
    if obj.tiene_atencion:
        t0 = time.perf_counter()
        g, off = obj.rollout_chefer(texto)
        dt = (time.perf_counter() - t0) / 2
        for k, v in g.items():
            A[k] = (obj.a_palabras(v, off, texto), 'palabra', dt)
    if obj.clave != 'modelo_variables':
        t0 = time.perf_counter()
        A['LIME-palabra'] = (xm.lime_unidades(obj, texto, sp_w, xm.N_LIME_PALABRA), 'palabra', time.perf_counter() - t0)
        t0 = time.perf_counter()
        A['SHAP-palabra'] = (xm.shap_unidades(obj, texto, sp_w, xm.N_SHAP_PALABRA), 'palabra', time.perf_counter() - t0)
    n_lime = xm.N_LIME_ORACION if obj.clave != 'modelo_variables' else 40
    t0 = time.perf_counter()
    A['MexGen'] = (xm.mexgen(obj, texto, sp_s), 'oracion', time.perf_counter() - t0)
    t0 = time.perf_counter()
    A['LIME-oración'] = (xm.lime_unidades(obj, texto, sp_s, n_lime), 'oracion', time.perf_counter() - t0)
    t0 = time.perf_counter()
    A['SHAP-oración'] = (xm.shap_unidades(obj, texto, sp_s, max(2 * len(sp_s) + 1, min(2 ** len(sp_s), 64))), 'oracion', time.perf_counter() - t0)
    t0 = time.perf_counter()
    A['MexGen-cláusulas'] = (xm.mexgen(obj, texto, sp_c), 'clausula', time.perf_counter() - t0)
    return A


def procesar_documento(obj, fila, nlp):
    texto = ' '.join(str(fila['Text']).split())
    target = obj.fijar_objetivo(texto)
    base = float(obj.f([texto])[0])
    sp_w, sp_s, sp_c = xm.spans_palabras(texto), xm.spans_oraciones(texto), xm.spans_clausulas(texto)
    A = atribuciones(obj, texto, sp_w, sp_s, sp_c)
    rng = np.random.default_rng(cfg.SEED)
    spans_de = {'palabra': sp_w, 'oracion': sp_s, 'clausula': sp_c}
    out = {'id_texto': fila['id_texto'], 'label_num': int(fila['label_num']), 'region': fila['region'], 'Fuente': fila['Fuente'],
           'clase_explicada': target, 'f_texto': base, 'n_palabras': len(sp_w), 'n_oraciones': len(sp_s), 'n_clausulas': len(sp_c),
           'oraciones_sin_verbo': xm.fragmentos_sin_verbo(texto, sp_s, nlp), 'clausulas_sin_verbo': xm.fragmentos_sin_verbo(texto, sp_c, nlp),
           'metodos': {}, 'aleatorio': {}}
    sent_vec = {}
    for m, (v, gran, seg) in A.items():
        r = xm.evaluar(obj, texto, spans_de[gran], v, base, gran, rng)
        r.update({'granularidad': gran, 'segundos': seg})
        out['metodos'][m] = r
        if gran == 'palabra':
            vs = xm.agregar_a_unidades(v, sp_w, sp_s)
            sent_vec[m] = vs
            ra = xm.evaluar(obj, texto, sp_s, vs, base, 'oracion', rng)
            ra.update({'granularidad': 'oracion (agregado)', 'segundos': seg})
            out['metodos'][f'{m}→oración'] = ra
        elif gran == 'oracion':
            sent_vec[m] = v
    for gran, sp in spans_de.items():
        out['aleatorio'][gran] = xm.evaluar_aleatorio(obj, texto, sp, base, gran, rng)
    out['acuerdo_con_mexgen'] = {m: xm.spearman(v, sent_vec['MexGen']) for m, v in sent_vec.items() if m != 'MexGen'}
    out['oraciones'] = [texto[a:b] for a, b in sp_s]
    out['mexgen_por_oracion'] = [float(x) for x in A['MexGen'][0]]
    out['_atribuciones'] = {m: [float(x) for x in v] for m, (v, _, _) in A.items()}
    return out


def adebayo(obj, fila, original):
    """Spearman entre las atribuciones con el modelo original y con el modelo con las 3 capas superiores reiniciadas."""
    texto = ' '.join(str(fila['Text']).split())
    sp_w, sp_s, sp_c = xm.spans_palabras(texto), xm.spans_oraciones(texto), xm.spans_clausulas(texto)
    m_rand = obj.copia_aleatorizada()
    if obj.clave == 'redundancia':
        orig_bert = obj.bert
        obj.bert, obj.sbert[0].model = m_rand, m_rand
    else:
        orig_model = obj.model
        obj.model = m_rand
    obj.target = original['clase_explicada']
    try:
        A = atribuciones(obj, texto, sp_w, sp_s, sp_c)
    finally:
        if obj.clave == 'redundancia':
            obj.bert, obj.sbert[0].model = orig_bert, orig_bert
        else:
            obj.model = orig_model
    return {m: xm.spearman(np.array(original['_atribuciones'][m]), v) for m, (v, _, _) in A.items()}


def ejecutar(clave):
    import os
    import torch
    if os.environ.get('XAI_HILOS'):
        torch.set_num_threads(int(os.environ['XAI_HILOS']))   # para correr dos objetivos en paralelo
    import spacy
    nlp = spacy.load('es_core_news_sm', disable=['ner', 'parser'])
    muestra = construir_muestra()
    if clave == 'modelo_variables':
        muestra = muestra.groupby(['region', 'label_num'], group_keys=False).head(N_MODELO_VARIABLES // 4)
    destino = CACHE / clave
    destino.mkdir(parents=True, exist_ok=True)
    obj = crear_objetivo(clave)
    if os.environ.get('XAI_HILOS'):
        torch.set_num_threads(int(os.environ['XAI_HILOS']))   # el extractor de variables fija sus propios hilos al cargarse
    print(f"\n=== {clave}: {len(muestra)} noticias (dispositivo: {xm.dispositivo()}) ===", flush=True)
    t0 = time.time()
    for i, fila in muestra.reset_index(drop=True).iterrows():
        ruta = destino / f"{fila['id_texto']}.json"
        if not ruta.exists():
            r = procesar_documento(obj, fila, nlp)
            ruta.write_text(json.dumps(r, ensure_ascii=False), encoding='utf-8')
        r = json.loads(ruta.read_text(encoding='utf-8'))
        if clave != 'modelo_variables' and i < N_ADEBAYO:
            ra = destino / f"adebayo_{fila['id_texto']}.json"
            if not ra.exists():
                ra.write_text(json.dumps(adebayo(obj, fila, r), ensure_ascii=False), encoding='utf-8')
        print(f"  [{i + 1}/{len(muestra)}] {fila['id_texto']} ({(time.time() - t0) / 60:.1f} min)", flush=True)


def complementar(clave):
    """Segunda evaluación de los métodos por palabra borrando palabras SIN quitar la puntuación de cierre de oración,
    y proporción de palabras con punto final entre las más importantes. Usa las atribuciones guardadas en caché."""
    destino = CACHE / clave
    rutas = [p for p in sorted(destino.glob('*.json')) if not p.name.startswith('adebayo_')]
    pend = [p for p in rutas if 'conservando_cierre' not in json.loads(p.read_text(encoding='utf-8'))]
    if not pend:
        return
    muestra = construir_muestra().set_index('id_texto')
    obj = crear_objetivo(clave)
    print(f"\n=== complemento {clave}: {len(pend)} noticias ===", flush=True)
    for i, ruta in enumerate(pend):
        r = json.loads(ruta.read_text(encoding='utf-8'))
        texto = ' '.join(str(muestra.loc[r['id_texto'], 'Text']).split())
        obj.target = r['clase_explicada'] if r['clase_explicada'] is not None else getattr(obj, 'target', None)
        base = r['f_texto']
        sp_w = xm.spans_palabras(texto)
        rng = np.random.default_rng(cfg.SEED)
        out, cierre = {}, {}
        for m, v in r['_atribuciones'].items():
            if r['metodos'][m]['granularidad'] != 'palabra':
                continue
            v = np.array(v)
            out[m] = xm.evaluar_palabra_conservando_cierre(obj, texto, sp_w, v, base)
            cierre[m] = xm.proporcion_cierre_en_top(texto, sp_w, v)
        al = [xm.evaluar_palabra_conservando_cierre(obj, texto, sp_w, rng.random(len(sp_w)), base) for _ in range(xm.N_ALEATORIOS)]
        out['_aleatorio'] = {'curva': np.mean([a['curva'] for a in al], 0).tolist(), 'aopc': float(np.mean([a['aopc'] for a in al]))}
        r['conservando_cierre'], r['cierre_top10'] = out, cierre
        ruta.write_text(json.dumps(r, ensure_ascii=False), encoding='utf-8')
        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(pend)}]", flush=True)


def consolidar():
    res = {'metadata': {
        'objetivos': {'jjneila': 'P(sensacionalista), JJNeila', 'sabert': 'P(falsa), SaBERT', 'redundancia': 'similitud media entre oraciones, hiiamsid',
                      'modelo_variables': 'P(falsa), Modelo sobre Variables (regresión logística)'},
        'rejilla_q': xm.REJILLA_Q.tolist(), 'q_suficiencia': xm.Q_SUF, 'pasos_ig': xm.N_PASOS_IG,
        'lime_palabra': xm.N_LIME_PALABRA, 'shap_palabra': xm.N_SHAP_PALABRA, 'lime_oracion': xm.N_LIME_ORACION,
        'n_aleatorios': xm.N_ALEATORIOS, 'max_palabras': MAX_PALABRAS}, 'documentos': {}, 'adebayo': {}}
    for d in sorted(CACHE.iterdir()) if CACHE.exists() else []:
        docs = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(d.glob('*.json')) if not p.name.startswith('adebayo_')]
        for x in docs:
            x.pop('_atribuciones', None)
        res['documentos'][d.name] = docs
        res['adebayo'][d.name] = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(d.glob('adebayo_*.json'))]
    salida = cfg.XAI_COMPARACION_DATOS_DIR / 'metricas_comparacion_xai.json'
    salida.write_text(json.dumps(res, ensure_ascii=False), encoding='utf-8')
    print(f"[OK] {salida}: " + ", ".join(f"{k}={len(v)}" for k, v in res['documentos'].items()))


if __name__ == '__main__':
    claves = [a for a in sys.argv[1:] if not a.startswith('-')] or ['jjneila', 'sabert', 'redundancia', 'modelo_variables']
    if '--solo-consolidar' not in sys.argv:
        for c in claves:
            if '--solo-complementar' not in sys.argv:
                ejecutar(c)
            if c != 'modelo_variables':
                complementar(c)
    consolidar()
