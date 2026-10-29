"""
Corrección P0-2: solapamiento entre MEX-A3T y FakeDeS (y entre MEX-A3T y Unified).

- Corpus: Noticias_entre_70_y_370_palabras_AMPLIADO_LATAM.csv (4.418 filas, todas las filas de cada fuente).
- Normalización solo para comparar: *NUMBER*, *URL* y *PHONE* se reemplazan por un token neutro ("mascara")
  y se eliminan todas las cifras en ambos lados (así un texto enmascarado y su original sin máscara coinciden).
- TF-IDF (unigramas de palabras, minúsculas, sublinear_tf=False; parámetros por defecto de scikit-learn)
  ajustado sobre la unión de los dos conjuntos comparados; similitud coseno de cada texto de MEX-A3T con cada
  texto de la otra fuente.
Salida: Analisis_complementarios/solapamiento_mexa3t.json y solapamiento_mexa3t_pares.csv (pares con similitud >= 0,8).
"""
import sys
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "config_proyecto.py").exists())
sys.path.insert(0, str(RAIZ))
import config_proyecto as cfg  # noqa: E402

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

SALIDA = Path(__file__).resolve().parent
MASCARA = re.compile(r'\*(NUMBER|URL|PHONE)\*')


def normalizar(t):
    t = MASCARA.sub(' mascara ', str(t))
    return re.sub(r'\d+', ' ', t)


def comparar(c, origen_a, origen_b):
    a = c[c['dataset_origen'] == origen_a]
    b = c[c['dataset_origen'] == origen_b]
    vec = TfidfVectorizer().fit(pd.concat([a['Text'], b['Text']]).map(normalizar))
    A = vec.transform(a['Text'].map(normalizar))
    B = vec.transform(b['Text'].map(normalizar))
    S = (A @ B.T).toarray()
    filas = []
    for i, j in zip(*np.where(S >= 0.8)):
        filas.append({'idx_mexa3t': int(a.index[i]), 'idx_otro': int(b.index[j]), 'fuente_otro': origen_b,
                      'similitud': float(S[i, j]), 'clase_mexa3t': int(a['clase_num'].iloc[i]),
                      'clase_otro': int(b['clase_num'].iloc[j]),
                      'texto_mexa3t_inicio': a['Text'].iloc[i][:120], 'texto_otro_inicio': b['Text'].iloc[j][:120]})
    pares = pd.DataFrame(filas)
    res = {'n_mexa3t': int(len(a)), 'n_otro': int(len(b)), 'similitud_maxima': float(S.max()),
           'textos_mexa3t_con_max_sobre_0_5': int((S.max(1) >= 0.5).sum())}
    for u in (0.9, 0.8):
        p = pares[pares['similitud'] >= u] if len(pares) else pares
        res[f'pares_sim_ge_{u}'] = int(len(p))
        res[f'pares_sim_ge_{u}_misma_etiqueta'] = int((p['clase_mexa3t'] == p['clase_otro']).sum()) if len(p) else 0
        res[f'pares_sim_ge_{u}_etiqueta_opuesta'] = int((p['clase_mexa3t'] != p['clase_otro']).sum()) if len(p) else 0
    return res, pares


def main():
    c = pd.read_csv(cfg.CORPUS_AMPLIADO_CSV)
    mex = 'MEX-A3T (UNAM / IPN)'
    out, todos = {'normalizacion': '*NUMBER*/*URL*/*PHONE* -> "mascara"; cifras eliminadas; TfidfVectorizer() por defecto ajustado sobre la unión de cada par de fuentes'}, []
    for otro, clave in [('FakeDeS 2021 (LatAm)', 'MEX-A3T vs FakeDeS'), ('Freiren Unified Corpus (España)', 'MEX-A3T vs Unified')]:
        r, p = comparar(c, mex, otro)
        out[clave] = r
        todos.append(p)
        print(clave, r)
    (SALIDA / 'solapamiento_mexa3t.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    pd.concat(todos).to_csv(SALIDA / 'solapamiento_mexa3t_pares.csv', index=False)


if __name__ == '__main__':
    main()
