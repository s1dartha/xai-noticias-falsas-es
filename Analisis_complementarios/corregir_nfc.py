"""
Corrección de las tildes descompuestas (Unicode NFD) en 1.252 textos del corpus (casi todos de Omdena).

En NFD «explicó» se guarda como «explico» + acento suelto: los léxicos con tilde (verbos de atribución, conectores,
palabras emocionales), spaCy y los tokenizadores de JJNeila, hiiamsid y SaBERT no reconocen esas palabras
(≈7 veces más [UNK]). La corrección es normalizar a NFC (no quitar tildes) en el momento de medir:
variables_texto.extraer(), auditor_estilo.auditar() y auditoria_censura_vs_domain_shift.SaBERT.p_falsa().

El corpus y la columna Text de la tabla NO se reescriben: id_texto se calcula sobre el texto original y de él
dependen los grupos de StratifiedGroupKFold; así los pliegues son idénticos a los de la tesis y la diferencia
en las métricas se debe solo a la corrección.

Este script:
1. Guarda una copia *_pre_nfc.csv de la tabla de variables y de las predicciones de SaBERT (solo la primera vez;
   las ejecuciones siguientes parten siempre de esa copia).
2. Recalcula las 43 variables y P(falsa) de SaBERT con el texto en NFC solo en las filas afectadas
   (las demás no cambian: la extracción reproduce la tabla con diferencias < 1e-6). Guarda el avance en
   datos_nfc/ y se puede reanudar.
3. Sustituye esas filas en tabla_variables_4418.csv y predicciones_sabert_ampliado_4418.csv y escribe
   corregir_nfc.json con el resumen.

Después hay que volver a ejecutar (en este orden):
  SABERT_Evaluacion/Codigos/auditoria_censura_vs_domain_shift.py --recalcular, generar_reporte_sabert.py,
  Modelo_sobre_Variables/Codigos/evaluar_modelo_sobre_variables_sin_fuga.py, seleccionar_modelo_sobre_variables.py,
  generar_reporte_modelo_sobre_variables_sin_fuga.py, Explicabilidad_Propia/codigos/estilos_discursivos.py, generar_reporte_estilos.py, Analisis_complementarios/sensibilidad_clasificador.py y ejecutar_casos_originales.py.
"""
import sys
import json
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "config_proyecto.py").exists())
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / 'Explicabilidad_Propia' / 'codigos'))
import config_proyecto as cfg  # noqa: E402

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

AQUI = Path(__file__).resolve().parent
DATOS = AQUI / 'datos_nfc'
CACHE_VARS = DATOS / 'variables_nfc.csv'
CACHE_SABERT = DATOS / 'sabert_nfc.csv'
SALIDA_JSON = AQUI / 'corregir_nfc.json'
# Textos de una sola oración: la tabla original trae estas dos variables imputadas y el extractor devuelve NaN
HEREDADAS = ('red_consecutivas_media', 'red_consecutivas_min')


def respaldo(ruta):
    copia = ruta.with_name(ruta.stem + '_pre_nfc.csv')
    if not copia.exists():
        copia.write_bytes(ruta.read_bytes())
    return pd.read_csv(copia)


def variables_nfc(textos, filas):
    hechas = pd.read_csv(CACHE_VARS) if CACHE_VARS.exists() else pd.DataFrame(columns=['fila'])
    pend = [i for i in filas if i not in set(hechas['fila'])]
    if pend:
        import variables_texto as vt
        ex, nuevas, t0 = vt.ExtractorVariables(), [], time.time()
        for n, i in enumerate(pend, 1):
            v, _ = ex.extraer(textos[i])
            v['fila'] = int(i)
            nuevas.append(v)
            if n % 50 == 0 or n == len(pend):
                hechas = pd.concat([hechas, pd.DataFrame(nuevas)], ignore_index=True)
                hechas.to_csv(CACHE_VARS, index=False)
                nuevas = []
                print(f"  variables {len(hechas)}/{len(filas)} ({time.time() - t0:.0f} s)", flush=True)
    return hechas.drop_duplicates('fila').set_index('fila').loc[filas]


def sabert_nfc(textos, filas):
    if CACHE_SABERT.exists():
        c = pd.read_csv(CACHE_SABERT)
        if list(c['fila']) == list(filas):
            return c['p_falsa'].values
    sys.path.insert(0, str(cfg.SABERT_CODIGOS_DIR))
    from auditoria_censura_vs_domain_shift import SaBERT  # normaliza a NFC dentro de p_falsa()
    p = SaBERT().p_falsa([textos[i] for i in filas])
    pd.DataFrame({'fila': filas, 'p_falsa': p}).to_csv(CACHE_SABERT, index=False)
    return p


def main():
    DATOS.mkdir(exist_ok=True)
    tabla, sab = respaldo(cfg.TABLA_VARIABLES_CSV), respaldo(cfg.PREDICCIONES_SABERT_4418_CSV)
    corpus = pd.read_csv(cfg.CORPUS_AMPLIADO_CSV)
    txt = tabla['Text'].astype(str)
    assert (txt.values == sab['Text'].astype(str).values).all() and (txt.values == corpus['Text'].astype(str).values).all()
    filas = [int(i) for i in np.where(txt.map(lambda t: unicodedata.normalize('NFC', t)) != txt)[0]]
    print(f"Filas con NFD: {len(filas)}", flush=True)

    meta = json.loads(cfg.TABLA_VARIABLES_META_JSON.read_text(encoding='utf-8'))
    variables = [v for g in meta['grupos'].values() for v in g]
    nv = variables_nfc(txt.values, filas)
    antes = tabla.loc[filas, variables].copy()
    for k in variables:
        nuevo = nv[k].values.astype(float)
        if k in HEREDADAS:
            nuevo = np.where(np.isnan(nuevo), antes[k].values, nuevo)
        tabla.loc[filas, k] = nuevo

    p = sabert_nfc(txt.values, filas)
    p_antes = sab.loc[filas, 'prob_fake'].values.copy()
    sab.loc[filas, 'prob_fake'] = p
    sab.loc[filas, 'prob_real'] = 1 - p
    sab.loc[filas, 'pred_sabert'] = (p >= 0.5).astype(int)

    tabla.to_csv(cfg.TABLA_VARIABLES_CSV, index=False)
    sab.to_csv(cfg.PREDICCIONES_SABERT_4418_CSV, index=False)

    despues = tabla.loc[filas, variables]
    cambian = ((despues - antes).abs() > 1e-9) & ~(despues.isna() & antes.isna())
    res = {
        'filas_corregidas': len(filas),
        'por_corpus_de_origen': corpus.loc[filas, 'dataset_origen'].value_counts().to_dict(),
        'textos_unicos_corregidos': int(tabla.loc[filas, 'id_texto'].nunique()),
        'variables_que_cambian_pct_filas': {k: round(float(cambian[k].mean() * 100), 1) for k in variables if cambian[k].any()},
        'medias_antes_despues': {k: [float(antes[k].mean()), float(despues[k].mean())] for k in variables if cambian[k].any()},
        'sabert': {'p_falsa_media_antes': float(p_antes.mean()), 'p_falsa_media_despues': float(p.mean()),
                   'predicciones_que_cambian': int(((p_antes >= 0.5) != (p >= 0.5)).sum())},
        'respaldos': [cfg.TABLA_VARIABLES_CSV.stem + '_pre_nfc.csv', cfg.PREDICCIONES_SABERT_4418_CSV.stem + '_pre_nfc.csv'],
        'nota': 'id_texto y la columna Text no cambian: los pliegues de validación cruzada son los mismos de la tesis.',
    }
    SALIDA_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f"[OK] {len(filas)} filas corregidas | SaBERT: {res['sabert']['predicciones_que_cambian']} predicciones cambian | {SALIDA_JSON.name}")


if __name__ == '__main__':
    main()
