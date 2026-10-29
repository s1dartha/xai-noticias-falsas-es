"""
Corrección 0.1 / P0-1: casos de estudio con los textos ORIGINALES del corpus, sin modificar.

1. Construye Analisis_complementarios/casos_originales.json con el texto exacto de
   Noticias_entre_70_y_370_palabras_AMPLIADO_LATAM.csv (índices 2357, 3071, 3653 y 2), la etiqueta tomada de clase_num,
   en_corpus = true, la fuente y el índice.
2. Ejecuta el auditor vigente (Explicabilidad_Propia/codigos/auditor_estilo.py, sin cambios) sobre cada texto.
3. Añade la P(falsa) fuera de pliegue del protocolo original (Analisis_complementarios/predicciones_oof_original.csv) y el
   umbral del pliegue.
4. Regenera las figuras con las funciones de ejecutar_casos_estilo.py, redirigiendo las rutas a
   Analisis_complementarios/casos_estilo/ y cambiando el título «MexGen por oraciones» por «Expl. por oraciones».
   El script original no se modifica (se lee su código, se sustituye la cadena en memoria y se ejecuta).
"""
import sys
import json
import time
import types
from pathlib import Path

import pandas as pd

RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "config_proyecto.py").exists())
sys.path.insert(0, str(RAIZ))
import config_proyecto as cfg  # noqa: E402

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

AQUI = Path(__file__).resolve().parent
SALIDA = AQUI / 'casos_estilo'
ORIGEN = cfg.EXPLICABILIDAD_CODIGOS_DIR / 'ejecutar_casos_estilo.py'
CASOS = [  # (id conservado del JSON anterior, índice en el CSV de 4.418)
    ('noticia_1_falsa_salud_mayusculas', 2357),
    ('noticia_2_falsa_deportes_chicharito', 3071),  # antes 2397 (coronacirco): el menos representativo de su estilo
    ('noticia_3_verdadera_medicos_zacatecoluca', 3653),
    ('noticia_4_verdadera_ciencia_efe', 2),
]
SUSTITUCIONES = [("MexGen por oraciones en los 4 casos", "Expl. por oraciones en los 4 casos")]


def construir_entrada():
    c = pd.read_csv(cfg.CORPUS_AMPLIADO_CSV)
    casos = []
    for id_, idx in CASOS:
        fila = c.loc[idx]
        y = int(fila['clase_num'])
        casos.append({'id': id_, 'indice_csv': idx, 'clase_num': y,
                      'tipo_referencia': f"{'Falsa' if y == 1 else 'Verdadera'} (clase_num = {y} en el corpus)",
                      'titulo': f"Caso {id_.split('_')[1]} · índice {idx} · {fila['dataset_origen']}",
                      'origen': f"{fila['dataset_origen']} ({fila['Fuente']})", 'fuente': fila['Fuente'],
                      'dataset_origen': fila['dataset_origen'], 'region': fila['region'],
                      'en_corpus': True, 'texto': fila['Text']})
    (AQUI / 'casos_originales.json').write_text(json.dumps(casos, ensure_ascii=False, indent=1), encoding='utf-8')
    return casos


def main():
    casos = construir_entrada()
    src = ORIGEN.read_text(encoding='utf-8')
    for a, b in SUSTITUCIONES:
        assert a in src, a
        src = src.replace(a, b)
    mod = types.ModuleType('casos_renombrado')
    mod.__file__ = str(ORIGEN)
    exec(compile(src, str(ORIGEN), 'exec'), mod.__dict__)
    mod.SALIDA = SALIDA
    mod.FIG = SALIDA / 'casos_dimensiones_iml.png'
    mod.FIG_RADAR = SALIDA / 'casos_radar_estilo.png'
    mod.FIG_MEXGEN = SALIDA / 'casos_mexgen_oraciones.png'
    SALIDA.mkdir(parents=True, exist_ok=True)

    oof = pd.read_csv(AQUI / 'predicciones_oof_original.csv').set_index('idx_csv')
    aud = mod.AuditorEstilo()
    res = []
    for c in casos:
        t0 = time.time()
        r = aud.auditar(c['texto'], c['titulo'])
        r['segundos_auditoria'] = round(time.time() - t0, 1)
        r.update({k: c[k] for k in ['id', 'tipo_referencia', 'origen', 'en_corpus', 'indice_csv', 'clase_num', 'fuente', 'dataset_origen', 'region']})
        o = oof.loc[c['indice_csv']]
        r['modelo_sobre_variables_fuera_de_pliegue'] = {
            'p_falsa_oof': float(o['p_falsa_oof']), 'umbral_pliegue': float(o['umbral_pliegue']), 'pliegue': int(o['pliegue']),
            'prediccion_oof': 'falsa' if int(o['pred_oof']) == 1 else 'real',
            'idx_csv_representante': int(o['idx_csv_representante']), 'es_representante': bool(o['es_representante']),
            'nota': 'Protocolo original (escenario a de sensibilidad_clasificador.py). Si el texto está duplicado, se usa la predicción '
                    'del representante conservado por drop_duplicates(keep=first).'}
        (SALIDA / f"{c['id']}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding='utf-8')
        res.append(r)
        print(f"  {c['id']:<42} IML={r['perfil']['iml']:5.1f} | {r['perfil']['estilo']} ({r['perfil']['pertenencia_estilo']} %) | "
              f"P(falsa) completo={r['modelo_sobre_variables']['p_falsa']} | OOF={o['p_falsa_oof']:.3f} (θ={o['umbral_pliegue']}) | {r['segundos_auditoria']} s", flush=True)
    ref = json.loads(cfg.REFERENCIA_ESTILOS_JSON.read_text(encoding='utf-8'))
    mod.figura(casos, res)
    mod.figura_radar(casos, res, ref)
    mod.figura_mexgen(casos, res)
    print('[OK]', SALIDA)


if __name__ == '__main__':
    main()
