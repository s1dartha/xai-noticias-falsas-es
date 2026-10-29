"""
Corrección P0-2 / P0-3 / P1-16: análisis de sensibilidad del clasificador interpretable.

Reutiliza las funciones de Modelo_sobre_Variables/Codigos/evaluar_modelo_sobre_variables_sin_fuga.py (cargar_datos,
ajustar_y_predecir, metricas, desglose, nombre_fuente, umbral anidado, bootstrap) sin modificarlas. Lo único que
cambia entre escenarios es (1) la variable de grupo de StratifiedGroupKFold (externa e interna), (2) los NaN de
red_consecutivas_* y (3) la región/fuente asignada a los 23 textos compartidos entre Unified y FakeDeS.

Escenarios
  a  Reproducción del protocolo original (grupo = id_texto).
  b  Grupo = clúster de casi duplicados (componentes conexas de pares con coseno TF-IDF >= 0,90).
  c  red_consecutivas_media y red_consecutivas_min = NaN en los documentos de una sola oración (ling_oraciones == 1);
     la imputación por mediana la hace el pipeline dentro del pliegue.
  d  Los 23 textos compartidos Unified/FakeDeS se asignan a FakeDeS / América Latina.
  e  b + c + d.
Salidas: sensibilidad_clasificador.json, predicciones_oof_original.csv, clusteres_casi_duplicados.csv
"""
import sys
import json
import time
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold

RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "config_proyecto.py").exists())
sys.path.insert(0, str(RAIZ))
import config_proyecto as cfg  # noqa: E402

_spec = importlib.util.spec_from_file_location('evaluar_sin_fuga', cfg.MODELO_VARIABLES_CODIGOS_DIR / 'evaluar_modelo_sobre_variables_sin_fuga.py')
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

SALIDA = Path(__file__).resolve().parent
UMBRAL_CASI_DUP = 0.90
URL_FAKEDES = 'https://huggingface.co/datasets/mariagrandury/fake_news_corpus_spanish'
URL_UNIFIED = 'https://huggingface.co/datasets/Freiren/Unified-and-Balanced-Spanish-Fake-News-Corpus'
VARS_CONSEC = ['red_consecutivas_media', 'red_consecutivas_min']
# Tras corregir_nfc.py (antes: Global 0.612, AUC 0.670, España 0.573, América Latina 0.647)
OBJETIVO_A = {'Global': 0.625, 'AUC': 0.676, 'España': 0.584, 'América Latina': 0.663}


# ---------------------------------------------------------------------------
def clusteres_casi_duplicados(textos, umbral=UMBRAL_CASI_DUP, bloque=500):
    """Componentes conexas del grafo de pares con similitud coseno TF-IDF >= umbral (TF-IDF sin etiquetas)."""
    X = TfidfVectorizer().fit_transform(textos)  # filas normalizadas L2: X·Xᵀ = coseno
    filas, cols, sims = [], [], []
    for i0 in range(0, X.shape[0], bloque):
        S = (X[i0:i0 + bloque] @ X.T).toarray()
        ii, jj = np.where(S >= umbral)
        m = (ii + i0) < jj
        filas += (ii[m] + i0).tolist(); cols += jj[m].tolist(); sims += S[ii[m], jj[m]].tolist()
    n = X.shape[0]
    A = csr_matrix((np.ones(len(filas)), (filas, cols)), shape=(n, n))
    _, etiquetas = connected_components(A, directed=False)
    return etiquetas, pd.DataFrame({'i': filas, 'j': cols, 'similitud': sims})


def vc(df, X, tipo, grupos):
    """Igual que E.validacion_cruzada, pero con la variable de grupo como parámetro (externa e interna)."""
    y = df['label_num'].values
    externo = list(StratifiedGroupKFold(n_splits=E.N_PLIEGUES, shuffle=True, random_state=cfg.SEED).split(df, y, grupos))
    p, pred, thetas, pliegue = np.zeros(len(y)), np.zeros(len(y), int), [], np.zeros(len(y), int)
    for k, (tr, te) in enumerate(externo):
        p[te], th = E.ajustar_y_predecir(X[tr], y[tr], grupos[tr], X[te], tipo)
        pred[te] = (p[te] >= th).astype(int)
        thetas.append(th)
        pliegue[te] = k
    return p, pred, thetas, pliegue


def particion(df, X, tipo, grupos, m_tr, m_te):
    y = df['label_num'].values
    p, th = E.ajustar_y_predecir(X[m_tr], y[m_tr], grupos[m_tr], X[m_te], tipo)
    return {'umbral': th, **E.metricas(y[m_te], p, (p >= th).astype(int))}


def escenario(df, grupos, X_por_config, configs):
    out = {}
    y = df['label_num'].values
    for nombre, tipo in configs.items():
        X = X_por_config[nombre]
        p, pred, thetas, pliegue = vc(df, X, tipo, grupos)
        r = {'umbrales_por_pliegue': thetas, **E.desglose(df, y, p, pred)}
        r['transferencia'] = {f'{o} → {d}': particion(df, X, tipo, grupos, (df['region'] == o).values, (df['region'] == d).values)
                              for o, d in [('España', 'América Latina'), ('América Latina', 'España')]}
        r['exclusion_fuente'] = {E.nombre_fuente(f): particion(df, X, tipo, grupos, (df['Fuente'] != f).values, (df['Fuente'] == f).values)
                                 for f in sorted(df['Fuente'].unique())}
        r['_oof'] = (p, pred, thetas, pliegue)
        out[nombre] = r
        g = r['Global']
        print(f"    {nombre:<22} F1m={g['f1_macro']:.4f} IC=[{g['f1_macro_ic95'][0]:.3f}, {g['f1_macro_ic95'][1]:.3f}] AUC={g['roc_auc']:.4f} "
              f"FP={g['FP']} FN={g['FN']} | ES={r['España']['f1_macro']:.4f} AL={r['América Latina']['f1_macro']:.4f} | "
              f"ES→AL={r['transferencia']['España → América Latina']['f1_macro']:.4f} AL→ES={r['transferencia']['América Latina → España']['f1_macro']:.4f} "
              f"sinUnified={r['exclusion_fuente']['Unified-and-Balanced-Spanish-Fake-News-Corpus']['f1_macro']:.4f} θ={thetas}", flush=True)
    return out


def main():
    t0 = time.time()
    df, n_noticias, meta = E.cargar_datos()
    tabla = pd.read_csv(cfg.TABLA_VARIABLES_CSV)
    idx_csv = tabla.drop_duplicates('id_texto', keep='first').index.values
    assert (tabla['Text'].values[idx_csv] == df['Text'].values).all()
    df['idx_csv'] = idx_csv
    G = meta['grupos']
    todas = G['S'] + G['R'] + G['E'] + G['O'] + G['L']
    assert len(todas) == 43
    CONFIGS = {'Completo (S+R+E+O+L)': 'lr', 'SaBERT': 'sabert'}
    VARS = {'Completo (S+R+E+O+L)': todas, 'SaBERT': ['sabert_p_falsa']}
    y = df['label_num'].values
    print(f"Textos únicos: {len(df)} (de {n_noticias})", flush=True)

    # --- (b) clústeres de casi duplicados
    clus, pares = clusteres_casi_duplicados(df['Text'].astype(str).tolist())
    tam = pd.Series(clus).value_counts()
    pares['etiqueta_opuesta'] = y[pares['i']] != y[pares['j']]
    pares['fuente_i'] = df['Fuente'].values[pares['i']].tolist()
    pares['fuente_j'] = df['Fuente'].values[pares['j']].tolist()
    pares['idx_csv_i'] = idx_csv[pares['i']]; pares['idx_csv_j'] = idx_csv[pares['j']]
    mixtos = pd.DataFrame({'c': clus, 'y': y}).groupby('c')['y'].nunique()
    solo_unif = (pares['fuente_i'] == URL_UNIFIED) & (pares['fuente_j'] == URL_UNIFIED)
    info_b = {
        'metodo': f'TfidfVectorizer() por defecto sobre los {len(df)} textos únicos; pares con coseno >= {UMBRAL_CASI_DUP}; componentes conexas',
        'pares_sim_ge_0_90': int(len(pares)), 'pares_etiqueta_opuesta': int(pares['etiqueta_opuesta'].sum()),
        'pares_dentro_de_Unified': int(solo_unif.sum()), 'pares_dentro_de_Unified_etiqueta_opuesta': int((solo_unif & pares['etiqueta_opuesta']).sum()),
        'pares_entre_fuentes_distintas': int((pares['fuente_i'] != pares['fuente_j']).sum()),
        'n_clusteres': int(len(tam)), 'n_clusteres_tamano_mayor_1': int((tam > 1).sum()),
        'textos_en_clusteres_tamano_mayor_1': int(tam[tam > 1].sum()), 'tamano_maximo': int(tam.max()),
        'distribucion_tamanos_mayor_1': {int(k): int(v) for k, v in tam[tam > 1].value_counts().sort_index().items()},
        'clusteres_con_etiquetas_mezcladas': int((mixtos > 1).sum()),
    }
    print('  (b)', info_b, flush=True)
    pd.DataFrame({'idx_csv': idx_csv, 'id_texto': df['id_texto'], 'cluster': clus, 'tamano_cluster': tam.reindex(clus).values,
                  'label_num': y, 'Fuente': df['Fuente']}).to_csv(SALIDA / 'clusteres_casi_duplicados.csv', index=False)
    pares.to_csv(SALIDA / 'pares_casi_duplicados_090.csv', index=False)

    # --- (c) NaN en red_consecutivas_* para documentos de una oración
    una = (df['ling_oraciones'] == 1).values
    valores_imputados = df.loc[una, VARS_CONSEC].drop_duplicates().values.tolist()
    info_c = {'docs_una_oracion_textos_unicos': int(una.sum()), 'docs_una_oracion_en_4418': int((tabla['ling_oraciones'] == 1).sum()),
              'valores_imputados_previos_[media,min]': valores_imputados,
              'docs_con_red_consecutivas_media_igual_0_379389_textos_unicos': int((df['red_consecutivas_media'].round(6) == 0.379389).sum()),
              'nota': 'Se identifican por ling_oraciones == 1; un documento de varias oraciones tiene por azar el mismo valor que la mediana, por eso no se usa el valor como criterio.'}
    print('  (c)', info_c, flush=True)
    df_c = df.copy()
    df_c.loc[una, VARS_CONSEC] = np.nan

    # --- (d) 23 textos compartidos Unified/FakeDeS
    ids_fakedes = set(tabla.loc[tabla['Fuente'] == URL_FAKEDES, 'id_texto'])
    compartidos = df['id_texto'].isin(ids_fakedes).values & (df['Fuente'] == URL_UNIFIED).values
    info_d = {'textos_compartidos_reasignados': int(compartidos.sum()),
              'falsas': int(y[compartidos].sum()), 'reales': int((y[compartidos] == 0).sum()),
              'idx_csv_conservado (Unified)': idx_csv[compartidos].tolist()}
    print('  (d)', {k: v for k, v in info_d.items() if not k.startswith('idx')}, flush=True)

    def con_d(d):
        d = d.copy()
        d.loc[compartidos, 'Fuente'] = URL_FAKEDES
        d.loc[compartidos, 'region'] = 'América Latina'
        return d

    g_id = df['id_texto'].values
    ESC = {
        'a_reproduccion': (df, g_id),
        'b_clusteres_casi_duplicados': (df, clus),
        'c_imputacion_en_pliegue': (df_c, g_id),
        'd_23_compartidos_a_FakeDeS': (con_d(df), g_id),
        'e_b_c_d_combinados': (con_d(df_c), clus),
    }
    res = {'metadata': {
        'script_reutilizado': 'Modelos_Individuales/Clasificar_fake/Modelo_sobre_Variables/Codigos/evaluar_modelo_sobre_variables_sin_fuga.py',
        'protocolo': 'textos únicos (drop_duplicates id_texto, keep=first); StratifiedGroupKFold(5, shuffle, seed 42); '
                     'pipeline mediana + StandardScaler + LogisticRegression(C=1, L2); umbral anidado con StratifiedGroupKFold(5, seed 43) '
                     'interna sobre rejilla 0,05–0,95 maximizando F1 macro; IC 95 % bootstrap 1.000 réplicas, semilla 42',
        'n_textos_unicos': int(len(df)), 'n_variables_completo': len(todas),
        'b_casi_duplicados': info_b, 'c_imputacion': info_c, 'd_compartidos': info_d,
        'nota_grupos': 'En b y e los clústeres se usan como grupos tanto en la validación externa como en la interna del umbral y en la transferencia/exclusión.',
    }, 'escenarios': {}}

    oof_a = None
    for nombre, (d, grupos) in ESC.items():
        print(f"\n[{nombre}]", flush=True)
        X_por = {k: d[v].values.astype(float) for k, v in VARS.items()}
        r = escenario(d, grupos, X_por, CONFIGS)
        if nombre == 'a_reproduccion':
            oof_a = {k: v['_oof'] for k, v in r.items()}
            g, es, al = (r['Completo (S+R+E+O+L)'][z] for z in ['Global', 'España', 'América Latina'])
            obtenido = {'Global': round(g['f1_macro'], 3), 'AUC': round(g['roc_auc'], 3), 'España': round(es['f1_macro'], 3), 'América Latina': round(al['f1_macro'], 3)}
            print('  reproducción:', obtenido, 'objetivo:', OBJETIVO_A, flush=True)
            if obtenido != OBJETIVO_A:
                raise SystemExit(f'La reproducción NO coincide: {obtenido} vs {OBJETIVO_A}')
            # Comparación con las predicciones guardadas por el script original
            orig = pd.read_csv(cfg.PREDICCIONES_MODELO_VARIABLES_SIN_FUGA_CSV)
            res['metadata']['a_max_dif_p_vs_predicciones_originales'] = {
                k: float(np.abs(orig[f'p__{k}'].values - v['_oof'][0]).max()) for k, v in r.items()}
            res['metadata']['a_predicciones_identicas'] = {k: bool((orig[f'pred__{k}'].values == v['_oof'][1]).all()) for k, v in r.items()}
        for v in r.values():
            del v['_oof']
        res['escenarios'][nombre] = r

    # Predicciones fuera de pliegue del protocolo original para las 4.418 filas
    p, pred, thetas, pliegue = oof_a['Completo (S+R+E+O+L)']
    ps, preds_s, thetas_s, _ = oof_a['SaBERT']
    rep = pd.DataFrame({'id_texto': df['id_texto'], 'idx_csv_representante': idx_csv, 'pliegue': pliegue,
                        'umbral_pliegue': np.array(thetas)[pliegue], 'p_falsa_oof': p, 'pred_oof': pred,
                        'sabert_p_falsa': ps, 'sabert_umbral_pliegue': np.array(thetas_s)[pliegue], 'sabert_pred_oof': preds_s})
    corpus = pd.read_csv(cfg.CORPUS_AMPLIADO_CSV)
    oof = pd.DataFrame({'idx_csv': np.arange(len(tabla)), 'id_texto': tabla['id_texto'], 'label_num': tabla['label_num'],
                        'region': tabla['region'], 'dataset_origen': corpus['dataset_origen'], 'Fuente': tabla['Fuente']}).merge(rep, on='id_texto', how='left')
    oof['es_representante'] = oof['idx_csv'] == oof['idx_csv_representante']
    assert oof['p_falsa_oof'].notna().all() and len(oof) == len(tabla)
    oof.to_csv(SALIDA / 'predicciones_oof_original.csv', index=False)

    res['metadata']['duracion_s'] = time.time() - t0
    (SALIDA / 'sensibilidad_clasificador.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    print(f"\n[OK] {SALIDA / 'sensibilidad_clasificador.json'} ({res['metadata']['duracion_s'] / 60:.1f} min)")


if __name__ == '__main__':
    main()
