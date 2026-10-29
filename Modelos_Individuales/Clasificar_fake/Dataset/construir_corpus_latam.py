"""
Ampliación latinoamericana del corpus base (Sección 3.2.2 de la tesis).

Descarga MEX-A3T (GitHub) y Omdena (Hugging Face), codifica la etiqueta, aplica el filtro de 70-370 palabras y la
deduplicación exacta, y añade al corpus base las noticias cuyo texto no estaba ya en él.

Nota: la salida directa conserva columnas de trabajo (class, label_num, pais, tema) y los 2 registros de Kaggle del
corpus base. El archivo canónico Noticias_entre_70_y_370_palabras_AMPLIADO_LATAM.csv (4.418 noticias) es la versión
armonizada (region, clase_num), que se conserva como archivo de datos (Apéndice A.3). Este script lo sobrescribe:
ejecútelo sobre una copia si solo quiere verificar la ampliación. Ver Dataset/README.md.
"""
import sys as _sys_cfg
from pathlib import Path as _Path_cfg
_sys_cfg.path.insert(0, str(next(p for p in _Path_cfg(__file__).resolve().parents if (p / "config_proyecto.py").exists())))
import config_proyecto as cfg
import os
import sys
import pandas as pd
import numpy as np
from datasets import load_dataset

dataset_dir = str(cfg.ROOT / "Modelos_Individuales/Clasificar_fake/Dataset")
os.makedirs(dataset_dir, exist_ok=True)

# Columna de Omdena usada como etiqueta. 'Prediction' es la salida de un clasificador del proyecto
# Omdena (con Prediction_prob ~0,5); 'Corrected_label' es la etiqueta corregida del dataset.
# Hasta el 2026-10-24 el corpus se construyó con 'Prediction'; se corrigió a 'Corrected_label'
# (138 noticias cambian de clase). Trazabilidad: Dataset/omdena_etiquetas_prediction_vs_corrected.csv
COLUMNA_ETIQUETA_OMDENA = 'Corrected_label'


def es_noticia_verdadera(valor):
    """Devuelve True (noticia real) o False (falsa); lanza error ante cualquier otro valor."""
    v = str(valor).strip().lower()
    if v == 'true':
        return True
    if v in ('fake', 'false'):
        return False
    raise ValueError(f"Etiqueta de origen no reconocida: {valor!r}")

print("="*80)
print(" CONSTRUCCIÓN DEL CORPUS LATINOAMERICANO DE FAKE NEWS Y FACT-CHECKING")
print("="*80)

# =========================================================================
# 1. FUENTE 1: MEX-A3T (UNAM / IPN / IberLEF - México)
# =========================================================================
print("\n>>> [1/3] Descargando y procesando MEX-A3T (México - UNAM / IPN)...")
url_train = 'https://raw.githubusercontent.com/jpposadas/FakeNewsCorpusSpanish/master/train.xlsx'
url_dev = 'https://raw.githubusercontent.com/jpposadas/FakeNewsCorpusSpanish/master/development.xlsx'

df_train = pd.read_excel(url_train)
df_dev = pd.read_excel(url_dev)
df_mex = pd.concat([df_train, df_dev]).reset_index(drop=True)

df_mex['Text'] = df_mex['Text'].astype(str).str.strip()
df_mex['conteo_palabras_text'] = df_mex['Text'].apply(lambda x: len(x.split()))
df_mex['class'] = df_mex['Category'].apply(es_noticia_verdadera)
df_mex['Fuente'] = 'https://github.com/jpposadas/FakeNewsCorpusSpanish (MEX-A3T - México)'
df_mex['pais'] = 'México'
df_mex['subfuente_medio'] = df_mex['Source'].fillna('Medios México / VerificadoMX')
df_mex['tema'] = df_mex['Topic'].fillna('General')
df_mex['dataset_origen'] = 'MEX-A3T (UNAM / IPN)'

print(f"  [OK] MEX-A3T cargado: {len(df_mex)} registros (Reales: {(df_mex['class']==True).sum()}, Falsas: {(df_mex['class']==False).sum()})")

# =========================================================================
# 2. FUENTE 2: FakeDeS 2021 (IberLEF - Latinoamérica)
# =========================================================================
print("\n>>> [2/3] Descargando y procesando FakeDeS 2021 (Latinoamérica)...")
url_fakedes = 'https://raw.githubusercontent.com/jpposadas/FakeNewsCorpusSpanish/master/test.xlsx'
df_fakedes = pd.read_excel(url_fakedes)

df_fakedes['Text'] = df_fakedes['TEXT'].astype(str).str.strip()
df_fakedes['conteo_palabras_text'] = df_fakedes['Text'].apply(lambda x: len(x.split()))
df_fakedes['class'] = df_fakedes['CATEGORY'].apply(es_noticia_verdadera)
df_fakedes['Fuente'] = 'https://github.com/jpposadas/FakeNewsCorpusSpanish (FakeDeS - LatAm)'
df_fakedes['pais'] = 'LatAm Multinacional (México/Colombia/Argentina)'
df_fakedes['subfuente_medio'] = df_fakedes['SOURCE'].fillna('Fact-checking LatAm')
df_fakedes['tema'] = df_fakedes['TOPICS'].fillna('General')
df_fakedes['dataset_origen'] = 'FakeDeS 2021 (IberLEF)'

print(f"  [OK] FakeDeS 2021 cargado: {len(df_fakedes)} registros (Reales: {(df_fakedes['class']==True).sum()}, Falsas: {(df_fakedes['class']==False).sum()})")

# =========================================================================
# 3. FUENTE 3: Omdena LATAM (Politics Fake News Detector in LATAM)
# =========================================================================
print("\n>>> [3/3] Descargando y procesando Omdena LATAM (Hugging Face)...")
ds_omdena = load_dataset('IsaacRodgz/Fake-news-latam-omdena')
df_omdena = pd.concat([ds_omdena['train'].to_pandas(), ds_omdena['test'].to_pandas()]).reset_index(drop=True)
df_om_web = df_omdena[df_omdena['Type'] == 'Web Article'].copy().reset_index(drop=True)

df_om_web['Text'] = df_om_web['Content'].astype(str).str.strip()
df_om_web['conteo_palabras_text'] = df_om_web['Text'].apply(lambda x: len(x.split()))
df_om_web['class'] = df_om_web[COLUMNA_ETIQUETA_OMDENA].apply(es_noticia_verdadera)
df_om_web['Fuente'] = 'https://huggingface.co/datasets/IsaacRodgz/Fake-news-latam-omdena'

def get_country(src):
    s = str(src).lower()
    if any(k in s for k in ['nuevo siglo', 'eltiempo', 'el tiempo', 'heraldo', 'rcn', 'medellin']):
        return 'Colombia'
    elif any(k in s for k in ['mundo', 'edh', 'salvador', 'times']):
        return 'El Salvador / Centroamérica'
    elif any(k in s for k in ['sinembargo', 'proceso', 'jornada', 'norte', 'milenio', 'universal', 'reforma', 'excelsior', 'expansion', 'wradio', 'economista', 'formula', 'puebla', 'doriga', 'televisa', 'forotv', 'forbes', 'mvs']):
        return 'México'
    else:
        return 'América Latina / Internacional'

df_om_web['pais'] = df_om_web['Source'].apply(get_country)
df_om_web['subfuente_medio'] = df_om_web['Source'].fillna('Web Article LatAm')
df_om_web['tema'] = 'Política / Actualidad'
df_om_web['dataset_origen'] = 'Omdena LATAM (Politics Fake News Detector)'

print(f"  [OK] Omdena Web cargado: {len(df_om_web)} registros (Reales: {(df_om_web['class']==True).sum()}, Falsas: {(df_om_web['class']==False).sum()})")

# =========================================================================
# 4. UNIFICACIÓN Y ESTANDARIZACIÓN CANÓNICA
# =========================================================================
cols = ['class', 'Text', 'Fuente', 'conteo_palabras_text', 'pais', 'subfuente_medio', 'tema', 'dataset_origen']
combined_all = pd.concat([df_mex[cols], df_fakedes[cols], df_om_web[cols]]).reset_index(drop=True)

# Limpieza de duplicados basados en texto idéntico
combined_all['text_clean'] = combined_all['Text'].str.lower().str.strip()
n_before = len(combined_all)
combined_all = combined_all.drop_duplicates(subset=['text_clean']).reset_index(drop=True)
combined_all = combined_all.drop(columns=['text_clean'])
print(f"\n[INFO] Deduplicación: {n_before} -> {len(combined_all)} registros únicos")

# Columnas canónicas requeridas para compatibilidad total
combined_all['label_num'] = combined_all['class'].apply(cfg.etiqueta_binaria)
combined_all['label_name'] = combined_all['label_num'].map({cfg.REAL: 'REAL', cfg.FAKE: 'FAKE'})
combined_all['categoria'] = combined_all['label_num'].map({cfg.REAL: 'VERDADERA', cfg.FAKE: 'FALSA'})

# Orden canónico de columnas (las primeras son idénticas a Noticias_entre_70_y_370_palabras (1).xlsx)
columnas_orden = [
    'class', 'Text', 'Fuente', 'conteo_palabras_text',
    'label_num', 'label_name', 'categoria',
    'pais', 'subfuente_medio', 'tema', 'dataset_origen'
]
combined_all = combined_all[columnas_orden]

# Subconjunto filtrado entre 70 y 370 palabras
combined_70_370 = combined_all[
    (combined_all['conteo_palabras_text'] >= 70) & (combined_all['conteo_palabras_text'] <= 370)
].copy().reset_index(drop=True)

# =========================================================================
# 5. GUARDAR ARCHIVOS DEL CORPUS LATINOAMERICANO
# =========================================================================
path_latam_70_370_xlsx = os.path.join(dataset_dir, "Corpus_LatAm_entre_70_y_370_palabras.xlsx")
path_latam_70_370_csv = os.path.join(dataset_dir, "Corpus_LatAm_entre_70_y_370_palabras.csv")
path_latam_full_xlsx = os.path.join(dataset_dir, "Corpus_LatAm_Completo_sin_filtrar.xlsx")
path_latam_full_csv = os.path.join(dataset_dir, "Corpus_LatAm_Completo_sin_filtrar.csv")

print("\n>>> Guardando archivos del Corpus Latinoamericano...")
combined_70_370.to_excel(path_latam_70_370_xlsx, index=False)
combined_70_370.to_csv(path_latam_70_370_csv, index=False)
print(f"  [OK] Guardado: {path_latam_70_370_xlsx} (N = {len(combined_70_370)})")
print(f"  [OK] Guardado: {path_latam_70_370_csv}")

combined_all.to_excel(path_latam_full_xlsx, index=False)
combined_all.to_csv(path_latam_full_csv, index=False)
print(f"  [OK] Guardado: {path_latam_full_xlsx} (N = {len(combined_all)})")
print(f"  [OK] Guardado: {path_latam_full_csv}")

# =========================================================================
# 6. CREAR DATASET AMPLIADO (ORIGINAL + NUEVO LATAM DEDUPLICADO)
# =========================================================================
orig_path = os.path.join(dataset_dir, "Noticias_entre_70_y_370_palabras (1).xlsx")
print(f"\n>>> Ampliando dataset original: {orig_path}...")
df_orig = pd.read_excel(orig_path)
print(f"  Registros en dataset original: {len(df_orig)}")

# Identificar noticias nuevas de LatAm que no estén en el dataset original
df_orig_texts = set(df_orig['Text'].astype(str).str.lower().str.strip())
nuevas_latam = combined_70_370[
    ~combined_70_370['Text'].astype(str).str.lower().str.strip().isin(df_orig_texts)
].copy()
print(f"  Nuevas noticias latinoamericanas no redundantes a incorporar: {len(nuevas_latam)}")

# Completar metadatos en df_orig si no los tiene
if 'pais' not in df_orig.columns:
    def map_orig_country(src):
        s = str(src).lower()
        if 'mariagrandury' in s: return 'LatAm Multinacional (México/Colombia/Argentina)'
        elif 'freiren' in s or 'edds' in s: return 'España (Peninsular)'
        elif 'arseniitretiakov' in s: return 'España / Hispanoamérica (Web abierta)'
        return 'España / Internacional'
    df_orig['pais'] = df_orig['Fuente'].apply(map_orig_country)

if 'subfuente_medio' not in df_orig.columns:
    df_orig['subfuente_medio'] = df_orig['Fuente']

if 'tema' not in df_orig.columns:
    df_orig['tema'] = 'General / Fact-checking'

if 'dataset_origen' not in df_orig.columns:
    def map_orig_ds(src):
        s = str(src).lower()
        if 'freiren' in s: return 'Freiren Unified Corpus (España)'
        elif 'edds' in s: return 'Edds Fixed (España)'
        elif 'mariagrandury' in s: return 'FakeDeS 2021 (LatAm)'
        elif 'arseniitretiakov' in s: return 'Kaggle Spanish Fakes'
        return 'Original'
    df_orig['dataset_origen'] = df_orig['Fuente'].apply(map_orig_ds)

# Concatenar dataset original + nuevas noticias latinoamericanas
df_ampliado = pd.concat([df_orig[columnas_orden], nuevas_latam[columnas_orden]]).reset_index(drop=True)

path_ampliado_xlsx = os.path.join(dataset_dir, "Noticias_entre_70_y_370_palabras_AMPLIADO_LATAM.xlsx")
path_ampliado_csv = os.path.join(dataset_dir, "Noticias_entre_70_y_370_palabras_AMPLIADO_LATAM.csv")

df_ampliado.to_excel(path_ampliado_xlsx, index=False)
df_ampliado.to_csv(path_ampliado_csv, index=False)
print(f"  [OK] Dataset Ampliado Guardado: {path_ampliado_xlsx} (N = {len(df_ampliado)})")
print(f"  [OK] Dataset Ampliado Guardado: {path_ampliado_csv}")

print("\n" + "="*80)
print(" RESUMEN ESTADÍSTICO FINAL")
print("="*80)
print("\n1. CORPUS LATINOAMERICANO 70-370 PALABRAS:")
print(f"   Total Noticias: {len(combined_70_370)}")
print(f"   Verdaderas (0/REAL): {(combined_70_370['class']==True).sum()} ({((combined_70_370['class']==True).sum()/len(combined_70_370))*100:.2f}%)")
print(f"   Falsas (1/FAKE): {(combined_70_370['class']==False).sum()} ({((combined_70_370['class']==False).sum()/len(combined_70_370))*100:.2f}%)")
print(f"   Promedio Palabras: {combined_70_370['conteo_palabras_text'].mean():.2f} (std: {combined_70_370['conteo_palabras_text'].std():.2f})")
print(f"   Mediana Palabras: {combined_70_370['conteo_palabras_text'].median():.2f} (min: {combined_70_370['conteo_palabras_text'].min()}, max: {combined_70_370['conteo_palabras_text'].max()})")
print("   Distribución por País:\n", combined_70_370['pais'].value_counts())
print("   Distribución por Dataset:\n", combined_70_370['dataset_origen'].value_counts())

print("\n2. DATASET GLOBAL AMPLIADO (ESPAÑA + LATAM):")
print(f"   Total Noticias: {len(df_ampliado)}")
print(f"   Verdaderas (0/REAL): {(df_ampliado['class']==True).sum()} ({((df_ampliado['class']==True).sum()/len(df_ampliado))*100:.2f}%)")
print(f"   Falsas (1/FAKE): {(df_ampliado['class']==False).sum()} ({((df_ampliado['class']==False).sum()/len(df_ampliado))*100:.2f}%)")
print("   Distribución Geográfica:\n", df_ampliado['pais'].value_counts())
print("="*80)
