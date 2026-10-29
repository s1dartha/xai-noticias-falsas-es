"""
=============================================================================
EXTRACCIÓN DE LAS 5 DIMENSIONES TEXTUALES PARA EL MODELO SOBRE VARIABLES (FASE 2)
=============================================================================
Módulo: Modelos_Individuales/Clasificar_fake/Modelo_sobre_Variables/Codigos/extraer_features_avanzadas_5d.py
Tesis: Detección de Fake News mediante Análisis Multidimensional del Texto.

Implementa la extracción computacional por lotes sobre GPU de las variables forenses
definidas en Tesis/Explicabilidad_Propia:
- D1: Dinámica Discursiva y Coherencia Secuencial SBERT (consec_sim)
- D2: Lingüística Forense y Marcadores Epistémicos (dicendi, quotes, hedges, boosters)
- D3: Riqueza Léxica, Legibilidad y Sintaxis (Guiraud, Hapax, Flesch-Szigriszt)
- D4: Morfosintaxis spaCy POS Tagging (adv_density, pron_1p, pron_3p, adj_noun_ratio)
- D5: Anclajes Factuales y Mayúsculas Sostenidas (all_caps, numbers, temporal, punct)

Nota: estos bloques D1–D5 son una organización interna de la extracción y NO corresponden a las dimensiones D1–D5
del IML (Explicabilidad_Propia/codigos/estilos_discursivos.py). Solo las columnas que construir_tabla_variables.py
mapea a las 43 variables (familias S, R, O y L) se usan en la tesis; el resto del CSV son columnas exploratorias.
=============================================================================
"""

import sys as _sys_cfg
from pathlib import Path as _Path_cfg
_sys_cfg.path.insert(0, str(next(p for p in _Path_cfg(__file__).resolve().parents if (p / "config_proyecto.py").exists())))
import config_proyecto as cfg
import os
import sys
import re
import time
import spacy
import torch
import numpy as np
import pandas as pd
import nltk
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Usando dispositivo: {device}")

features_base_csv = str(cfg.ROOT / "Modelos_Individuales/Clasificar_fake/Modelo_sobre_Variables/Codigos/features_modelo_sobre_variables_ampliado_4418.csv")
print(f"⏳ Leyendo features base: {features_base_csv}")
df = pd.read_csv(features_base_csv)
print(f"✅ Cargadas {len(df)} noticias. Columnas actuales: {len(df.columns)}")

# -----------------------------------------------------------------------------
# 1. DIMENSIÓN 1: Dinámica Discursiva y Flujo de Coherencia Secuencial
# -----------------------------------------------------------------------------
print("⏳ [Dimensión 1/5] Computando Flujo de Coherencia Consecutiva con BETO SBERT...")
sbert_name = "hiiamsid/sentence_similarity_spanish_es"
sbert_model = SentenceTransformer(sbert_name, revision=cfg.MODELOS_HF['SBERT_Redundancia']['revision'], device=str(device))  # revisión fija

# Segmentar oraciones por documento
all_sentences_flat = []
doc_sent_spans = []

for text in df['Text']:
    sents = [s.strip() for s in nltk.sent_tokenize(str(text), language='spanish') if len(s.strip()) > 8]
    if len(sents) == 0:
        sents = [str(text).strip()[:200]]
    start = len(all_sentences_flat)
    all_sentences_flat.extend(sents)
    end = len(all_sentences_flat)
    doc_sent_spans.append((start, end))

print(f"📊 Codificando {len(all_sentences_flat)} oraciones en GPU...")
t0 = time.time()
embeddings = sbert_model.encode(
    all_sentences_flat,
    batch_size=256,
    show_progress_bar=True,
    convert_to_tensor=True,
    normalize_embeddings=True
)
print(f"✅ Embeddings listos en {time.time()-t0:.2f} s")

# Calcular similitud consecutiva s_i · s_{i+1}
consec_mean_list = []
consec_min_list = []
consec_std_list = []

for start, end in doc_sent_spans:
    k = end - start
    if k < 2:
        consec_mean_list.append(np.nan)
        consec_min_list.append(np.nan)
        consec_std_list.append(0.0)
    else:
        doc_embs = embeddings[start:end] # (k, 384)
        # Producto punto entre e_i y e_{i+1}
        e_cur = doc_embs[:-1]
        e_next = doc_embs[1:]
        consec_sims = torch.sum(e_cur * e_next, dim=1).cpu().numpy()
        
        consec_mean_list.append(float(np.mean(consec_sims)))
        consec_min_list.append(float(np.min(consec_sims)))
        consec_std_list.append(float(np.std(consec_sims)))

med_cmean = np.nanmedian(consec_mean_list)
med_cmin = np.nanmedian(consec_min_list)

df['consec_sim_mean'] = pd.Series(consec_mean_list).fillna(med_cmean)
df['consec_sim_min'] = pd.Series(consec_min_list).fillna(med_cmin)
df['consec_sim_std'] = pd.Series(consec_std_list).fillna(0.0)

# Liberar memoria de GPU
del sbert_model
del embeddings
torch.cuda.empty_cache()

# -----------------------------------------------------------------------------
# 2. DIMENSIÓN 2: Lingüística Forense y Marcadores Epistémicos (Certeza vs Cautela)
# -----------------------------------------------------------------------------
print("⏳ [Dimensión 2/5] Extrayendo Marcadores Epistémicos, Hedges, Boosters y Citas...")

# Regex compilados para español
PAT_HEDGES = re.compile(
    r'\b(presunto|presunta|presuntos|presuntas|presuntamente|al parecer|según fuentes|'
    r'podría|podrían|se presume|supuesto|supuesta|supuestos|supuestamente|aparentemente|'
    r'probablemente|posiblemente|se sospecha|se infiere|de confirmarse|no se descarta|'
    r'trascendió que|habría sido|habrían sido)\b',
    re.IGNORECASE
)

PAT_BOOSTERS = re.compile(
    r'\b(indudablemente|obviamente|claramente|sin duda|sin lugar a dudas|es un hecho|'
    r'irrefutable|totalmente|absolutamente|jamás|nunca jamás|la verdad oculta|revelado|'
    r'comprobado|definitivamente|rotundamente|innegable|a ciencia cierta|categóricamente|'
    r'escándalo total|evidencia irrefutable|científicamente comprobado)\b',
    re.IGNORECASE
)

PAT_DICENDI = re.compile(
    r'\b(dijo|declaró|manifestó|indicó|explicó|informó|expresó|afirmó|sostuvo|aseveró|'
    r'aseguró|añadió|puntualizó|precisó|concluyó|señaló|comentó|declararon|informaron)\b',
    re.IGNORECASE
)

PAT_QUOTES = re.compile(r'["«»“”]')

hedges_dens_list = []
boosters_dens_list = []
epistemic_ratio_list = []
quotes_dens_list = []
dicendi_dens_list = []

for text, n_words, n_sents in zip(df['Text'], df['conteo_palabras_text'], df['num_sentences']):
    t_str = str(text)
    w_count = max(n_words, 1)
    s_count = max(n_sents, 1)
    
    n_hedges = len(PAT_HEDGES.findall(t_str))
    n_boosters = len(PAT_BOOSTERS.findall(t_str))
    n_quotes = len(PAT_QUOTES.findall(t_str))
    n_dicendi = len(PAT_DICENDI.findall(t_str))
    
    h_dens = (n_hedges / w_count) * 100.0
    b_dens = (n_boosters / w_count) * 100.0
    ep_ratio = float(n_boosters / (n_hedges + 0.1))
    q_dens = float(n_quotes / s_count)
    dic_dens = (n_dicendi / w_count) * 100.0
    
    hedges_dens_list.append(h_dens)
    boosters_dens_list.append(b_dens)
    epistemic_ratio_list.append(ep_ratio)
    quotes_dens_list.append(q_dens)
    dicendi_dens_list.append(dic_dens)

df['hedges_density'] = hedges_dens_list
df['boosters_density'] = boosters_dens_list
df['epistemic_ratio'] = epistemic_ratio_list
df['quotes_density'] = quotes_dens_list
df['dicendi_density'] = dicendi_dens_list

# -----------------------------------------------------------------------------
# 3. DIMENSIÓN 3: Riqueza Léxica, Complejidad Sintáctica y Legibilidad
# -----------------------------------------------------------------------------
print("⏳ [Dimensión 3/5] Calculando Índices de Legibilidad (Flesch-Szigriszt, Guiraud TTR)...")

def count_syllables_es(word):
    # Conteo heurístico de núcleos silábicos en español
    word = word.lower()
    # Tratamiento de diptongos comunes
    vowels = "aeiouáéíóúü"
    count = 0
    in_vowel = False
    for char in word:
        if char in vowels:
            if not in_vowel:
                count += 1
                in_vowel = True
        else:
            in_vowel = False
    return max(1, count)

flesch_list = []
gp_list = []
guiraud_list = []
hapax_list = []

for text, n_words, n_sents in zip(df['Text'], df['conteo_palabras_text'], df['num_sentences']):
    t_clean = re.sub(r'[^\w\s]', '', str(text).lower())
    tokens = t_clean.split()
    w_count = max(len(tokens), 1)
    s_count = max(n_sents, 1)
    
    # Sílabas y letras
    total_syllables = sum(count_syllables_es(tok) for tok in tokens)
    total_chars = sum(len(tok) for tok in tokens)
    
    # Flesch-Szigriszt para español
    # IFSZ = 206.835 - 62.3 * (sílabas/palabras) - (palabras/frases)
    ifsz = 206.835 - 62.3 * (total_syllables / w_count) - (w_count / s_count)
    
    # Gutiérrez de Polini
    # GP = 95.2 - 9.7 * (letras/palabras) - (palabras/frases)
    gp = 95.2 - 9.7 * (total_chars / w_count) - (w_count / s_count)
    
    # Vocabulario
    vocab = set(tokens)
    guiraud = len(vocab) / np.sqrt(w_count)
    
    # Hapax Legomena (palabras únicas con frecuencia 1)
    counts = pd.Series(tokens).value_counts()
    n_hapax = int((counts == 1).sum())
    hapax_ratio = float(n_hapax / w_count)
    
    flesch_list.append(float(np.clip(ifsz, 0, 120)))
    gp_list.append(float(np.clip(gp, 0, 100)))
    guiraud_list.append(float(guiraud))
    hapax_list.append(float(hapax_ratio))

df['flesch_szigriszt'] = flesch_list
df['gutierrez_polini'] = gp_list
df['guiraud_ttr'] = guiraud_list
df['hapax_ratio'] = hapax_list

# -----------------------------------------------------------------------------
# 4. DIMENSIÓN 4: Perfilado Morfosintáctico (POS Tagging) y Subjetividad
# -----------------------------------------------------------------------------
print("⏳ [Dimensión 4/5] Ejecutando POS Tagging con spaCy (Adjetivos, Pronombres)...")
nlp = spacy.load('es_core_news_sm', disable=['ner', 'parser'])

PRON_1P = {'yo', 'nosotros', 'nosotras', 'me', 'nos', 'mi', 'mis', 'mío', 'míos', 'mía', 'mías', 'nuestro', 'nuestra', 'nuestros', 'nuestras'}
PRON_3P = {'él', 'ella', 'ello', 'ellos', 'ellas', 'se', 'le', 'les', 'lo', 'los', 'la', 'las', 'su', 'sus', 'suyo', 'suya', 'suyos', 'suyas'}

adj_noun_ratio_list = []
pron_1p_dens_list = []
pron_3p_dens_list = []
adv_dens_list = []

for doc, n_words in zip(tqdm(nlp.pipe(df['Text'].astype(str), batch_size=100), total=len(df), desc="spaCy POS"), df['conteo_palabras_text']):
    w_count = max(n_words, 1)
    
    n_adj = 0
    n_noun = 0
    n_adv = 0
    n_p1 = 0
    n_p3 = 0
    
    for token in doc:
        pos = token.pos_
        t_low = token.text.lower()
        if pos == 'ADJ':
            n_adj += 1
        elif pos == 'NOUN':
            n_noun += 1
        elif pos == 'ADV':
            n_adv += 1
        
        if pos == 'PRON' or t_low in PRON_1P or t_low in PRON_3P:
            if t_low in PRON_1P:
                n_p1 += 1
            elif t_low in PRON_3P:
                n_p3 += 1

    adj_noun_ratio = float(n_adj / (n_noun + 1))
    p1_dens = (n_p1 / w_count) * 100.0
    p3_dens = (n_p3 / w_count) * 100.0
    adv_dens = (n_adv / w_count) * 100.0
    
    adj_noun_ratio_list.append(adj_noun_ratio)
    pron_1p_dens_list.append(p1_dens)
    pron_3p_dens_list.append(p3_dens)
    adv_dens_list.append(adv_dens)

df['adj_noun_ratio'] = adj_noun_ratio_list
df['pron_1p_density'] = pron_1p_dens_list
df['pron_3p_density'] = pron_3p_dens_list
df['adv_density'] = adv_dens_list

# -----------------------------------------------------------------------------
# 5. DIMENSIÓN 5: Anclajes Factuales, Mayúsculas Sostenidas y Puntuación
# -----------------------------------------------------------------------------
print("⏳ [Dimensión 5/5] Extrayendo Mayúsculas Sostenidas, Puntuación Enfática y Anclajes Numéricos...")

PAT_ALL_CAPS = re.compile(r'\b[A-ZÁÉÍÓÚÑ]{3,}\b')
PAT_CENSORED_MASKS = re.compile(r'\*(?:NUMBER|URL|PHONE|\d+)\*', re.IGNORECASE)
PAT_CENSORED_NUMBER = re.compile(r'\*(?:NUMBER|\d+)\*', re.IGNORECASE)
PAT_EXCLAMATION = re.compile(r'[!¡]')
PAT_QUESTION = re.compile(r'[?¿]')
PAT_ELLIPSIS = re.compile(r'(\.\.\.|…)')
PAT_DIGITS = re.compile(r'\b\d+(?:[\.,]\d+)?\b')
PAT_PERCENT = re.compile(r'(?:\d+\s*%|\bpor\s+ciento\b)', re.IGNORECASE)
PAT_TEMPORAL = re.compile(r'\b(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre|2019|2020|2021|2022|2023|2024|2025|2026)\b', re.IGNORECASE)

SIGLAS_EXCLUIDAS = {
    'ONU', 'UE', 'EEUU', 'OMS', 'PIB', 'FMI', 'OTAN', 'VOX', 'PSOE', 'PP', 'INE',
    'USA', 'EFE', 'IAC', 'NICA', 'NUMBER', 'URL', 'PHONE'
}

all_caps_count_list = []
all_caps_ratio_list = []
upper_chars_ratio_list = []
excl_dens_list = []
quest_dens_list = []
ellipsis_dens_list = []
punct_intensity_list = []
numbers_dens_list = []
percent_count_list = []
temporal_dens_list = []

for text, n_words in zip(df['Text'], df['conteo_palabras_text']):
    t_str = str(text)
    t_str_no_masks = PAT_CENSORED_MASKS.sub(' ', t_str)
    w_count = max(n_words, 1)
    
    # Mayúsculas sostenidas (excluyendo máscaras de anonimización *NUMBER*, *URL*, *PHONE* y siglas)
    caps_words = PAT_ALL_CAPS.findall(t_str_no_masks)
    caps_filt = [w for w in caps_words if w not in SIGLAS_EXCLUIDAS]
    n_caps = len(caps_filt)
    caps_ratio = float(n_caps / w_count) * 100.0
    
    # Ratio de caracteres en mayúsculas sobre letras (sin contar máscaras de anonimización)
    alpha_chars = [c for c in t_str_no_masks if c.isalpha()]
    n_upper = sum(1 for c in alpha_chars if c.isupper())
    upper_chars_ratio = float(n_upper / max(len(alpha_chars), 1))
    
    # Puntuación
    n_excl = len(PAT_EXCLAMATION.findall(t_str_no_masks))
    n_quest = len(PAT_QUESTION.findall(t_str_no_masks))
    n_ellip = len(PAT_ELLIPSIS.findall(t_str_no_masks))
    
    excl_dens = (n_excl / w_count) * 100.0
    quest_dens = (n_quest / w_count) * 100.0
    ellip_dens = (n_ellip / w_count) * 100.0
    punct_intens = ((n_excl + n_quest + n_ellip) / w_count) * 100.0
    
    # Anclajes factuales (contabilizando tanto cifras explícitas como máscaras numéricas *NUMBER*)
    n_digits = len(PAT_DIGITS.findall(t_str_no_masks)) + len(PAT_CENSORED_NUMBER.findall(t_str))
    n_percent = len(PAT_PERCENT.findall(t_str))
    n_temp = len(PAT_TEMPORAL.findall(t_str_no_masks))
    
    num_dens = (n_digits / w_count) * 100.0
    temp_dens = (n_temp / w_count) * 100.0
    
    all_caps_count_list.append(n_caps)
    all_caps_ratio_list.append(caps_ratio)
    upper_chars_ratio_list.append(upper_chars_ratio)
    excl_dens_list.append(excl_dens)
    quest_dens_list.append(quest_dens)
    ellipsis_dens_list.append(ellip_dens)
    punct_intensity_list.append(punct_intens)
    numbers_dens_list.append(num_dens)
    percent_count_list.append(n_percent)
    temporal_dens_list.append(temp_dens)

df['all_caps_count'] = all_caps_count_list
df['all_caps_ratio'] = all_caps_ratio_list
df['upper_chars_ratio'] = upper_chars_ratio_list
df['excl_density'] = excl_dens_list
df['quest_density'] = quest_dens_list
df['ellipsis_density'] = ellipsis_dens_list
df['punct_intensity'] = punct_intensity_list
df['numbers_density'] = numbers_dens_list
df['percent_count'] = percent_count_list
df['temporal_density'] = temporal_dens_list

# Guardar dataset con todas las características enriquecidas
out_csv = str(cfg.ROOT / "Modelos_Individuales/Clasificar_fake/Modelo_sobre_Variables/Codigos/features_modelo_sobre_variables_ampliado_avanzado_4418.csv")
df.to_csv(out_csv, index=False)
print(f"\n🎉 ¡Todas las características de las 5 dimensiones extraídas y guardadas!")
print(f"📁 Archivo: {out_csv} (Total columnas: {len(df.columns)})")
