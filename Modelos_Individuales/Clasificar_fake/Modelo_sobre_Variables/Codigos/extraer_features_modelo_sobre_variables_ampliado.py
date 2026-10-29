"""
Extracción de origen de las variables de sensacionalismo (JJNeila) y redundancia (hiiamsid) sobre el corpus ampliado
(4.418 noticias). Salida: features_modelo_sobre_variables_ampliado_4418.csv (y sincroniza las columnas en
features_modelo_sobre_variables_ampliado_avanzado_4418.csv si existe).

Solo las columnas que construir_tabla_variables.py mapea a las variables S y R se usan en la tesis; los umbrales
semánticos y las discretizaciones de este script son exploratorios y no entran en la tabla de 43 variables.
"""
import sys as _sys_cfg
from pathlib import Path as _Path_cfg
_sys_cfg.path.insert(0, str(next(p for p in _Path_cfg(__file__).resolve().parents if (p / "config_proyecto.py").exists())))
import config_proyecto as cfg
import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd
import nltk
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Usando dispositivo: {device}")

# Jerarquía de Umbrales Semánticos Calibrados con BETO SBERT (hiiamsid/sentence_similarity_spanish_es)
TAU_CONTINUIDAD_GMM = 0.3434      # Continuidad temática basal (GMM)
TAU_PARAFRASIS_PAWSX = 0.5340     # Paráfrasis moderada (P90 / PAWS-X = 0.533)
TAU_REDUNDANCIA_ESTRICTA = 0.5996 # Redundancia semántica estricta (P95 / Shannon = 0.6079)
TAU_HIPER_REDUNDANCIA = 0.7287    # Hiper-redundancia circular (P99)

# 1. Cargar dataset ampliado (N = 4.418) o reusar Sensacionalismo BETO ya calculado
dataset_path = str(cfg.ROOT / "Modelos_Individuales/Clasificar_fake/Dataset/Noticias_entre_70_y_370_palabras_AMPLIADO_LATAM.csv")
out_csv = str(cfg.ROOT / "Modelos_Individuales/Clasificar_fake/Modelo_sobre_Variables/Codigos/features_modelo_sobre_variables_ampliado_4418.csv")

if os.path.exists(out_csv):
    print(f"⏳ Leyendo base ampliada con Sensacionalismo BETO en caché: {out_csv}")
    df = pd.read_csv(out_csv)
else:
    print(f"⏳ Leyendo dataset base: {dataset_path}")
    df = pd.read_csv(dataset_path)
# La etiqueta del modelo siempre se recalcula desde el corpus (clase_num: 1 = falsa), incluso con caché
df['label_num'] = df['clase_num'].astype(int)
assert set(df['label_num'].unique()) <= {cfg.REAL, cfg.FAKE}

# Segmentar oraciones
print("⏳ [Paso 1/3] Segmentando oraciones del corpus ampliado (N = 4.418)...")
all_sentences_flat = []
doc_sentence_indices = []

for text in df['Text']:
    sents = [s.strip() for s in nltk.sent_tokenize(str(text), language='spanish') if len(s.strip()) > 8]
    if len(sents) == 0:
        sents = [str(text).strip()[:200]]
    start_idx = len(all_sentences_flat)
    all_sentences_flat.extend(sents)
    end_idx = len(all_sentences_flat)
    doc_sentence_indices.append((start_idx, end_idx, sents))

print(f"📊 Total de oraciones extraídas en el corpus: {len(all_sentences_flat)}")

# Si no están en caché las variables de sensacionalismo BETO, calcularlas
if 'P_full' not in df.columns or 'delta_p_gatillo' not in df.columns:
    model_sens_name = "JJNeila/bert-spanish-sensationalism-oss"
    print(f"⏳ Cargando BETO Sensacionalismo ({model_sens_name})...")
    rev_sens = cfg.MODELOS_HF['Sensacionalismo_JJNeila']['revision']  # revisión fija (config_proyecto.py)
    tokenizer_sens = AutoTokenizer.from_pretrained(model_sens_name, revision=rev_sens)
    model_sens = AutoModelForSequenceClassification.from_pretrained(model_sens_name, revision=rev_sens).to(device)
    model_sens.eval()

    def predict_sensationalism_batch(text_list, batch_size=64, max_len=256):
        probs_all = []
        for i in range(0, len(text_list), batch_size):
            batch = [str(t)[:1000] for t in text_list[i:i+batch_size]]
            inputs = tokenizer_sens(batch, padding=True, truncation=True, max_length=max_len, return_tensors='pt').to(device)
            with torch.no_grad():
                logits = model_sens(**inputs).logits
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                probs_all.extend(probs.tolist())
        return probs_all

    df['P_full'] = predict_sensationalism_batch(df['Text'].tolist(), batch_size=64, max_len=256)
    all_sentence_probs = predict_sensationalism_batch(all_sentences_flat, batch_size=128, max_len=128)

    p_mean_list, p_max_list, p_top2_list, sigma_sens_list, dr_list = [], [], [], [], []
    ablated_texts, doc_needs_ablation = [], []

    for i, (start_idx, end_idx, sents) in enumerate(doc_sentence_indices):
        s_probs = all_sentence_probs[start_idx:end_idx]
        p_full = df['P_full'].iloc[i]
        p_mean = float(np.mean(s_probs))
        p_max = float(np.max(s_probs))
        top_indices = np.argsort(s_probs)[::-1]
        if len(s_probs) >= 2:
            p_top2 = float((s_probs[top_indices[0]] + s_probs[top_indices[1]]) / 2.0)
            sigma = float(np.std(s_probs))
        else:
            p_top2 = p_max
            sigma = 0.0
        dr = float(p_full / (p_max + 1e-6))
        p_mean_list.append(p_mean)
        p_max_list.append(p_max)
        p_top2_list.append(p_top2)
        sigma_sens_list.append(sigma)
        dr_list.append(dr)
        if len(sents) >= 3:
            top2_set = {top_indices[0], top_indices[1]}
            ablated_sents = [s for s_idx, s in enumerate(sents) if s_idx not in top2_set]
            ablated_texts.append(" ".join(ablated_sents))
            doc_needs_ablation.append((i, True))
        else:
            doc_needs_ablation.append((i, False))

    ablated_probs = predict_sensationalism_batch(ablated_texts, batch_size=64, max_len=256)
    delta_p_gatillo_list = [0.0] * len(df)
    abl_idx = 0
    for doc_idx, has_abl in doc_needs_ablation:
        if has_abl:
            p_abl = ablated_probs[abl_idx]
            abl_idx += 1
            delta_p_gatillo_list[doc_idx] = float(df['P_full'].iloc[doc_idx] - p_abl)
        else:
            delta_p_gatillo_list[doc_idx] = float(df['P_full'].iloc[doc_idx] - p_mean_list[doc_idx])

    df['P_mean'] = p_mean_list
    df['P_max'] = p_max_list
    df['P_top2'] = p_top2_list
    df['sigma_sens'] = sigma_sens_list
    df['dilution_ratio'] = dr_list
    df['delta_p_gatillo'] = delta_p_gatillo_list
    del model_sens, tokenizer_sens
    torch.cuda.empty_cache()
else:
    print("✅ Variables de Sensacionalismo BETO (JJNeila) ya presentes en caché.")

# 2. Cargar Modelo Ganador BETO SBERT para Redundancia (d=768)
sbert_name = "hiiamsid/sentence_similarity_spanish_es"
print(f"⏳ [Paso 2/3] Cargando Modelo Ganador BETO SBERT ({sbert_name}, d=768)...")
sbert_model = SentenceTransformer(sbert_name, revision=cfg.MODELOS_HF['SBERT_Redundancia']['revision'], device=str(device))  # revisión fija

print(f"⏳ Codificando {len(all_sentences_flat)} oraciones con BETO SBERT en GPU...")
t0 = time.time()
all_embeddings = sbert_model.encode(
    all_sentences_flat,
    batch_size=128,
    show_progress_bar=True,
    convert_to_tensor=True,
    normalize_embeddings=True
)
t_sbert = time.time() - t0
print(f"✅ BETO SBERT completado en {t_sbert:.2f} s ({t_sbert/len(all_sentences_flat)*1000:.2f} ms/oración).")

# 3. Calcular métricas de redundancia intra-documental, multi-umbral, MexGen y flujo secuencial
print("⏳ [Paso 3/3] Calculando matriz de similitud coseno intra-documento, jerarquía multi-umbral y MexGen...")
num_sentences_list = []
num_pairs_list = []
mean_sim_list = []
max_sim_list = []
second_max_sim_list = []
top3_mean_sim_list = []
p90_sim_list = []
var_sim_list = []

topic_cont_pairs_list = []
topic_cont_density_list = []
paraphrase_p90_pairs_list = []
paraphrase_p90_density_list = []
strict_p95_pairs_list = []
strict_p95_density_list = []
hyper_pairs_list = []
hyper_density_list = []

max_post_ablation_list = []
mexgen_gap_list = []
delta_mean_morf_list = []

consec_mean_list = []
consec_min_list = []
consec_std_list = []

for i, (start_idx, end_idx, sents) in enumerate(tqdm(doc_sentence_indices, desc="Redundancia BETO SBERT por doc")):
    n_sents = end_idx - start_idx
    num_sentences_list.append(n_sents)

    if n_sents < 2:
        num_pairs_list.append(0)
        mean_sim_list.append(np.nan)
        max_sim_list.append(np.nan)
        second_max_sim_list.append(np.nan)
        top3_mean_sim_list.append(np.nan)
        p90_sim_list.append(np.nan)
        var_sim_list.append(np.nan)

        topic_cont_pairs_list.append(0)
        topic_cont_density_list.append(0.0)
        paraphrase_p90_pairs_list.append(0)
        paraphrase_p90_density_list.append(0.0)
        strict_p95_pairs_list.append(0)
        strict_p95_density_list.append(0.0)
        hyper_pairs_list.append(0)
        hyper_density_list.append(0.0)

        max_post_ablation_list.append(np.nan)
        mexgen_gap_list.append(0.0)
        delta_mean_morf_list.append(0.0)

        consec_mean_list.append(np.nan)
        consec_min_list.append(np.nan)
        consec_std_list.append(0.0)
    else:
        doc_emb = all_embeddings[start_idx:end_idx]  # (n_sents, 768)
        sim_matrix = torch.matmul(doc_emb, doc_emb.T).cpu().numpy()

        # Flujo secuencial adyacente
        consec_sims = np.array([sim_matrix[idx, idx + 1] for idx in range(n_sents - 1)], dtype=float)
        consec_mean_list.append(float(np.mean(consec_sims)))
        consec_min_list.append(float(np.min(consec_sims)))
        consec_std_list.append(float(np.std(consec_sims)))

        # Triángulo superior estricto
        triu_indices = np.triu_indices(n_sents, k=1)
        pairwise_sims = sim_matrix[triu_indices]
        n_pairs = len(pairwise_sims)
        sorted_pairs = np.sort(pairwise_sims)[::-1]

        mean_sim = float(np.mean(pairwise_sims))
        max_sim = float(sorted_pairs[0])
        second_max_sim = float(sorted_pairs[1]) if n_pairs >= 2 else max_sim
        top3_mean_sim = float(np.mean(sorted_pairs[:min(3, n_pairs)]))
        p90_sim = float(np.percentile(pairwise_sims, 90))
        var_sim = float(np.var(pairwise_sims))

        n_topic = int(np.sum(pairwise_sims >= TAU_CONTINUIDAD_GMM))
        n_pawsx = int(np.sum(pairwise_sims >= TAU_PARAFRASIS_PAWSX))
        n_strict = int(np.sum(pairwise_sims >= TAU_REDUNDANCIA_ESTRICTA))
        n_hyper = int(np.sum(pairwise_sims >= TAU_HIPER_REDUNDANCIA))

        num_pairs_list.append(n_pairs)
        mean_sim_list.append(mean_sim)
        max_sim_list.append(max_sim)
        second_max_sim_list.append(second_max_sim)
        top3_mean_sim_list.append(top3_mean_sim)
        p90_sim_list.append(p90_sim)
        var_sim_list.append(var_sim)

        topic_cont_pairs_list.append(n_topic)
        topic_cont_density_list.append(float(n_topic / n_pairs))
        paraphrase_p90_pairs_list.append(n_pawsx)
        paraphrase_p90_density_list.append(float(n_pawsx / n_pairs))
        strict_p95_pairs_list.append(n_strict)
        strict_p95_density_list.append(float(n_strict / n_pairs))
        hyper_pairs_list.append(n_hyper)
        hyper_density_list.append(float(n_hyper / n_pairs))

        # Ablación Oracional No Tautológica (MexGen)
        if n_sents >= 3:
            max_pair_idx = np.unravel_index(np.argmax(sim_matrix - np.eye(n_sents) * 999.0), sim_matrix.shape)
            i_star, j_star = int(max_pair_idx[0]), int(max_pair_idx[1])
            row_means = (np.sum(sim_matrix, axis=1) - np.diag(sim_matrix)) / (n_sents - 1)
            s_star = i_star if row_means[i_star] >= row_means[j_star] else j_star
            s_hub = int(np.argmax(row_means))

            keep_mask = [idx for idx in range(n_sents) if idx != s_star]
            sub_mat = sim_matrix[np.ix_(keep_mask, keep_mask)]
            max_post = float(np.max(sub_mat[np.triu_indices(len(keep_mask), k=1)]))
            mexgen_gap = float(max_sim - max_post)

            keep_hub = [idx for idx in range(n_sents) if idx != s_hub]
            sub_hub = sim_matrix[np.ix_(keep_hub, keep_hub)]
            mean_post_hub = float(np.mean(sub_hub[np.triu_indices(len(keep_hub), k=1)]))
            delta_morf = float(mean_sim - mean_post_hub)
        else:
            max_post = second_max_sim
            mexgen_gap = 0.0
            delta_morf = 0.0

        max_post_ablation_list.append(max_post)
        mexgen_gap_list.append(mexgen_gap)
        delta_mean_morf_list.append(delta_morf)

df['num_sentences'] = num_sentences_list
df['num_pairs'] = num_pairs_list
df['mean_intra_similarity'] = mean_sim_list
df['max_intra_similarity'] = max_sim_list
df['second_max_intra_similarity'] = second_max_sim_list
df['top3_mean_intra_similarity'] = top3_mean_sim_list
df['p90_intra_similarity'] = p90_sim_list
df['var_intra_similarity'] = var_sim_list

df['topic_continuity_pairs_count'] = topic_cont_pairs_list
df['topic_continuity_density'] = topic_cont_density_list
df['paraphrase_pairs_count_p90'] = paraphrase_p90_pairs_list
df['paraphrase_density_p90'] = paraphrase_p90_density_list
df['strict_redundant_pairs_count_p95'] = strict_p95_pairs_list
df['strict_redundancy_density_p95'] = strict_p95_density_list
df['hyper_redundant_pairs_count'] = hyper_pairs_list
df['hyper_redundancy_density'] = hyper_density_list

df['redundant_pairs_count'] = strict_p95_pairs_list
df['redundancy_density'] = strict_p95_density_list

df['max_post_ablation_mexgen'] = max_post_ablation_list
df['mexgen_persistence_gap'] = mexgen_gap_list
df['delta_mean_sim_morf'] = delta_mean_morf_list

# Imputar nulos en docs de 1 sola frase con la mediana
for col in ['max_intra_similarity', 'mean_intra_similarity', 'second_max_intra_similarity',
            'top3_mean_intra_similarity', 'p90_intra_similarity', 'var_intra_similarity', 'max_post_ablation_mexgen']:
    med = df[col].dropna().median()
    df[f'{col}_clean'] = df[col].fillna(med)

df['consec_sim_mean'] = pd.Series(consec_mean_list).fillna(pd.Series(consec_mean_list).median())
df['consec_sim_min'] = pd.Series(consec_min_list).fillna(pd.Series(consec_min_list).median())
df['consec_sim_std'] = pd.Series(consec_std_list).fillna(0.0)

# Discretizaciones canónicas sobre BETO SBERT (en CV se calculan In-Fold sin fuga)
cuts_4 = [-np.inf, 0.3740, 0.6079, 0.7287, np.inf]
df['shannon_bin_4'] = pd.cut(df['max_intra_similarity_clean'], bins=cuts_4, labels=[1, 2, 3, 4]).astype(int)

cuts_7 = [-np.inf, 0.3740, 0.4500, 0.5340, 0.6079, 0.6650, 0.7287, np.inf]
df['shannon_bin_7'] = pd.cut(df['max_intra_similarity_clean'], bins=cuts_7, labels=[1, 2, 3, 4, 5, 6, 7]).astype(int)

df.to_csv(out_csv, index=False)
print(f"🎉 ¡Características BETO SBERT extraídas y guardadas en: {out_csv} ({len(df)} registros)!")

# Actualizar también features_modelo_sobre_variables_ampliado_avanzado_4418.csv sincronizando todas las columnas nuevas de D0 y D1
adv_csv = str(cfg.ROOT / "Modelos_Individuales/Clasificar_fake/Modelo_sobre_Variables/Codigos/features_modelo_sobre_variables_ampliado_avanzado_4418.csv")
if os.path.exists(adv_csv):
    df_adv = pd.read_csv(adv_csv)
    if len(df_adv) == len(df):
        # La sincronizacion es por posicion: se exige que ambos archivos tengan los mismos textos en el mismo orden
        assert (df_adv['Text'].astype(str).values == df['Text'].astype(str).values).all(), "Filas desalineadas"
        for col in df.columns:
            df_adv[col] = df[col].values
        df_adv.to_csv(adv_csv, index=False)
        print(f"🔄 Sincronizado automáticamente: {adv_csv} ({len(df_adv.columns)} columnas)")
