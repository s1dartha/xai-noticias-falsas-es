"""
Pipeline reproducible de redundancia semántica intra-documental con un único modelo:
hiiamsid/sentence_similarity_spanish_es (Sentence-BERT sobre BETO, d = 768).

Pasos:
  1. Calibración de umbrales con STS-B en español (similitud anotada por personas de 0 a 5).
     Los umbrales se eligen en la partición dev y se evalúan en test:
       - tau_tema       : separa pares del mismo tema (puntaje >= 1) de pares de temas distintos
       - tau_parafrasis : separa pares aproximadamente equivalentes (>= 3)
       - tau_redundante : separa pares casi equivalentes, es decir, que repiten la misma afirmación (>= 4)
  2. Validación externa en PAWS-X en español (umbral elegido en validation, evaluado en test).
  3. Cálculo de las métricas de redundancia de cada noticia del corpus base (2.604).
  4. Asociación descriptiva con la veracidad (1 = falsa) sobre textos únicos, sin umbrales
     ajustados con las etiquetas (no hay estratos supervisados en este pipeline).

Salidas:
  Dataset/redundancia_corpus_hiiamsid.csv y Reportes/metricas_redundancia_hiiamsid.json
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import nltk
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, recall_score, accuracy_score

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config_proyecto.py").exists())))
import config_proyecto as cfg

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

np.random.seed(cfg.SEED)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

MODELO = cfg.MODELOS_HF['SBERT_Redundancia']
BATCH = 64
N_BOOT = 2000

# Niveles de la guía de anotación de STS (Agirre et al.): 5 equivalentes, 4 casi equivalentes,
# 3 aproximadamente equivalentes, 2 comparten detalles, 1 mismo tema, 0 temas distintos.
NIVELES = {
    'tau_tema': (1.0, 'Mismo tema (STS >= 1)'),
    'tau_parafrasis': (3.0, 'Aproximadamente equivalentes (STS >= 3)'),
    'tau_redundante': (4.0, 'Casi equivalentes: repiten la misma afirmación (STS >= 4)'),
}


def cargar_modelo():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODELO['id'], revision=MODELO['revision'], device='cpu')


def coseno_pares(modelo, a, b):
    ea = modelo.encode(list(a), batch_size=BATCH, normalize_embeddings=True, show_progress_bar=False)
    eb = modelo.encode(list(b), batch_size=BATCH, normalize_embeddings=True, show_progress_bar=False)
    return np.sum(ea * eb, axis=1)


def umbral_youden(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    i = int(np.argmax(tpr - fpr))
    return float(thr[i])


def metricas_binarias(y, s, tau):
    pred = (s >= tau).astype(int)
    return {
        'roc_auc': float(roc_auc_score(y, s)),
        'accuracy': float(accuracy_score(y, pred)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'prevalencia_positiva': float(np.mean(y)),
    }


def segmentar(texto):
    # Misma regla que Explicabilidad_Propia/codigos/variables_texto.py: Punkt en español, oraciones de más de 8 caracteres
    oraciones = [o.strip() for o in nltk.sent_tokenize(str(texto), language='spanish') if len(o.strip()) > 8]
    return oraciones if oraciones else [str(texto)[:200]]


def bootstrap_diferencia_medianas(a, b, n_boot=N_BOOT):
    rng = np.random.default_rng(cfg.SEED)
    difs = [np.median(rng.choice(a, len(a))) - np.median(rng.choice(b, len(b))) for _ in range(n_boot)]
    return [float(np.percentile(difs, 2.5)), float(np.percentile(difs, 97.5))]


def main():
    t_inicio = time.time()
    print("=" * 80)
    print(f" PIPELINE DE REDUNDANCIA SEMÁNTICA ({MODELO['id']})")
    print("=" * 80)
    modelo = cargar_modelo()
    resultados = {'metadata': {
        'modelo': MODELO['id'], 'revision': MODELO['revision'], 'semilla': cfg.SEED,
        'segmentacion': "nltk.sent_tokenize(language='spanish'), oraciones > 8 caracteres",
        'etiqueta_veracidad': '1 = falsa, 0 = real (config_proyecto.etiqueta_binaria)',
    }}

    # ------------------------------------------------------------------
    # 1. CALIBRACIÓN CON STS-B ES
    # ------------------------------------------------------------------
    print("\n[1/4] Calibración de umbrales con STS-B en español...")
    sts = {}
    for particion in ['dev', 'test']:
        df = pd.read_parquet(cfg.REDUNDANCIA_BENCH_DIR / f"stsb_es_{particion}.parquet")
        df['cos'] = coseno_pares(modelo, df.sentence1, df.sentence2)
        sts[particion] = df
    test = sts['test']
    resultados['sts_es'] = {
        'n_dev': len(sts['dev']), 'n_test': len(test),
        'spearman_test': float(spearmanr(test.cos, test.similarity_score).statistic),
        'pearson_test': float(pearsonr(test.cos, test.similarity_score).statistic),
        'umbrales': {},
    }
    umbrales = {}
    for nombre, (corte, descripcion) in NIVELES.items():
        y_dev = (sts['dev'].similarity_score >= corte).astype(int).values
        tau = umbral_youden(y_dev, sts['dev'].cos.values)
        y_test = (test.similarity_score >= corte).astype(int).values
        umbrales[nombre] = tau
        resultados['sts_es']['umbrales'][nombre] = {
            'descripcion': descripcion, 'corte_sts': corte, 'tau': tau,
            'criterio': 'Youden en STS-B dev',
            'test': metricas_binarias(y_test, test.cos.values, tau),
        }
        m = resultados['sts_es']['umbrales'][nombre]['test']
        print(f"  {nombre:<15} tau={tau:.4f} | test AUC={m['roc_auc']:.4f} F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f}")
    print(f"  Spearman(cos, STS) en test = {resultados['sts_es']['spearman_test']:.4f}")

    # ------------------------------------------------------------------
    # 2. VALIDACIÓN EXTERNA CON PAWS-X ES
    # ------------------------------------------------------------------
    print("\n[2/4] Validación externa con PAWS-X en español...")
    paws = {}
    for particion in ['validation', 'test']:
        df = pd.read_parquet(cfg.REDUNDANCIA_BENCH_DIR / f"pawsx_es_{particion}.parquet")
        df['cos'] = coseno_pares(modelo, df.sentence1, df.sentence2)
        paws[particion] = df
    tau_paws = umbral_youden(paws['validation'].label.values, paws['validation'].cos.values)
    resultados['pawsx_es'] = {
        'n_validation': len(paws['validation']), 'n_test': len(paws['test']),
        'tau_youden_validation': tau_paws,
        'test_con_tau_youden': metricas_binarias(paws['test'].label.values, paws['test'].cos.values, tau_paws),
        'test_con_tau_redundante_sts': metricas_binarias(paws['test'].label.values, paws['test'].cos.values, umbrales['tau_redundante']),
        'nota': 'PAWS-X contiene pares adversariales con alto solapamiento léxico; un coseno de embeddings de oración separa mal paráfrasis de no paráfrasis con las mismas palabras.',
    }
    m = resultados['pawsx_es']['test_con_tau_youden']
    print(f"  tau (Youden en validation) = {tau_paws:.4f} | test AUC={m['roc_auc']:.4f} F1={m['f1']:.4f} Acc={m['accuracy']:.4f}")

    # ------------------------------------------------------------------
    # 3. CORPUS BASE (2.604 noticias)
    # ------------------------------------------------------------------
    print("\n[3/4] Redundancia intra-documental en el corpus base...")
    corpus = pd.read_excel(cfg.CORPUS_BASE_XLSX)
    corpus['label_num'] = corpus['class'].apply(cfg.etiqueta_binaria)
    assert (corpus['label_num'] == corpus['label_num'].astype(int)).all()
    oraciones = [segmentar(t) for t in corpus.Text]
    planas = [o for doc in oraciones for o in doc]
    t0 = time.time()
    emb = modelo.encode(planas, batch_size=BATCH, normalize_embeddings=True, show_progress_bar=True)
    t_enc = time.time() - t0
    print(f"  {len(planas)} oraciones codificadas en {t_enc / 60:.1f} min")

    filas, pos = [], 0
    for doc, texto in zip(oraciones, corpus.Text):
        k = len(doc)
        E = emb[pos:pos + k]
        pos += k
        fila = {'num_sentences': k, 'num_pairs': k * (k - 1) // 2}
        if k >= 2:
            S = E @ E.T
            iu = np.triu_indices(k, 1)
            sims = np.sort(S[iu])[::-1]
            consec = np.array([S[i, i + 1] for i in range(k - 1)])
            fila.update({
                'max_intra_similarity': float(sims[0]),
                'second_max_intra_similarity': float(sims[1]) if len(sims) > 1 else np.nan,
                'top3_mean_intra_similarity': float(sims[:3].mean()),
                'mean_intra_similarity': float(sims.mean()),
                'p90_intra_similarity': float(np.percentile(sims, 90)),
                'var_intra_similarity': float(sims.var()),
                'consec_sim_mean': float(consec.mean()),
                'consec_sim_min': float(consec.min()),
                'consec_sim_std': float(consec.std()),
            })
            for nombre, tau in umbrales.items():
                n_sobre = int((sims >= tau).sum())
                fila[f'pares_sobre_{nombre}'] = n_sobre
                fila[f'densidad_{nombre}'] = n_sobre / len(sims)
            fila['pairwise_similarities'] = json.dumps([round(float(x), 6) for x in S[iu]])
        filas.append(fila)
    red = pd.concat([corpus[['class', 'label_num', 'Text', 'Fuente']].reset_index(drop=True), pd.DataFrame(filas)], axis=1)
    red['texto_normalizado'] = red.Text.str.lower().str.split().str.join(' ')
    red['is_unique_text'] = ~red.duplicated('texto_normalizado')
    red = red.drop(columns='texto_normalizado')
    red.to_csv(cfg.REDUNDANCIA_CORPUS_CSV, index=False)

    validos = red[red.num_sentences >= 2]
    unicos = validos[validos.is_unique_text]
    resultados['corpus'] = {
        'n_noticias': len(red), 'n_con_2_o_mas_oraciones': len(validos), 'n_unicos_validos': len(unicos),
        'n_oraciones': int(red.num_sentences.sum()), 'n_oraciones_en_validos': int(validos.num_sentences.sum()),
        'n_pares': int(validos.num_pairs.sum()), 'tiempo_codificacion_s': t_enc,
        'pct_pares_sobre_umbral': {n: float((np.concatenate([json.loads(s) for s in validos.pairwise_similarities]) >= t).mean())
                                   for n, t in umbrales.items()},
        'pct_noticias_con_algun_par_sobre_umbral': {n: float((validos[f'pares_sobre_{n}'] > 0).mean()) for n in umbrales},
    }
    print(f"  Noticias con >=2 oraciones: {len(validos)} | únicas: {len(unicos)} | pares: {resultados['corpus']['n_pares']}")
    for n in umbrales:
        print(f"  {n:<15} pares sobre el umbral: {resultados['corpus']['pct_pares_sobre_umbral'][n]*100:5.1f} % | "
              f"noticias con algún par: {resultados['corpus']['pct_noticias_con_algun_par_sobre_umbral'][n]*100:5.1f} %")

    # ------------------------------------------------------------------
    # 4. ASOCIACIÓN DESCRIPTIVA CON LA VERACIDAD (textos únicos)
    # ------------------------------------------------------------------
    print("\n[4/4] Asociación con la veracidad (Mann-Whitney, textos únicos)...")
    variables = ['max_intra_similarity', 'second_max_intra_similarity', 'mean_intra_similarity',
                 'consec_sim_mean', 'densidad_tau_parafrasis', 'densidad_tau_redundante']
    asociacion = {}
    for v in variables:
        a = unicos.loc[unicos.label_num == cfg.FAKE, v].dropna().values
        b = unicos.loc[unicos.label_num == cfg.REAL, v].dropna().values
        u = mannwhitneyu(a, b, alternative='two-sided')
        asociacion[v] = {
            'mediana_falsas': float(np.median(a)), 'mediana_reales': float(np.median(b)),
            'n_falsas': len(a), 'n_reales': len(b),
            'U': float(u.statistic), 'p_valor': float(u.pvalue),
            # Positiva si las falsas tienden a valores mayores (U de scipy corresponde a la primera muestra)
            'correlacion_biserial_rangos': float(2 * u.statistic / (len(a) * len(b)) - 1),
            'auc_univariado_falsa': float(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b])),
            'ic95_diferencia_medianas': bootstrap_diferencia_medianas(a, b),
        }
        print(f"  {v:<30} med F={asociacion[v]['mediana_falsas']:.4f} R={asociacion[v]['mediana_reales']:.4f} "
              f"AUC={asociacion[v]['auc_univariado_falsa']:.3f} p={u.pvalue:.2e}")
    # Corrección de Holm para la familia de pruebas
    orden = sorted(asociacion, key=lambda k: asociacion[k]['p_valor'])
    m = len(orden)
    p_ajustado_prev = 0.0
    for i, k in enumerate(orden):
        p_aj = min(1.0, max(p_ajustado_prev, (m - i) * asociacion[k]['p_valor']))
        asociacion[k]['p_holm'] = p_aj
        p_ajustado_prev = p_aj
    resultados['asociacion_veracidad_textos_unicos'] = asociacion
    resultados['metadata']['tiempo_total_s'] = time.time() - t_inicio

    with open(cfg.METRICAS_REDUNDANCIA_JSON, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] {cfg.REDUNDANCIA_CORPUS_CSV}\n[OK] {cfg.METRICAS_REDUNDANCIA_JSON}")
    print(f"Tiempo total: {resultados['metadata']['tiempo_total_s'] / 60:.1f} min")


if __name__ == '__main__':
    main()
