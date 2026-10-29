"""
Cálculo de las 43 variables de la tabla única (S, R, E, O, L) para UNA noticia nueva.

Reproduce exactamente la extracción usada sobre el corpus (construir_tabla_variables.py, que a su vez reutiliza
extraer_features_modelo_sobre_variables_ampliado.py y extraer_features_avanzadas_5d.py) para que el IML y los
estilos de un texto nuevo sean comparables con los del corpus. La equivalencia se comprueba con
verificar_contra_tabla() sobre noticias del corpus.

Además de las variables, devuelve el detalle por oración (sensacionalismo, emociones, similitudes) que usa el
auditor para explicar en qué oraciones se concentra cada rasgo.
"""

import os
import re
import sys
import unicodedata
import importlib.util
from pathlib import Path

import numpy as np

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config_proyecto.py").exists())))
import config_proyecto as cfg

EMOCIONES = ['others', 'joy', 'sadness', 'anger', 'surprise', 'disgust', 'fear']
EMOCIONES_USADAS = {'anger': 'ira', 'fear': 'miedo', 'sadness': 'tristeza', 'joy': 'alegria', 'surprise': 'sorpresa'}

# --- Patrones de extraer_features_avanzadas_5d.py (copiados sin cambios) ---
PAT_HEDGES = re.compile(
    r'\b(presunto|presunta|presuntos|presuntas|presuntamente|al parecer|según fuentes|'
    r'podría|podrían|se presume|supuesto|supuesta|supuestos|supuestamente|aparentemente|'
    r'probablemente|posiblemente|se sospecha|se infiere|de confirmarse|no se descarta|'
    r'trascendió que|habría sido|habrían sido)\b', re.IGNORECASE)
PAT_BOOSTERS = re.compile(
    r'\b(indudablemente|obviamente|claramente|sin duda|sin lugar a dudas|es un hecho|'
    r'irrefutable|totalmente|absolutamente|jamás|nunca jamás|la verdad oculta|revelado|'
    r'comprobado|definitivamente|rotundamente|innegable|a ciencia cierta|categóricamente|'
    r'escándalo total|evidencia irrefutable|científicamente comprobado)\b', re.IGNORECASE)
PAT_DICENDI = re.compile(
    r'\b(dijo|declaró|manifestó|indicó|explicó|informó|expresó|afirmó|sostuvo|aseveró|'
    r'aseguró|añadió|puntualizó|precisó|concluyó|señaló|comentó|declararon|informaron)\b', re.IGNORECASE)
PAT_QUOTES = re.compile(r'["«»“”]')
PAT_ALL_CAPS = re.compile(r'\b[A-ZÁÉÍÓÚÑ]{3,}\b')
PAT_CENSORED_MASKS = re.compile(r'\*(?:NUMBER|URL|PHONE|\d+)\*', re.IGNORECASE)
PAT_CENSORED_NUMBER = re.compile(r'\*(?:NUMBER|\d+)\*', re.IGNORECASE)
PAT_EXCLAMATION = re.compile(r'[!¡]')
PAT_QUESTION = re.compile(r'[?¿]')
PAT_ELLIPSIS = re.compile(r'(\.\.\.|…)')
PAT_DIGITS = re.compile(r'\b\d+(?:[\.,]\d+)?\b')
PAT_PERCENT = re.compile(r'(?:\d+\s*%|\bpor\s+ciento\b)', re.IGNORECASE)
PAT_TEMPORAL = re.compile(r'\b(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre|2019|2020|2021|2022|2023|2024|2025|2026)\b', re.IGNORECASE)
SIGLAS_EXCLUIDAS = {'ONU', 'UE', 'EEUU', 'OMS', 'PIB', 'FMI', 'OTAN', 'VOX', 'PSOE', 'PP', 'INE',
                    'USA', 'EFE', 'IAC', 'NICA', 'NUMBER', 'URL', 'PHONE'}
PRON_1P = {'yo', 'nosotros', 'nosotras', 'me', 'nos', 'mi', 'mis', 'mío', 'míos', 'mía', 'mías', 'nuestro', 'nuestra', 'nuestros', 'nuestras'}
PRON_3P = {'él', 'ella', 'ello', 'ellos', 'ellas', 'se', 'le', 'les', 'lo', 'los', 'la', 'las', 'su', 'sus', 'suyo', 'suya', 'suyos', 'suyas'}


def segmentar(texto):
    import nltk
    sents = [s.strip() for s in nltk.sent_tokenize(str(texto), language='spanish') if len(s.strip()) > 8]
    return sents if sents else [str(texto).strip()[:200]]


def _silabas(word):
    word, vocales, n, en_vocal = word.lower(), "aeiouáéíóúü", 0, False
    for ch in word:
        if ch in vocales:
            if not en_vocal:
                n += 1
                en_vocal = True
        else:
            en_vocal = False
    return max(1, n)


def variables_superficiales(texto, n_oraciones, nlp):
    """Grupos O y L (regex, legibilidad, spaCy), con las mismas fórmulas que extraer_features_avanzadas_5d.py."""
    t = str(texto)
    n_palabras = len(t.split())
    w, s = max(n_palabras, 1), max(n_oraciones, 1)
    sin_mascaras = PAT_CENSORED_MASKS.sub(' ', t)
    v = {
        'ling_atenuadores': len(PAT_HEDGES.findall(t)) / w * 100, 'ling_intensificadores': len(PAT_BOOSTERS.findall(t)) / w * 100,
        'orto_comillas': len(PAT_QUOTES.findall(t)) / s, 'ling_verbos_dicendi': len(PAT_DICENDI.findall(t)) / w * 100,
    }
    tokens = re.sub(r'[^\w\s]', '', t.lower()).split()
    wt = max(len(tokens), 1)
    v['ling_flesch_szigriszt'] = float(np.clip(206.835 - 62.3 * (sum(_silabas(x) for x in tokens) / wt) - (wt / s), 0, 120))
    v['ling_guiraud'] = len(set(tokens)) / np.sqrt(wt)
    cuenta = {}
    for x in tokens:
        cuenta[x] = cuenta.get(x, 0) + 1
    v['ling_hapax'] = sum(1 for c in cuenta.values() if c == 1) / wt
    n_adj = n_noun = n_adv = n_p1 = n_p3 = 0
    for tok in nlp(t):
        low = tok.text.lower()
        n_adj += tok.pos_ == 'ADJ'
        n_noun += tok.pos_ == 'NOUN'
        n_adv += tok.pos_ == 'ADV'
        if tok.pos_ == 'PRON' or low in PRON_1P or low in PRON_3P:
            if low in PRON_1P:
                n_p1 += 1
            elif low in PRON_3P:
                n_p3 += 1
    v.update({'ling_adjetivos_sustantivos': n_adj / (n_noun + 1), 'ling_pronombres_1p': n_p1 / w * 100,
              'ling_pronombres_3p': n_p3 / w * 100, 'ling_adverbios': n_adv / w * 100})
    caps = [x for x in PAT_ALL_CAPS.findall(sin_mascaras) if x not in SIGLAS_EXCLUIDAS]
    letras = [c for c in sin_mascaras if c.isalpha()]
    n_excl, n_preg, n_susp = (len(p.findall(sin_mascaras)) for p in (PAT_EXCLAMATION, PAT_QUESTION, PAT_ELLIPSIS))
    v.update({
        'orto_mayusculas_sostenidas': len(caps) / w * 100,
        'orto_proporcion_mayusculas': sum(c.isupper() for c in letras) / max(len(letras), 1),
        'orto_exclamaciones': n_excl / w * 100, 'orto_interrogaciones': n_preg / w * 100, 'orto_puntos_suspensivos': n_susp / w * 100,
        'ling_cifras': (len(PAT_DIGITS.findall(sin_mascaras)) + len(PAT_CENSORED_NUMBER.findall(t))) / w * 100,
        'ling_porcentajes': len(PAT_PERCENT.findall(t)),
        'ling_referencias_temporales': len(PAT_TEMPORAL.findall(sin_mascaras)) / w * 100,
        'ling_palabras': n_palabras, 'ling_oraciones': n_oraciones,
    })
    detalle = {'mayusculas_sostenidas': caps, 'exclamaciones': n_excl, 'interrogaciones': n_preg, 'puntos_suspensivos': n_susp,
               'intensificadores': [m.group(0) for m in PAT_BOOSTERS.finditer(t)], 'atenuadores': [m.group(0) for m in PAT_HEDGES.finditer(t)],
               'verbos_atribucion': [m.group(0) for m in PAT_DICENDI.finditer(t)]}
    return v, detalle


class ExtractorVariables:
    """Carga los tres modelos (revisión fija) y spaCy una sola vez."""

    def __init__(self):
        import torch
        import spacy
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(os.cpu_count() or 4)
        # GPU si existe (XAI_DISPOSITIVO=cpu la desactiva)
        self.dev = os.environ.get('XAI_DISPOSITIVO') or ('cuda' if torch.cuda.is_available() else 'cpu')
        ms, mr, me = (cfg.MODELOS_HF[k] for k in ['Sensacionalismo_JJNeila', 'SBERT_Redundancia', 'Emociones_RoBERTuito'])
        self.idx_sens = ms['idx_positivo']
        self.tok_s = AutoTokenizer.from_pretrained(ms['id'], revision=ms['revision'])
        self.mod_s = AutoModelForSequenceClassification.from_pretrained(ms['id'], revision=ms['revision']).eval().to(self.dev)
        self.sbert = SentenceTransformer(mr['id'], revision=mr['revision'], device=self.dev)
        self.tok_e = AutoTokenizer.from_pretrained(me['id'], revision=me['revision'])
        self.mod_e = AutoModelForSequenceClassification.from_pretrained(me['id'], revision=me['revision']).eval().to(self.dev)
        id2 = {int(k): v.lower() for k, v in self.mod_e.config.id2label.items()}
        self.orden_e = [next(k for k, v in id2.items() if v == e) for e in EMOCIONES]
        self.max_len_e = me['max_len']
        spec = importlib.util.find_spec('pysentimiento')
        sp = importlib.util.spec_from_file_location('pysentimiento_preprocessing', Path(spec.submodule_search_locations[0]) / 'preprocessing.py')
        self.pre = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(self.pre)
        self.nlp = spacy.load('es_core_news_sm', disable=['ner', 'parser'])

    def _probs(self, tok, mod, textos, max_len):
        import torch
        with torch.inference_mode():
            enc = tok(textos, padding=True, truncation=True, max_length=max_len, return_tensors='pt').to(self.dev)
            return torch.softmax(mod(**enc).logits.float(), -1).cpu().numpy()

    def extraer(self, texto):
        # NFC: 1.252 textos (casi todos de Omdena) traen las tildes descompuestas (NFD) y los léxicos y tokenizadores no las reconocen
        texto = unicodedata.normalize('NFC', str(texto))
        oraciones = segmentar(texto)
        # S: sensacionalismo del documento (primeros 1.000 caracteres, 256 tokens) y por oración (128 tokens)
        p_doc = float(self._probs(self.tok_s, self.mod_s, [str(texto)[:1000]], 256)[0, self.idx_sens])
        p_or = self._probs(self.tok_s, self.mod_s, oraciones, 128)[:, self.idx_sens]
        v = {'sens_p_documento': p_doc, 'sens_p_media_oraciones': float(p_or.mean()), 'sens_p_max_oraciones': float(p_or.max()),
             'sens_sd_oraciones': float(p_or.std()) if len(p_or) > 1 else 0.0}
        # R: similitud entre oraciones (sin umbrales)
        sim = None
        if len(oraciones) >= 2:
            e = self.sbert.encode(oraciones, normalize_embeddings=True)
            sim = e @ e.T
            pares = np.sort(sim[np.triu_indices(len(e), 1)])[::-1]
            consec = np.array([sim[i, i + 1] for i in range(len(e) - 1)])
            v.update({'red_sim_max': float(pares[0]), 'red_sim_media': float(pares.mean()), 'red_sim_top3': float(pares[:3].mean()),
                      'red_sim_varianza': float(pares.var()), 'red_consecutivas_media': float(consec.mean()), 'red_consecutivas_min': float(consec.min())})
        else:
            v.update({k: np.nan for k in ['red_sim_max', 'red_sim_media', 'red_sim_top3', 'red_sim_varianza', 'red_consecutivas_media', 'red_consecutivas_min']})
        # E: emociones por oración, agregadas
        q = self._probs(self.tok_e, self.mod_e, [self.pre.preprocess_tweet(o, lang='es') for o in oraciones], self.max_len_e)[:, self.orden_e]
        inten = 1 - q[:, 0]
        v.update({'emo_intensidad_media': float(inten.mean()), 'emo_intensidad_max': float(inten.max())})
        for en, es in EMOCIONES_USADAS.items():
            j = EMOCIONES.index(en)
            v[f'emo_{es}_media'] = float(q[:, j].mean())
            v[f'emo_{es}_max'] = float(q[:, j].max())
        # O y L
        sup, detalle = variables_superficiales(texto, len(oraciones), self.nlp)
        v.update(sup)
        return v, {'oraciones': oraciones, 'sens_oraciones': p_or.tolist(), 'emociones_oraciones': q.tolist(),
                   'similitud': None if sim is None else sim.tolist(), 'marcadores': detalle}


def verificar_contra_tabla(n=30):
    """Compara las variables recalculadas con las de la tabla en n noticias del corpus."""
    import pandas as pd
    tabla = pd.read_csv(cfg.TABLA_VARIABLES_CSV)
    idx = np.random.default_rng(cfg.SEED).choice(len(tabla), n, replace=False)
    ex = ExtractorVariables()
    dif = {}
    for i in idx:
        v, _ = ex.extraer(tabla['Text'].iloc[i])
        for k, x in v.items():
            ref = tabla[k].iloc[i]
            if not (np.isnan(x) and np.isnan(ref)):
                dif[k] = max(dif.get(k, 0.0), abs(x - ref))
    return dif


if __name__ == '__main__':
    d = verificar_contra_tabla()
    peor = sorted(d.items(), key=lambda x: -x[1])
    print("Diferencia absoluta máxima por variable (30 noticias):")
    for k, x in peor:
        print(f"  {k:<32} {x:.2e}")
