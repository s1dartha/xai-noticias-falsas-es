"""
Objetivos a explicar, métodos de explicabilidad y evaluación de fidelidad (Fase 6).

Objetivos (qué número se explica)
---------------------------------
  jjneila            P(sensacionalista) de JJNeila/bert-spanish-sensationalism-oss
  sabert             P(falsa) de VerificadoProfesional/SaBERT-Spanish-Fake-News
  redundancia        similitud coseno media entre las oraciones del documento (hiiamsid)
  modelo_variables   P(falsa) del Modelo sobre Variables (43 variables -> regresión logística)
En los clasificadores se explica la probabilidad de la clase PREDICHA para el documento completo (como en ERASER,
DeYoung et al., 2020).

Métodos
-------
  Nivel palabra (granularidad nativa de los métodos de la literatura):
    IG             Integrated Gradients (Sundararajan et al., 2017), 32 pasos, referencia [PAD]
    IxG            Input × Gradient (Shrikumar et al., 2017)
    Rollout        Attention Rollout (Abnar y Zuidema, 2020)
    LRP-Chefer     relevancia atención × gradiente (Chefer, Gur y Wolf, 2021)
    LIME-palabra   LIME (Ribeiro et al., 2016), librería lime, 200 perturbaciones
    SHAP-palabra   SHAP Partition (Lundberg y Lee, 2017), librería shap, 300 evaluaciones
  Nivel oración:
    MexGen         explicabilidad propia: ablación de una oración por vez (Δ = f(texto) − f(texto sin la oración))
    LIME-oración   variante nativa de LIME con las oraciones como rasgos
    SHAP-oración   variante nativa de SHAP con las oraciones como unidades de la máscara
    MexGen-cláusulas  versión anterior de MexGen (corte en comas, punto y coma, dos puntos…), solo para justificar la segmentación
  Agregación (adaptación, no nativa): los métodos de palabra se suman por oración ("IG→oración", etc.).

Fidelidad
---------
Las unidades se BORRAN del texto (no se reemplazan por [MASK], que produce entradas fuera de distribución). Para comparar
métodos de distinta granularidad, la curva de borrado se mide en fracción de palabras eliminadas q ∈ {0,1 … 0,5}; en los
métodos por oración el valor en q se interpola entre el número de oraciones que queda justo por debajo y por encima.
  AOPC (borrado)   media de f(texto) − f(texto sin las unidades más importantes) sobre la rejilla de q
  Ganancia         AOPC − AOPC del orden aleatorio de la MISMA granularidad (controla que borrar oraciones completas
                   y borrar palabras sueltas alteran el texto de forma distinta)
  Suficiencia@20 % f(texto) − f(solo el 20 % más importante); menor es mejor
  Cambio de clase@20 %  si la clase predicha cambia al borrar el 20 % más importante
"""

import os
import re
import sys
import time
import json
import copy
from pathlib import Path

import numpy as np

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config_proyecto.py").exists())))
import config_proyecto as cfg

REJILLA_Q = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
Q_SUF = 0.2
N_PASOS_IG = 32
N_LIME_PALABRA, N_SHAP_PALABRA = 200, 300
N_LIME_ORACION = 50
N_ALEATORIOS = 3


def dispositivo():
    """GPU si existe; XAI_DISPOSITIVO=cpu la desactiva."""
    import torch
    return os.environ.get('XAI_DISPOSITIVO') or ('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# SEGMENTACIÓN
# =============================================================================
def spans_palabras(texto):
    return [(m.start(), m.end()) for m in re.finditer(r'\S+', texto)]


def spans_oraciones(texto):
    """Oraciones (punkt en español: cierra en punto, signo de exclamación o de interrogación)."""
    import nltk
    out, cur = [], 0
    for s in nltk.sent_tokenize(texto, language='spanish'):
        i = texto.find(s, cur)
        if i < 0:
            continue
        out.append((i, i + len(s)))
        cur = i + len(s)
    return out or [(0, len(texto))]


def spans_clausulas(texto):
    """Segmentación de la versión anterior de MexGen: corta también en comas, punto y coma, dos puntos, comillas y paréntesis."""
    out, ini = [], 0
    for m in re.finditer(r'[,;:.!?¡¿\n—–"«»()]+', texto):
        if texto[ini:m.start()].strip():
            out.append((ini, m.end()))
        ini = m.end()
    if texto[ini:].strip():
        out.append((ini, len(texto)))
    return out or [(0, len(texto))]


def texto_sin(texto, spans, quitar):
    quitar = set(quitar)
    partes = [texto[a:b] for i, (a, b) in enumerate(spans) if i not in quitar]
    return re.sub(r'\s+', ' ', ' '.join(partes)).strip()


CIERRE = re.compile(r'[.!?…]+["»”)]*$')


def texto_sin_conservando_cierre(texto, spans, quitar):
    """Borra palabras pero conserva su puntuación de cierre de oración (evita fusionar oraciones al borrar)."""
    quitar = set(quitar)
    partes = []
    for i, (a, b) in enumerate(spans):
        if i in quitar:
            m = CIERRE.search(texto[a:b])
            if m:
                partes.append(m.group(0))
        else:
            partes.append(texto[a:b])
    return re.sub(r'\s+', ' ', ' '.join(partes)).strip()


def evaluar_palabra_conservando_cierre(obj, texto, spans, attr, base):
    """Curva de borrado a nivel palabra sin destruir los límites de oración."""
    orden = np.argsort(-attr, kind='stable')
    W = len(spans)
    en_q = base - obj.f([texto_sin_conservando_cierre(texto, spans, orden[:int(np.ceil(q * W))]) for q in REJILLA_Q])
    return {'curva': [float(x) for x in en_q], 'aopc': float(np.mean(en_q))}


def proporcion_cierre_en_top(texto, spans, attr, q=0.1):
    """Proporción de palabras con puntuación de cierre entre el q más importante, y en todo el texto."""
    con = np.array([bool(CIERRE.search(texto[a:b])) for a, b in spans])
    top = np.argsort(-attr, kind='stable')[:max(1, int(np.ceil(q * len(spans))))]
    return float(con[top].mean()), float(con.mean())


def texto_solo(texto, spans, conservar):
    conservar = set(conservar)
    return re.sub(r'\s+', ' ', ' '.join(texto[a:b] for i, (a, b) in enumerate(spans) if i in conservar)).strip()


def palabras_por_unidad(spans_unidad, spans_pal):
    """Número de palabras de cada unidad (oración o cláusula)."""
    return np.array([sum(1 for (p, q) in spans_pal if p >= a and q <= b) or 1 for (a, b) in spans_unidad])


def agregar_a_unidades(attr_pal, spans_pal, spans_unidad):
    out = np.zeros(len(spans_unidad))
    for w, (p, q) in zip(attr_pal, spans_pal):
        for j, (a, b) in enumerate(spans_unidad):
            if p >= a and p < b:
                out[j] += w
                break
    return out


# =============================================================================
# OBJETIVOS
# =============================================================================
class Clasificador:
    tiene_gradientes, tiene_atencion = True, True

    def __init__(self, clave, batch=16, max_len=256):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        m = cfg.MODELOS_HF[{'jjneila': 'Sensacionalismo_JJNeila', 'sabert': 'M2_SaBERT'}[clave]]
        self.clave, self.batch, self.max_len, self.idx_pos = clave, batch, max_len, m['idx_positivo']
        self.dev = dispositivo()
        self.tok = AutoTokenizer.from_pretrained(m['id'], revision=m['revision'])
        self.model = AutoModelForSequenceClassification.from_pretrained(m['id'], revision=m['revision'], attn_implementation='eager').eval().to(self.dev)
        self.target = self.idx_pos
        self.torch = torch

    def fijar_objetivo(self, texto):
        """Explica la clase predicha para el texto completo."""
        p = self.probs([texto])[0]
        self.target = int(p.argmax())
        return self.target

    def probs(self, textos):
        torch = self.torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(textos), self.batch):
                enc = self.tok([t if t.strip() else '.' for t in textos[i:i + self.batch]], padding=True, truncation=True,
                               max_length=self.max_len, return_tensors='pt').to(self.dev)
                out.append(torch.softmax(self.model(**enc).logits.float(), -1).cpu().numpy())
        return np.concatenate(out)

    def f(self, textos):
        return self.probs(list(textos))[:, self.target]

    def predice_positiva(self, textos):
        return self.probs(list(textos)).argmax(1) == self.idx_pos

    # ---- métodos de gradiente y atención (por token) ----
    def _codificar(self, texto):
        enc = self.tok(texto, truncation=True, max_length=self.max_len, return_offsets_mapping=True, return_tensors='pt')
        off = enc.pop('offset_mapping')[0].numpy()
        return enc.to(self.dev), off

    def ig_ixg(self, texto, modelo=None):
        torch = self.torch
        model = modelo or self.model
        enc, off = self._codificar(texto)
        ids = enc['input_ids']
        emb_layer = model.get_input_embeddings()
        with torch.no_grad():
            x = emb_layer(ids)
            base_ids = ids.clone()
            esp = torch.tensor([i in self.tok.all_special_ids for i in ids[0].tolist()], device=ids.device)
            base_ids[0, ~esp] = self.tok.pad_token_id
            x0 = emb_layer(base_ids)
        grads = torch.zeros_like(x)
        alphas = torch.linspace(1.0 / N_PASOS_IG, 1.0, N_PASOS_IG, device=x.device)
        for k in range(0, N_PASOS_IG, 8):
            a = alphas[k:k + 8].view(-1, 1, 1)
            xs = (x0 + a * (x - x0)).detach().requires_grad_(True)
            out = model(inputs_embeds=xs, attention_mask=enc['attention_mask'].expand(len(a), -1))
            torch.softmax(out.logits, -1)[:, self.target].sum().backward()
            grads += xs.grad.sum(0, keepdim=True)
        ig = ((x - x0) * grads / N_PASOS_IG).sum(-1)[0].detach().cpu().numpy()
        xr = x.detach().clone().requires_grad_(True)
        out = model(inputs_embeds=xr, attention_mask=enc['attention_mask'])
        torch.softmax(out.logits, -1)[0, self.target].backward()
        ixg = (xr.grad * xr).sum(-1)[0].detach().cpu().numpy()
        return {'IG': ig, 'IxG': ixg}, off

    def rollout_chefer(self, texto, modelo=None):
        torch = self.torch
        model = modelo or self.model
        enc, off = self._codificar(texto)
        out = model(**enc, output_attentions=True)
        atts = out.attentions
        for a in atts:
            a.retain_grad()
        torch.softmax(out.logits, -1)[0, self.target].backward()
        n = atts[0].shape[-1]
        eye = torch.eye(n, device=atts[0].device)
        roll, R = eye.clone(), eye.clone()
        for a in atts:
            A = a[0].detach().mean(0)
            A = 0.5 * A + 0.5 * eye
            A = A / A.sum(-1, keepdim=True)
            roll = A @ roll
            cam = (a.grad[0] * a[0]).clamp(min=0).mean(0).detach()
            R = R + cam @ R
        model.zero_grad()
        return {'Rollout': roll[0].cpu().numpy(), 'LRP-Chefer': R[0].cpu().numpy()}, off

    def a_palabras(self, attr_tok, off, texto):
        sp = spans_palabras(texto)
        out = np.zeros(len(sp))
        for v, (a, b) in zip(attr_tok, off):
            if b <= a:
                continue   # tokens especiales
            for j, (p, q) in enumerate(sp):
                if a >= p and a < q:
                    out[j] += v
                    break
        return out

    def copia_aleatorizada(self, n_capas=3):
        """Prueba de Adebayo et al. (2018): reinicia la cabeza y las n capas superiores del codificador."""
        m = copy.deepcopy(self.model)
        torch = self.torch
        torch.manual_seed(cfg.SEED)
        capas = m.base_model.encoder.layer
        for capa in list(capas)[-n_capas:]:
            capa.apply(m._init_weights)
        m.classifier.apply(m._init_weights)
        return m.eval()


class Redundancia:
    """Similitud coseno media entre las oraciones del documento (hiiamsid)."""
    tiene_gradientes, tiene_atencion = True, False

    def __init__(self):
        import torch
        from sentence_transformers import SentenceTransformer
        m = cfg.MODELOS_HF['SBERT_Redundancia']
        self.dev = dispositivo()
        self.sbert = SentenceTransformer(m['id'], revision=m['revision'], device=self.dev)
        self.bert = self.sbert[0].model   # módulo real (auto_model es una propiedad de solo lectura)
        self.tok = self.sbert.tokenizer
        self.torch = torch
        self.clave = 'redundancia'

    def fijar_objetivo(self, texto):
        return None

    def _valor(self, texto):
        import nltk
        s = [x for x in nltk.sent_tokenize(texto, language='spanish') if x.strip()]
        if len(s) < 2:
            return 0.0
        e = self.sbert.encode(s, normalize_embeddings=True)
        m = e @ e.T
        return float(m[np.triu_indices(len(s), 1)].mean())

    def f(self, textos):
        return np.array([self._valor(t) for t in textos])

    def predice_positiva(self, textos):
        return None

    def ig_ixg(self, texto, modelo=None):
        """IG e IxG sobre los embeddings de entrada de TODAS las oraciones a la vez (el objetivo depende de todas)."""
        torch = self.torch
        bert = modelo or self.bert
        sp = spans_oraciones(texto)
        sents = [texto[a:b] for a, b in sp]
        enc = self.tok(sents, padding=True, truncation=True, max_length=128, return_offsets_mapping=True, return_tensors='pt')
        off = enc.pop('offset_mapping').numpy()
        enc = enc.to(self.dev)
        ids, mask = enc['input_ids'], enc['attention_mask']
        emb = bert.get_input_embeddings()
        with torch.no_grad():
            x = emb(ids)
            base = ids.clone()
            esp = torch.isin(ids, torch.tensor(self.tok.all_special_ids, device=ids.device))
            base[~esp] = self.tok.pad_token_id
            x0 = emb(base)

        def objetivo(xs):
            h = bert(inputs_embeds=xs, attention_mask=mask).last_hidden_state
            v = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
            v = torch.nn.functional.normalize(v, dim=-1)
            s = v @ v.T
            iu = torch.triu_indices(len(sents), len(sents), 1, device=s.device)
            return s[iu[0], iu[1]].mean()

        grads = torch.zeros_like(x)
        for a in torch.linspace(1.0 / N_PASOS_IG, 1.0, N_PASOS_IG):
            xs = (x0 + a * (x - x0)).detach().requires_grad_(True)
            objetivo(xs).backward()
            grads += xs.grad
        ig = ((x - x0) * grads / N_PASOS_IG).sum(-1).detach().cpu().numpy()
        xr = x.detach().clone().requires_grad_(True)
        objetivo(xr).backward()
        ixg = (xr.grad * xr).sum(-1).detach().cpu().numpy()
        # Offsets de cada oración -> offsets en el texto completo
        offs = [(a0 + o[0], a0 + o[1]) if o[1] > o[0] else (0, 0) for (a0, _), fila in zip(sp, off) for o in fila]
        return {'IG': ig.reshape(-1), 'IxG': ixg.reshape(-1)}, np.array(offs)

    a_palabras = Clasificador.a_palabras

    def copia_aleatorizada(self, n_capas=3):
        m = copy.deepcopy(self.bert)
        self.torch.manual_seed(cfg.SEED)
        for capa in list(m.encoder.layer)[-n_capas:]:
            capa.apply(m._init_weights)
        return m.eval()

    def f_con_modelo(self, modelo, textos):
        original = self.sbert[0].model
        self.sbert[0].model = modelo
        try:
            return self.f(textos)
        finally:
            self.sbert[0].model = original


class ModeloVariables:
    """P(falsa) del Modelo sobre Variables: texto -> 43 variables -> regresión logística (no diferenciable)."""
    tiene_gradientes, tiene_atencion = False, False

    def __init__(self):
        import pandas as pd
        sys.path.insert(0, str(cfg.EXPLICABILIDAD_CODIGOS_DIR))
        sys.path.insert(0, str(cfg.MODELO_VARIABLES_CODIGOS_DIR))
        from variables_texto import ExtractorVariables
        import evaluar_modelo_sobre_variables_sin_fuga as ev
        meta = json.loads(cfg.TABLA_VARIABLES_META_JSON.read_text(encoding='utf-8'))
        self.vars = [v for g in ['S', 'R', 'E', 'O', 'L'] for v in meta['grupos'][g]]
        df = pd.read_csv(cfg.TABLA_VARIABLES_CSV).drop_duplicates('id_texto')
        self.lr = ev.modelo('lr').fit(df[self.vars].values.astype(float), df['label_num'].values)
        self.ex = ExtractorVariables()
        self.target = 1
        self.clave = 'modelo_variables'
        self._cache = {}

    def _p(self, t):
        if t not in self._cache:
            v, _ = self.ex.extraer(t if t.strip() else '.')
            self._cache[t] = self.lr.predict_proba(np.array([[v[k] for k in self.vars]], dtype=float))[0]
        return self._cache[t]

    def fijar_objetivo(self, texto):
        self._cache = {}
        self.target = int(self._p(texto).argmax())
        return self.target

    def f(self, textos):
        return np.array([self._p(t)[self.target] for t in textos])

    def predice_positiva(self, textos):
        return np.array([self._p(t).argmax() == 1 for t in textos])


# =============================================================================
# MÉTODOS DE PERTURBACIÓN (LIME, SHAP, MexGen)
# =============================================================================
def _clf_fn(obj):
    def fn(textos):
        p = np.clip(obj.f(list(textos)), 0, 1)
        return np.stack([1 - p, p], 1)
    return fn


def lime_unidades(obj, texto, spans, n_muestras):
    from lime.lime_text import LimeTextExplainer
    unidades = [texto[a:b] for a, b in spans]
    exp = LimeTextExplainer(split_expression=lambda s: unidades, bow=False, mask_string='', random_state=cfg.SEED)
    e = exp.explain_instance(texto, _clf_fn(obj), num_features=len(unidades), num_samples=n_muestras, labels=(1,))
    pesos = dict(e.as_map()[1])
    return np.array([pesos.get(i, 0.0) for i in range(len(unidades))])


def shap_unidades(obj, texto, spans, max_evals):
    import shap

    # SHAP necesita un tokenizador que devuelva unidades y posiciones para cualquier cadena (también la vacía);
    # para el texto explicado se usan exactamente las unidades dadas.
    def tokenizador(s, return_offsets_mapping=True):
        sp = list(spans) if s == texto else ([] if not s.strip() else
                                             (spans_palabras(s) if len(spans) and spans == spans_palabras(texto) else spans_oraciones(s)))
        return {'input_ids': list(range(1, len(sp) + 1)), 'offset_mapping': sp}

    # mask_token numérico para que shap no intente tokenizar la máscara; luego se vacía para que la unidad se BORRE
    # (por defecto shap la reemplaza por "...", que el modelo leería como texto)
    masker = shap.maskers.Text(tokenizador, mask_token=0, collapse_mask_token=True)
    masker.mask_token = ''
    explainer = shap.Explainer(lambda xs: obj.f([str(x) for x in xs]), masker, algorithm='partition', seed=cfg.SEED)
    sv = explainer([texto], max_evals=max_evals, silent=True)
    return np.array(sv.values[0], dtype=float)


def mexgen(obj, texto, spans):
    """Explicabilidad propia: Δ_i = f(texto) − f(texto sin la unidad i)."""
    base = obj.f([texto])[0]
    sin = obj.f([texto_sin(texto, spans, [i]) for i in range(len(spans))])
    return base - sin


# =============================================================================
# EVALUACIÓN DE FIDELIDAD
# =============================================================================
def clase_cambia(obj, texto_modificado):
    """True si la clase predicha deja de ser la explicada (solo clasificadores)."""
    if obj.clave == 'redundancia':
        return None
    if obj.clave == 'modelo_variables':
        return bool(obj._p(texto_modificado).argmax() != obj.target)
    return bool(obj.probs([texto_modificado])[0].argmax() != obj.target)


def evaluar(obj, texto, spans, attr, base, granularidad, rng):
    orden = np.argsort(-attr, kind='stable')
    if granularidad == 'palabra':
        # Se evalúa exactamente en la rejilla: se borran ceil(q·W) palabras
        W = len(spans)
        ms = [int(np.ceil(q * W)) for q in REJILLA_Q]
        en_q = base - obj.f([texto_sin(texto, spans, orden[:m]) for m in ms])
        m20 = int(np.ceil(Q_SUF * W))
    else:
        # Por oración: se borran 1, 2, … oraciones hasta superar el 50 % de las palabras y se interpola en la rejilla
        pal = palabras_por_unidad(spans, spans_palabras(texto))
        cum = np.cumsum(pal[orden]) / pal.sum()
        m_max = int(min(len(orden), np.searchsorted(cum, REJILLA_Q[-1]) + 1))
        vals = base - obj.f([texto_sin(texto, spans, orden[:m]) for m in range(1, m_max + 1)])
        en_q = np.interp(REJILLA_Q, np.concatenate([[0.0], cum[:m_max]]), np.concatenate([[0.0], vals]))
        m20 = int(min(len(orden), np.searchsorted(cum, Q_SUF) + 1))
    suf = base - obj.f([texto_solo(texto, spans, orden[:m20])])[0]
    return {'curva': [float(x) for x in en_q], 'aopc': float(np.mean(en_q)), 'suficiencia_20': float(suf),
            'cambio_clase_20': clase_cambia(obj, texto_sin(texto, spans, orden[:m20]))}


def evaluar_aleatorio(obj, texto, spans, base, granularidad, rng):
    res = [evaluar(obj, texto, spans, rng.random(len(spans)), base, granularidad, rng) for _ in range(N_ALEATORIOS)]
    cc = [r['cambio_clase_20'] for r in res]
    return {'curva': np.mean([r['curva'] for r in res], 0).tolist(), 'aopc': float(np.mean([r['aopc'] for r in res])),
            'suficiencia_20': float(np.mean([r['suficiencia_20'] for r in res])),
            'cambio_clase_20': None if cc[0] is None else float(np.mean(cc))}


def spearman(a, b):
    from scipy.stats import spearmanr
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(spearmanr(a, b).statistic)


def fragmentos_sin_verbo(texto, spans, nlp):
    """Proporción de unidades que no contienen un verbo ni un auxiliar (idea incompleta)."""
    sin = 0
    for a, b in spans:
        doc = nlp(texto[a:b])
        if not any(t.pos_ in ('VERB', 'AUX') for t in doc):
            sin += 1
    return sin / len(spans)
