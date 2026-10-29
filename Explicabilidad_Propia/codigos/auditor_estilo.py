"""
Auditor de estilo discursivo para una noticia (versión vigente de la explicabilidad propia).

Para un texto entrega, sin juzgar su veracidad:
  1. Las cinco dimensiones del IML (percentiles respecto del corpus), el IML y el estilo discursivo con sus grados de
     pertenencia (estilos_discursivos.perfilar).
  2. Qué variables explican cada dimensión (percentil de cada variable).
  3. Dónde está cada rasgo en el texto: la oración más sensacionalista, la más emotiva (y su emoción), el par de
     oraciones más parecido (marcado como «repite la misma afirmación» si supera tau_redundante, el único umbral de
     redundancia que se conserva, calibrado con STS-B) y los marcadores ortotipográficos y lingüísticos encontrados.
  4. Por separado, la lectura del Modelo sobre Variables (regresión logística): P(falsa) y las variables que más
     empujan esa probabilidad (coeficiente × valor estandarizado). Es la explicación del clasificador, no del estilo.
  5. Explicación por oraciones (adaptación de MExGen por leave-one-out; tesis, Subsección 3.6.3), con la misma
     segmentación de la comparación de explicabilidad (xai_metodos.spans_oraciones y xai_metodos.texto_sin): Δᵢ = f(texto) − f(texto sin la oración i), con f = IML
     (y cada dimensión D1–D5) y f = P(falsa) del Modelo sobre Variables. Una extracción de variables por versión del
     texto sirve para los dos (k + 1 extracciones para k oraciones).
"""

import sys
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config_proyecto.py").exists())))
import config_proyecto as cfg
sys.path.insert(0, str(Path(__file__).resolve().parent))
from variables_texto import ExtractorVariables, EMOCIONES
from estilos_discursivos import DIMENSIONES, perfilar, percentil, dimensiones_de
sys.path.insert(0, str(cfg.XAI_COMPARACION_DIR / 'codigos'))
import xai_metodos as xm

NOMBRE_EMOCION = {'others': 'sin emoción dominante', 'joy': 'alegría', 'sadness': 'tristeza', 'anger': 'ira',
                  'surprise': 'sorpresa', 'disgust': 'asco', 'fear': 'miedo'}


def _f(x, d=1):
    return '—' if x is None else f"{x:.{d}f}".replace('.', ',')


def informe_markdown(r):
    """Líneas Markdown con el perfil, las oraciones que concentran cada rasgo y la lectura del Modelo sobre Variables."""
    p, o, f = r['perfil'], r['oraciones'], _f
    L = [f"**IML = {f(p['iml'])}** · **Estilo:** {p['estilo']} ({f(p['pertenencia_estilo'], 0)} %), seguido de {p['estilo_secundario']} ({f(p['pertenencia_secundario'], 0)} %)", "",
         f"*{p['estilo'].split(': ', 1)[-1]}:* {p['lema_estilo']}", "",
         "| Dimensión | Percentil | Variables (percentil; ↑ suma, ↓ resta) |", "|:---|---:|:---|"]
    for d, info in r['variables_por_dimension'].items():
        vs = ", ".join(f"`{k}` {f(x['percentil'], 0)}{'↑' if x['direccion'] > 0 else '↓'}" for k, x in info['variables'].items())
        L.append(f"| {d} {info['nombre']} | {f(p['dimensiones'][d], 0)} | {vs} |")
    L += ["", "**Dónde está cada rasgo en el texto:**", "",
          f"- Oración más sensacionalista (n.º {o['mas_sensacionalista']['indice']}, P = {f(o['mas_sensacionalista']['probabilidad'], 2)}): «{o['mas_sensacionalista']['texto']}»",
          f"- Oración más emotiva (n.º {o['mas_emotiva']['indice']}, intensidad {f(o['mas_emotiva']['intensidad'], 2)}, {o['mas_emotiva']['emocion']}): «{o['mas_emotiva']['texto']}»"]
    if 'par_mas_parecido' in o:
        q = o['par_mas_parecido']
        L.append(f"- Par de oraciones más parecido (n.º {q['oracion_a']} y {q['oracion_b']}, similitud {f(q['similitud'], 2)}): "
                 + ("**repiten la misma afirmación** (supera τ = " + f(q['tau_redundante'], 3) + ")" if q['repite_la_misma_afirmacion']
                    else "no llegan a repetir la misma afirmación (τ = " + f(q['tau_redundante'], 3) + ")"))
    m = r['marcadores']
    L.append(f"- Marcadores: mayúsculas sostenidas {m['mayusculas_sostenidas'] or 'ninguna'}; exclamaciones {m['exclamaciones']}; "
             f"interrogaciones {m['interrogaciones']}; puntos suspensivos {m['puntos_suspensivos']}; intensificadores {m['intensificadores'] or 'ninguno'}; "
             f"atenuadores {m['atenuadores'] or 'ninguno'}; verbos de atribución {m['verbos_atribucion'] or 'ninguno'}.")
    ms = r['modelo_sobre_variables']
    L += ["", f"**Lectura del Modelo sobre Variables:** P(falsa) = {f(ms['p_falsa'], 2)} (umbral {f(ms['umbral'], 2)}) → {ms['prediccion']}. "
          "Variables que más empujan esa probabilidad:", "",
          "| Variable | Valor | z | Contribución (log-odds) | Empuja hacia |", "|:---|---:|---:|---:|:---|"]
    for c in ms['principales_contribuciones'][:6]:
        L.append(f"| `{c['variable']}` | {f(c['valor'], 3)} | {f(c['z'], 2)} | {f(c['contribucion_log_odds'], 2)} | {c['empuja_hacia']} |")
    mg = r.get('mexgen')
    if mg:
        L += ["", "**MexGen por oraciones** (Δ = valor con el texto completo − valor sin la oración; Δ > 0: la oración sostiene el valor):", ""]
        if not mg['aplica']:
            return L + [f"No aplica: {mg['motivo']}.", ""]
        L += ["| N.º | Oración | ΔIML | ΔP(falsa) |", "|---:|:---|---:|---:|"]
        for x in mg['oraciones']:
            t = ' '.join(x['texto'].split()).replace('|', '/')
            t = t if len(t) <= 110 else t[:107] + '…'
            L.append(f"| {x['indice']} | {t} | {_signo(x['delta_iml'], 1)} | {_signo(x['delta_p_falsa'], 3)} |")
        e = mg['extremos']
        L += ["", f"- Oración que más **sube** el IML: n.º {e['sube_iml']['indice']} (ΔIML {_signo(e['sube_iml']['delta'], 1)}); "
                  f"la que más lo **baja**: n.º {e['baja_iml']['indice']} (ΔIML {_signo(e['baja_iml']['delta'], 1)}).",
              f"- Oración que más empuja hacia **falsa**: n.º {e['hacia_falsa']['indice']} (ΔP {_signo(e['hacia_falsa']['delta'], 3)}); "
              f"hacia **real**: n.º {e['hacia_real']['indice']} (ΔP {_signo(e['hacia_real']['delta'], 3)})."]
    return L + [""]


def _signo(x, d):
    return '—' if x is None else (f"{x:+.{d}f}").replace('.', ',').replace('-', '−')


class AuditorEstilo:
    def __init__(self):
        self.ex = ExtractorVariables()
        self.ref = json.loads(cfg.REFERENCIA_ESTILOS_JSON.read_text(encoding='utf-8'))
        red = json.loads(cfg.METRICAS_REDUNDANCIA_JSON.read_text(encoding='utf-8'))
        self.tau_redundante = red['sts_es']['umbrales']['tau_redundante']['tau']
        self._ajustar_modelo_sobre_variables()

    def _ajustar_modelo_sobre_variables(self):
        """Regresión logística del Modelo sobre Variables ajustada con los textos únicos del corpus (uso descriptivo).
        El umbral es la mediana de los umbrales anidados de la validación cruzada."""
        sys.path.insert(0, str(cfg.MODELO_VARIABLES_CODIGOS_DIR))
        import evaluar_modelo_sobre_variables_sin_fuga as ev
        meta = json.loads(cfg.TABLA_VARIABLES_META_JSON.read_text(encoding='utf-8'))
        self.vars_modelo = [v for g in ['S', 'R', 'E', 'O', 'L'] for v in meta['grupos'][g]]
        df = pd.read_csv(cfg.TABLA_VARIABLES_CSV).drop_duplicates('id_texto')
        self.lr = ev.modelo('lr').fit(df[self.vars_modelo].values.astype(float), df['label_num'].values)
        res = json.loads(cfg.METRICAS_MODELO_VARIABLES_SIN_FUGA_JSON.read_text(encoding='utf-8'))
        self.theta = float(np.median(res['validacion_cruzada']['Completo (S+R+E+O+L)']['umbrales_por_pliegue']))

    def lectura_modelo(self, v):
        imp, esc, lr = self.lr[0], self.lr[1], self.lr[2]
        x = imp.transform(np.array([[v[k] for k in self.vars_modelo]], dtype=float))
        z = esc.transform(x)[0]
        contrib = lr.coef_[0] * z
        p = float(self.lr.predict_proba(x)[0, 1])
        orden = np.argsort(-np.abs(contrib))[:8]
        return {'p_falsa': round(p, 3), 'umbral': self.theta, 'prediccion': 'falsa' if p >= self.theta else 'real',
                'principales_contribuciones': [{'variable': self.vars_modelo[i], 'valor': round(float(v[self.vars_modelo[i]]), 4) if not np.isnan(v[self.vars_modelo[i]]) else None,
                                                'z': round(float(z[i]), 2), 'contribucion_log_odds': round(float(contrib[i]), 3),
                                                'empuja_hacia': 'falsa' if contrib[i] > 0 else 'real'} for i in orden]}

    def _p_falsa(self, v):
        return float(self.lr.predict_proba(np.array([[v[k] for k in self.vars_modelo]], dtype=float))[0, 1])

    def _lectura(self, v):
        d = dimensiones_de(v, self.ref)
        return float(np.nanmean(list(d.values()))), d, self._p_falsa(v)

    def mexgen(self, texto, v):
        """Pilar 3: MexGen por oraciones sobre el IML (y D1–D5) y sobre P(falsa) del Modelo sobre Variables."""
        sp = xm.spans_oraciones(texto)
        if len(sp) < 2:
            return {'aplica': False, 'motivo': 'el texto tiene una sola oración'}
        iml0, d0, p0 = self._lectura(v)
        filas = []
        for i, (a, b) in enumerate(sp):
            vi, _ = self.ex.extraer(xm.texto_sin(texto, sp, [i]))
            iml_i, d_i, p_i = self._lectura(vi)
            filas.append({'indice': i + 1, 'texto': texto[a:b], 'delta_iml': round(iml0 - iml_i, 2),
                          'delta_dimensiones': {k: None if np.isnan(d0[k]) or np.isnan(d_i[k]) else round(d0[k] - d_i[k], 2) for k in d0},
                          'delta_p_falsa': round(p0 - p_i, 4)})

        def extremo(clave, signo):
            x = max(filas, key=lambda y: signo * y[clave])
            return {'indice': x['indice'], 'delta': x[clave]}
        return {'aplica': True, 'segmentacion': 'Punkt en español (xai_metodos.spans_oraciones)', 'n_oraciones': len(sp),
                'iml_texto': round(iml0, 2), 'p_falsa_texto': round(p0, 4), 'oraciones': filas,
                'extremos': {'sube_iml': extremo('delta_iml', 1), 'baja_iml': extremo('delta_iml', -1),
                             'hacia_falsa': extremo('delta_p_falsa', 1), 'hacia_real': extremo('delta_p_falsa', -1)}}

    def auditar(self, texto, titulo='', con_mexgen=True):
        texto = unicodedata.normalize('NFC', str(texto))  # mismas posiciones de oración que en extraer()
        v, det = self.ex.extraer(texto)
        perfil = perfilar(v, self.ref)
        variables_por_dimension = {}
        for d, (nombre, _, vs) in DIMENSIONES.items():
            variables_por_dimension[d] = {'nombre': nombre, 'variables': {
                k: {'valor': None if np.isnan(v[k]) else round(float(v[k]), 4),
                    'percentil': None if np.isnan(v[k]) else round(percentil(v[k], np.asarray(self.ref['percentiles'][k])), 1),
                    'direccion': s} for k, s in vs.items()}}
        o = det['oraciones']
        ps = np.array(det['sens_oraciones'])
        q = np.array(det['emociones_oraciones'])
        i_s, i_e = int(ps.argmax()), int((1 - q[:, 0]).argmax())
        emo_i = int(q[i_e, 1:].argmax()) + 1
        oraciones = {
            'n_oraciones': len(o),
            'mas_sensacionalista': {'indice': i_s + 1, 'texto': o[i_s], 'probabilidad': round(float(ps[i_s]), 3)},
            'mas_emotiva': {'indice': i_e + 1, 'texto': o[i_e], 'intensidad': round(float(1 - q[i_e, 0]), 3),
                            'emocion': NOMBRE_EMOCION[EMOCIONES[emo_i]], 'probabilidad_emocion': round(float(q[i_e, emo_i]), 3)},
            'detalle': [{'indice': k + 1, 'texto': o[k], 'sensacionalismo': round(float(ps[k]), 3), 'intensidad_emocional': round(float(1 - q[k, 0]), 3),
                         'emocion_principal': NOMBRE_EMOCION[EMOCIONES[int(q[k, 1:].argmax()) + 1]]} for k in range(len(o))],
        }
        if det['similitud'] is not None:
            s = np.array(det['similitud'])
            np.fill_diagonal(s, -1)
            a, b = np.unravel_index(s.argmax(), s.shape)
            a, b = sorted((int(a), int(b)))
            oraciones['par_mas_parecido'] = {'oracion_a': a + 1, 'texto_a': o[a], 'oracion_b': b + 1, 'texto_b': o[b], 'similitud': round(float(s[a, b]), 3),
                                             'repite_la_misma_afirmacion': bool(s[a, b] >= self.tau_redundante), 'tau_redundante': self.tau_redundante}
        return {'titulo': titulo, 'perfil': perfil, 'variables_por_dimension': variables_por_dimension,
                'oraciones': oraciones, 'marcadores': det['marcadores'], 'modelo_sobre_variables': self.lectura_modelo(v),
                'mexgen': self.mexgen(texto, v) if con_mexgen else None,
                'variables': {k: (None if np.isnan(x) else float(x)) for k, x in v.items()}}
