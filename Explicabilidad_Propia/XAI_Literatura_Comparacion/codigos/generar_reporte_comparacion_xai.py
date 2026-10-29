"""
Genera reportes/Reporte_Comparativo_XAI_Literatura_vs_Propia.md y sus figuras a partir de
reportes/datos/metricas_comparacion_xai.json (producido por comparar_explicabilidad.py).
Todas las cifras se calculan del JSON.
"""

import sys
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config_proyecto.py").exists())))
import config_proyecto as cfg
import graficos_comunes as gc

REP =cfg.XAI_COMPARACION_DIR / 'reportes'
IMG = REP / 'imagenes'
OBJ = ['jjneila', 'sabert', 'redundancia', 'modelo_variables']
NOMBRE_OBJ = {'jjneila': 'JJNeila (sensacionalismo)', 'sabert': 'SaBERT (veracidad)', 'redundancia': 'hiiamsid (redundancia)',
              'modelo_variables': 'Modelo sobre Variables'}
PALABRA = ['IG', 'IxG', 'Rollout', 'LRP-Chefer', 'LIME-palabra', 'SHAP-palabra']
ORACION = ['MexGen', 'LIME-oración', 'SHAP-oración']
AGREGADOS = [f'{m}→oración' for m in PALABRA]
N_BOOT = 2000
EQUIPO = 'GPU NVIDIA GeForce GTX 1650 (4 GB), Ubuntu 26.04, torch 2.14.1+cu126'
NOMBRE_GRAN = {'palabra': 'palabra', 'oracion': 'oración', 'oracion (agregado)': 'oración (agregado)', 'clausula': 'cláusula'}


def f(x, d=3):
    return '—' if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}".replace('.', ',')


def fp(p):
    return 'p < 0,001' if p < 0.001 else f"p = {f(p)}"


def pct(x, d=0):
    return '—' if x is None or np.isnan(x) else f"{100 * x:.{d}f}".replace('.', ',') + " %"


def ic_media(v, rng):
    v = np.asarray([x for x in v if x is not None and not np.isnan(x)], float)
    if len(v) == 0:
        return np.nan, (np.nan, np.nan)
    b = [rng.choice(v, len(v)).mean() for _ in range(N_BOOT)]
    return float(v.mean()), (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))


def gran_base(gran):
    return 'oracion' if gran.startswith('oracion') else gran


def holm(ps):
    orden = np.argsort(ps)
    out, m, acc = np.empty(len(ps)), len(ps), 0.0
    for i, j in enumerate(orden):
        acc = max(acc, min(1.0, (m - i) * ps[j]))
        out[j] = acc
    return out


def resumir(docs, ade):
    """Por método: AOPC, ganancia sobre el azar de su granularidad, suficiencia, cambio de clase, tiempo, acuerdo y Adebayo."""
    rng = np.random.default_rng(cfg.SEED)
    metodos = [m for m in PALABRA + AGREGADOS + ORACION + ['MexGen-cláusulas'] if m in docs[0]['metodos']]
    res = {}
    for m in metodos:
        gran = docs[0]['metodos'][m]['granularidad']
        g = [d['metodos'][m]['aopc'] - d['aleatorio'][gran_base(gran)]['aopc'] for d in docs]
        cc = [d['metodos'][m]['cambio_clase_20'] for d in docs]
        res[m] = {
            'granularidad': gran,
            'aopc': ic_media([d['metodos'][m]['aopc'] for d in docs], rng),
            'ganancia': ic_media(g, rng), 'ganancia_vec': g,
            'supera_azar': float(np.mean(np.array(g) > 0)),
            'suficiencia': ic_media([d['metodos'][m]['suficiencia_20'] for d in docs], rng),
            'cambio_clase': None if cc[0] is None else float(np.mean(cc)),
            'segundos': float(np.mean([d['metodos'][m]['segundos'] for d in docs])),
            'acuerdo_mexgen': ic_media([d['acuerdo_con_mexgen'].get(m.replace('→oración', '')) for d in docs], rng)[0]
                              if m not in ('MexGen', 'MexGen-cláusulas') else np.nan,
            'adebayo': ic_media([a.get(m) for a in ade], rng)[0] if ade and m in ade[0] else np.nan,
            'curva': np.mean([d['metodos'][m]['curva'] for d in docs], 0).tolist(),
        }
    res['_aleatorio'] = {k: {'aopc': float(np.mean([d['aleatorio'][k]['aopc'] for d in docs])),
                             'curva': np.mean([d['aleatorio'][k]['curva'] for d in docs], 0).tolist(),
                             'cambio_clase': None if docs[0]['aleatorio'][k]['cambio_clase_20'] is None else float(np.mean([d['aleatorio'][k]['cambio_clase_20'] for d in docs]))}
                         for k in docs[0]['aleatorio']}
    # MexGen frente a cada método en ganancia (Wilcoxon pareado, Holm)
    comps = [m for m in metodos if m != 'MexGen']
    ps, difs = [], []
    for m in comps:
        a, b = np.array(res['MexGen']['ganancia_vec']), np.array(res[m]['ganancia_vec'])
        difs.append(float(np.mean(a - b)))
        ps.append(wilcoxon(a, b).pvalue if np.any(a != b) else 1.0)
    for m, d_, p_, ph in zip(comps, difs, ps, holm(np.array(ps))):
        res[m]['mexgen_menos_metodo'] = d_
        res[m]['p_holm'] = float(ph)
    return res


def figuras(R, meta):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    IMG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.dpi': 150, 'savefig.dpi': 200})
    TINTA, T2, REJ = '#0b0b0b', '#52514e', '#e1e0d9'
    COL = {'palabra': '#2a78d6', 'oracion (agregado)': '#86b6ef', 'oracion': '#eb6834', 'clausula': '#898781'}
    out = {}
    objs = [o for o in OBJ if o in R]
    # 1. Ganancia sobre el azar por método y modelo
    fig, axes = plt.subplots(1, len(objs), figsize=(4.2 * len(objs), 5.2), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, o in zip(axes, objs):
        ms = [m for m in PALABRA + ORACION + ['MexGen-cláusulas'] if m in R[o]]
        vals = [R[o][m]['ganancia'][0] for m in ms]
        err = np.array([[R[o][m]['ganancia'][0] - R[o][m]['ganancia'][1][0], R[o][m]['ganancia'][1][1] - R[o][m]['ganancia'][0]] for m in ms]).T
        cols = [COL[R[o][m]['granularidad']] for m in ms]
        y = np.arange(len(ms))[::-1]
        ax.barh(y, vals, 0.65, color=cols, edgecolor='white', xerr=err, error_kw={'ecolor': T2, 'elinewidth': 0.8, 'capsize': 2})
        ax.axvline(0, color='#c3c2b7', linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(ms, fontsize=8.5, color=TINTA)
        ax.set_title(NOMBRE_OBJ[o], fontsize=10, color=TINTA, loc='left')
        ax.grid(axis='x', color=REJ, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=T2, labelsize=8)
        ax.set_xlabel('Ganancia de AOPC sobre el azar', fontsize=8.5, color=TINTA)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=COL['palabra'], label='Nivel palabra'), Patch(color=COL['oracion'], label='Nivel oración (nativo)'),
                        Patch(color=COL['clausula'], label='Cláusulas (MexGen anterior)')], loc='lower center', ncol=3, frameon=False, fontsize=9)
    fig.suptitle('Fidelidad: cuánto más cae la salida del modelo al borrar lo que el método marca como importante,\nfrente a borrar al azar la misma cantidad de texto',
                 fontsize=10.5, x=0.01, ha='left', color=TINTA)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    out['ganancia'] = IMG / 'ganancia_fidelidad_por_metodo.png'
    fig.savefig(out['ganancia'])
    plt.close(fig)
    # 2. Curvas de borrado
    q = np.array(meta['rejilla_q']) * 100
    fig, axes = plt.subplots(1, len(objs), figsize=(4.2 * len(objs), 3.8))
    axes = np.atleast_1d(axes)
    estilo = {'MexGen': ('#eb6834', '-', 2.6), 'LIME-oración': ('#e87ba4', '-', 1.6), 'SHAP-oración': ('#6250d6', '-', 1.6),
              'IG': ('#2a78d6', '--', 1.4), 'LIME-palabra': ('#1baf7a', '--', 1.4), 'SHAP-palabra': ('#008300', '--', 1.4)}
    for ax, o in zip(axes, objs):
        for m, (c, ls, lw) in estilo.items():
            if m in R[o]:
                ax.plot(q, R[o][m]['curva'], color=c, linestyle=ls, linewidth=lw, label=m)
        for k, ls in [('palabra', ':'), ('oracion', '-.')]:
            ax.plot(q, R[o]['_aleatorio'][k]['curva'], color='#898781', linestyle=ls, linewidth=1.2, label=f'Azar ({k.replace("oracion", "oración")})')
        ax.set_title(NOMBRE_OBJ[o], fontsize=10, color=TINTA, loc='left')
        ax.set_xlabel('% de palabras eliminadas', fontsize=8.5, color=TINTA)
        ax.grid(color=REJ, linewidth=0.6)
        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=T2, labelsize=8)
    axes[0].set_ylabel('Caída de la salida del modelo', fontsize=8.5, color=TINTA)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    out['curvas'] = IMG / 'curvas_borrado.png'
    fig.savefig(out['curvas'])
    plt.close(fig)
    # 3. Adebayo
    ms = [m for m in PALABRA + ORACION + ['MexGen-cláusulas']]
    objs_a = [o for o in objs if o != 'modelo_variables']
    M = np.array([[R[o][m]['adebayo'] if m in R[o] else np.nan for o in objs_a] for m in ms])
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('div', ['#2a78d6', '#f0efec', '#e34948'])
    fig, ax = plt.subplots(figsize=(5.8, 5))
    im = ax.imshow(M, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
    for i in range(len(ms)):
        for j in range(len(objs_a)):
            ax.text(j, i, '—' if np.isnan(M[i, j]) else f'{M[i, j]:.2f}'.replace('.', ','), ha='center', va='center', fontsize=8.5, color=TINTA)
    ax.set_xticks(range(len(objs_a)))
    ax.set_xticklabels([NOMBRE_OBJ[o].split(' (')[0] for o in objs_a], fontsize=9)
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels(ms, fontsize=8.5)
    ax.set_title('Prueba de Adebayo: correlación entre la explicación\ncon el modelo original y con el modelo aleatorizado\n(cerca de 0 = el método depende del modelo; cerca de 1 = no)',
                 fontsize=9.5, loc='left', color=TINTA)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    out['adebayo'] = IMG / 'prueba_adebayo.png'
    fig.savefig(out['adebayo'])
    plt.close(fig)
    return {k: 'imagenes/' + v.name for k, v in out.items()}


def elegir_ejemplo(objetivo='jjneila'):
    """Noticia de JJNeila para el ejemplo cualitativo: explicada como sensacionalista, de 4 a 6 oraciones y corta, con la
    oración más influyente según MexGen más marcada (criterio fijo, sin elegir a mano). Lee las atribuciones de codigos/cache/."""
    carpeta = cfg.XAI_COMPARACION_DIR / 'codigos' / 'cache' / objetivo
    docs = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(carpeta.glob('*.json')) if not p.name.startswith('adebayo_')] \
        if carpeta.exists() else []
    docs = [d for d in docs if '_atribuciones' in d and 'oraciones' in d]
    def ok(d, clase):
        return (clase is None or d['clase_explicada'] == clase) and 4 <= d['n_oraciones'] <= 6 and d['n_palabras'] <= 130 \
            and len(' '.join(d['oraciones']).split()) == len(d['_atribuciones']['SHAP-palabra'])
    for clase in (1, None):
        cand = [d for d in docs if ok(d, clase)]
        if cand:
            return max(cand, key=lambda d: max(abs(x) for x in d['_atribuciones']['MexGen']))
    return None


def figuras_extra(R, j):
    """Figuras añadidas con el estilo común: ejemplo cualitativo, acuerdo con MexGen, costo frente a ganancia y borrado al azar."""
    sys.path.insert(0, str(cfg.ROOT))
    import graficos_comunes as gc
    plt = gc.pyplot()
    objs = [o for o in OBJ if o in R]
    out, info = {}, {}
    COL_OR = {'MexGen': gc.SERIE[1], 'LIME-oración': gc.SERIE[4], 'SHAP-oración': gc.SERIE[6]}

    # (a) Ejemplo cualitativo: importancia por oración (3 métodos) y por palabra (SHAP-palabra)
    d = elegir_ejemplo('jjneila')
    if d is not None:
        info['ejemplo'] = d
        ors = d['oraciones']
        palabras = ' '.join(ors).split()
        fig = plt.figure(figsize=(10, 6.4))
        a1 = fig.add_axes((0.30, 0.50, 0.66, 0.44))
        a2 = fig.add_axes((0.02, 0.0, 0.96, 0.38))
        y = np.arange(len(ors))[::-1]
        ms = ['MexGen', 'LIME-oración', 'SHAP-oración']
        for k, m in enumerate(ms):
            v = np.array(d['_atribuciones'][m], float)
            v = v / (np.abs(v).max() or 1)
            a1.barh(y + (1 - k) * 0.26, v, height=0.24, color=COL_OR[m], label=m)
        a1.axvline(0, color=gc.EJE, lw=1)
        etiquetas = []
        for i, o in enumerate(ors):
            w = o.split()
            etiquetas.append(f"O{i + 1}: «{' '.join(w[:6])}{' …' if len(w) > 6 else ''}»")
        a1.set_yticks(y, etiquetas, fontsize=8.5)
        a1.set_xlim(-1.1, 1.1)
        a1.set_xticks([-1, -0.5, 0, 0.5, 1])
        gc.coma(a1, 'x', 1)
        a1.set_xlabel('Importancia normalizada (máximo |valor| = 1 por método)')
        a1.grid(axis='y', visible=False)
        a1.legend(fontsize=8.5, loc='lower right')
        a1.set_title('Por oración: tres métodos que explican la misma unidad')
        # Palabras coloreadas por SHAP-palabra
        a2.axis('off')
        a2.set_xlim(0, 1)
        a2.set_ylim(0, 1)
        a2.text(0, 0.98, 'Por palabra: SHAP-palabra sobre la misma noticia (rojo = empuja hacia la clase explicada; azul = en contra)',
                fontsize=10, color=gc.TINTA, va='top')
        v = np.array(d['_atribuciones']['SHAP-palabra'], float)
        v = v / (np.abs(v).max() or 1)
        cmap = gc.divergente()
        r = fig.canvas.get_renderer()
        inv = a2.transAxes.inverted()
        x, yy, alto = 0.0, 0.86, 0.09
        for w_, s in zip(palabras, v):
            t = a2.text(x, yy, w_, fontsize=9, va='top', color=gc.TINTA,
                        bbox=dict(fc=cmap((s + 1) / 2), ec='none', pad=1.2, boxstyle='round,pad=0.15'))
            bb = inv.transform(t.get_window_extent(r))
            ancho = bb[1, 0] - bb[0, 0]
            if x + ancho > 0.99 and x > 0:
                x, yy = 0.0, yy - alto
                t.set_position((x, yy))
            x += ancho + 0.006
        # Recorta el panel de palabras a lo usado sin mover el texto (mismo mapeo a la figura)
        piso = yy - alto
        p = a2.get_position()
        a2.set_position((p.x0, p.y0 + p.height * piso, p.width, p.height * (1 - piso)))
        a2.set_ylim(piso, 1)
        out['ejemplo'] = gc.guardar(fig, IMG / 'ejemplo_oracion_vs_palabra.png')

    # (b) Acuerdo con MexGen (ρ de Spearman del orden de las oraciones)
    ms = [m for m in PALABRA + ['LIME-oración', 'SHAP-oración'] if any(m in R[o] for o in objs)]
    M = np.array([[R[o][m]['acuerdo_mexgen'] if m in R[o] else np.nan for o in objs] for m in ms])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    im = ax.imshow(M, cmap=gc.divergente(), vmin=-1, vmax=1, aspect='auto')
    ax.grid(False)
    for i in range(len(ms)):
        for k in range(len(objs)):
            ax.text(k, i, 'no aplica' if np.isnan(M[i, k]) else f(M[i, k], 2), ha='center', va='center', fontsize=8.5,
                    color=gc.TINTA_SUAVE if np.isnan(M[i, k]) else gc.TINTA)
    ax.set_xticks(range(len(objs)), [NOMBRE_OBJ[o].replace(' (', '\n(') for o in objs], fontsize=8.5)
    ax.set_yticks(range(len(ms)), ms, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04, label='ρ de Spearman')
    ax.set_title('Acuerdo con MexGen en el orden de las oraciones\n(1 = mismo orden; 0 = sin relación)', fontsize=10)
    out['acuerdo'] = gc.guardar(fig, IMG / 'acuerdo_con_mexgen.png')

    # (c) Costo frente a ganancia
    from matplotlib.ticker import MaxNLocator
    # Posiciones candidatas de cada etiqueta (desplazamiento en puntos); se usa la primera que no choca con otra etiqueta ni punto
    CAND = [(6, 4, 'left'), (6, -11, 'left'), (-6, 4, 'right'), (-6, -11, 'right'), (8, 16, 'left'), (8, -24, 'left'),
            (-8, 16, 'right'), (-8, -24, 'right'), (10, 28, 'left'), (-10, -34, 'right')]
    fig, axs = plt.subplots(1, len(objs), figsize=(3.8 * len(objs), 3.9), sharey=False)
    puntos = {}
    for ax, o in zip(np.atleast_1d(axs), objs):
        ms = [x for x in PALABRA + ORACION if x in R[o]]
        puntos[o] = {m: (max(R[o][m]['segundos'], 0.01), R[o][m]['ganancia'][0]) for m in ms}
        for m, (s, g) in puntos[o].items():
            ax.scatter(s, g, s=90 if m == 'MexGen' else 45, color=gc.SERIE[1] if m in ORACION else gc.SERIE[0],
                       edgecolor=gc.TINTA if m == 'MexGen' else 'white', linewidth=1.2, zorder=3)
        ax.set_xscale('log')
        ax.set_xlim(0.007, 80)
        ax.margins(y=0.15)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
        ax.axhline(0, color=gc.EJE, lw=1)
        ax.set_title(NOMBRE_OBJ[o], fontsize=10)
        ax.set_xlabel('Segundos por noticia (escala log)')
        gc.coma(ax, 'y', 2)
    np.atleast_1d(axs)[0].set_ylabel('Ganancia sobre el azar')
    fig.tight_layout()   # antes de colocar las etiquetas, para que las posiciones en píxeles sean las finales
    fig.canvas.draw()
    for ax, o in zip(np.atleast_1d(axs), objs):
        pts = puntos[o]
        a_px = 72 / fig.dpi   # puntos por píxel
        ocupado = [(*ax.transData.transform(p), 5, 5) for p in pts.values()]   # cajas (x, y, semiancho, semialto) en píxeles
        # MexGen primero y a la izquierda de su punto; el resto, de mayor a menor ganancia
        for m, (s, g) in sorted(pts.items(), key=lambda kv: (kv[0] != 'MexGen', -kv[1][1])):
            px, py = ax.transData.transform((s, g))
            w, h = len(m) * 4.6 / a_px, 6 / a_px
            for dx, dy, ha in ([(-9, -3, 'right')] if m == 'MexGen' else []) + CAND:
                cx = px + (dx / a_px) + (w / 2 if ha == 'left' else -w / 2)
                cy = py + dy / a_px + h / 2
                if all(abs(cx - ox) > w / 2 + ow or abs(cy - oy) > h / 2 + oh for ox, oy, ow, oh in ocupado):
                    break
            ocupado.append((cx, cy, w / 2, h / 2))
            ax.annotate(m, (s, g), xytext=(dx, dy), textcoords='offset points', fontsize=7.5, color=gc.TINTA_2, ha=ha,
                        arrowprops=dict(arrowstyle='-', color=gc.TINTA_SUAVE, lw=0.6, shrinkA=1, shrinkB=4))
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker='o', ls='', color=gc.SERIE[0], label='Método por palabra'),
                        Line2D([], [], marker='o', ls='', color=gc.SERIE[1], label='Método por oración')],
               loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.0))
    out['costo'] = gc.guardar(fig, IMG / 'costo_frente_a_ganancia.png')

    # (d) Borrado al azar: palabras sueltas frente a oraciones completas
    q = np.array(j['metadata']['rejilla_q']) * 100
    fig, axs = plt.subplots(1, len(objs), figsize=(3.6 * len(objs), 3.4))
    for ax, o in zip(np.atleast_1d(axs), objs):
        for k, (g, nombre) in enumerate([('palabra', 'Borrar palabras al azar'), ('oracion', 'Borrar oraciones al azar')]):
            ax.plot(q, R[o]['_aleatorio'][g]['curva'], color=gc.SERIE[k], marker='o', ms=4, label=nombre)
        ax.set_title(NOMBRE_OBJ[o], fontsize=10)
        ax.set_xlabel('% de palabras eliminadas')
        tope = max(max(np.abs(R[o]['_aleatorio'][g]['curva'])) for g in ('palabra', 'oracion'))
        gc.coma(ax, 'y', 3 if tope < 0.05 else 2)
    np.atleast_1d(axs)[0].set_ylabel('Caída de la salida del modelo')
    h, l = np.atleast_1d(axs)[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout()
    out['azar'] = gc.guardar(fig, IMG / 'borrado_azar_palabras_vs_oraciones.png')
    return {k: 'imagenes/' + v.name for k, v in out.items()}, info


COMPATIBILIDAD = [
    ('IG', 'Palabra (token)', 'Suma de tokens por oración (adaptación)', 'Sí', 'Sí (gradiente del coseno medio respecto de todas las oraciones a la vez)', 'No: el modelo no es diferenciable (expresiones regulares, conteos, percentiles)'),
    ('IxG', 'Palabra (token)', 'Suma de tokens por oración (adaptación)', 'Sí', 'Sí (igual que IG)', 'No (igual que IG)'),
    ('Rollout', 'Palabra (token)', 'Suma por oración (adaptación); sin signo', 'Sí', 'No: el objetivo combina varias codificaciones independientes; la atención de cada oración no dice nada de su similitud con las demás', 'No: no hay atención'),
    ('LRP-Chefer', 'Palabra (token)', 'Suma por oración (adaptación); solo relevancia positiva', 'Sí', 'No: la regla se define para la ruta del token [CLS] hacia una cabeza de clasificación', 'No: no hay atención'),
    ('LIME-palabra', 'Palabra', 'Variante nativa (LIME-oración)', 'Sí', 'Sí', 'No se usa: textos perturbados agramaticales y ~200 evaluaciones por noticia'),
    ('SHAP-palabra', 'Palabra', 'Variante nativa (SHAP-oración)', 'Sí', 'Sí', 'No se usa: mismo motivo (~300 evaluaciones por noticia)'),
    ('MexGen', 'Oración', '—', 'Sí', 'Sí', 'Sí (k + 1 evaluaciones)'),
]


def conclusion(R, j, filas):
    """Conclusión de la comparación; todas las cifras salen del JSON."""
    T = {m: (g, n, a, t) for m, g, n, a, t in filas}
    L = ["### Conclusión", ""]
    # 1. MexGen frente a LIME y SHAP por oración (misma unidad)
    eq, mejor, peor, det = 0, [], [], []
    for o in R:
        xs = [R[o][m] for m in ('LIME-oración', 'SHAP-oración') if m in R[o]]
        if all(x['p_holm'] >= 0.05 for x in xs):
            eq += 1
        elif all(x['mexgen_menos_metodo'] > 0 for x in xs if x['p_holm'] < 0.05):
            mejor.append(NOMBRE_OBJ[o])
        else:
            peor.append(NOMBRE_OBJ[o])
        det.append(f"{NOMBRE_OBJ[o]}: " + ", ".join(f"{f(R[o][m]['mexgen_menos_metodo'])} frente a {m} ({fp(R[o][m]['p_holm'])})"
                                                   for m in ('LIME-oración', 'SHAP-oración') if m in R[o]))
    acu = [R[o][m]['acuerdo_mexgen'] for o in R for m in ('LIME-oración', 'SHAP-oración') if m in R[o]]
    resto = ((f"; en {', '.join(mejor)} MexGen es significativamente mejor" if mejor else '') +
             (f"; en {', '.join(peor)} es significativamente peor" if peor else ''))
    L += [f"1. **A nivel de oración, MexGen es tan fiel como LIME y SHAP.** La diferencia de ganancia no es significativa en {eq} de {len(R)} modelos{resto} "
          f"(MexGen − método, Wilcoxon con corrección de Holm: {'; '.join(det)}). Además ordenan las oraciones de forma parecida (ρ de Spearman entre {f(min(acu), 2)} y {f(max(acu), 2)}). "
          f"MexGen lo logra con menos costo ({f(T['MexGen'][3], 1)} s por noticia frente a {f(T['LIME-oración'][3], 1)} de LIME-oración y "
          f"{f(T['SHAP-oración'][3], 1)} de SHAP-oración), sin muestreo aleatorio (el resultado es el mismo en cada ejecución) y con una lectura directa: "
          "cuánto cambia la salida al quitar la oración.", ""]
    # 2. Sumar palabras por oración pierde fidelidad
    sup, tot = 0, 0
    for o in R:
        for m in AGREGADOS:
            if m in R[o]:
                tot += 1
                sup += R[o][m]['mexgen_menos_metodo'] > 0 and R[o][m]['p_holm'] < 0.05
    L += [f"2. **Llevar los métodos por palabra a oraciones sumando sus valores no funciona bien:** MexGen supera significativamente a la versión agregada en {sup} de "
          f"{tot} comparaciones. Es la razón para tener un método que trabaje directamente con oraciones.", ""]
    # 3. Métodos por palabra: más ganancia, pero con matices
    cc = []
    if 'redundancia' in R and 'conservando_cierre' in j['documentos']['redundancia'][0]:
        docs = j['documentos']['redundancia']
        for m in ('LIME-palabra', 'SHAP-palabra'):
            g_norm = np.mean([d['metodos'][m]['aopc'] - d['aleatorio']['palabra']['aopc'] for d in docs])
            g_cc = np.mean([d['conservando_cierre'][m]['aopc'] - d['conservando_cierre']['_aleatorio']['aopc'] for d in docs])
            cc.append(f"{m} pasa de {f(g_norm)} a {f(g_cc)}")
    L += [f"3. **LIME y SHAP por palabra obtienen la mayor ganancia** ({f(T['SHAP-palabra'][0])} y {f(T['LIME-palabra'][0])} de media), pero con tres matices: "
          "(a) su ganancia se mide contra el borrado al azar de palabras y la de MexGen contra el de oraciones, de modo que las cifras no son directamente comparables; "
          "(b) explican palabras sueltas, que un lector no puede verificar como afirmaciones, a partir de textos perturbados que ya no están bien escritos "
          "(ver «Limitación de los métodos que perturban palabras sueltas»)" +
          (f"; (c) en redundancia parte de su fidelidad se debe a la puntuación: al conservar el punto final {' y '.join(cc)}." if cc else ".") +
          f" Además son más lentos ({f(T['SHAP-palabra'][3], 1)} y {f(T['LIME-palabra'][3], 1)} s por noticia).", ""]
    # 4. Adebayo
    L += [f"4. **Attention Rollout y LRP-Chefer no superan la prueba de Adebayo:** su explicación casi no cambia al reiniciar las capas superiores del modelo "
          f"(ρ medio {f(T['Rollout'][2], 2)} y {f(T['LRP-Chefer'][2], 2)}), es decir, reflejan sobre todo la estructura del texto y de las capas inferiores, no la decisión "
          f"del modelo. MexGen ({f(T['MexGen'][2], 2)}), LIME y SHAP sí cambian. En redundancia la mayoría de los métodos cambian poco; una explicación probable es que la "
          "similitud entre oraciones depende en buena parte de las capas inferiores del codificador, que no se reinician.", ""]
    # 5. Cláusulas
    gc = [R[o]['MexGen-cláusulas']['ganancia'][0] - R[o]['MexGen']['ganancia'][0] for o in R if 'MexGen-cláusulas' in R[o]]
    L += [f"5. **Cortar por comas da más ganancia, pero explicaciones peores para leer:** MexGen por cláusulas gana en promedio {f(np.mean(gc))} más que por oraciones, "
          "porque con unidades más pequeñas la curva de borrado es más fina; a cambio, una gran parte de las cláusulas no tiene verbo y no expresa una idea completa. "
          "Por eso la tesis prefiere la oración.", "",
          "**En resumen:** MexGen no es el método con la fidelidad más alta en términos absolutos, pero es el único que reúne a la vez fidelidad equivalente a LIME y SHAP "
          "en la unidad que se explica (la oración), el menor costo entre los métodos por oración, resultados deterministas, aplicación a los cuatro modelos (incluido el "
          "Modelo sobre Variables, que no es diferenciable) y una explicación que un lector puede leer y contrastar.", ""]
    return L


def main():
    j = json.load(open(cfg.XAI_COMPARACION_DATOS_DIR / 'metricas_comparacion_xai.json', encoding='utf-8'))
    meta = j['metadata']
    R = {o: resumir(j['documentos'][o], j['adebayo'].get(o, [])) for o in OBJ if j['documentos'].get(o)}
    figs = figuras(R, meta)
    extra, info = figuras_extra(R, j)
    figs.update(extra)
    ej = info.get('ejemplo')
    TXT_EJEMPLO = [] if ej is None else [
        f"![Ejemplo por oración y por palabra]({figs['ejemplo']})", "",
        f"**Ejemplo (noticia `{ej['id_texto']}`, {ej['region']}, {ej['n_oraciones']} oraciones, {ej['n_palabras']} palabras):** JJNeila la clasifica como "
        f"{'sensacionalista' if ej['clase_explicada'] == 1 else 'no sensacionalista'} (P = {f(ej['f_texto'], 2)}). Arriba, la importancia de cada oración según MexGen, "
        "LIME-oración y SHAP-oración (cada método normalizado a su máximo); abajo, la misma noticia explicada por SHAP-palabra, con cada palabra coloreada. "
        "La explicación por oración señala afirmaciones completas que se pueden leer y contrastar; la explicación por palabra reparte la importancia en "
        "palabras sueltas, varias de ellas funcionales o signos, que no forman una idea. La noticia se elige con un criterio fijo (la de JJNeila, explicada "
        "como sensacionalista, de 4 a 6 oraciones y hasta 130 palabras, cuya oración más influyente según MexGen tiene mayor efecto).", ""]
    seg = {o: {'oraciones': np.mean([d['n_oraciones'] for d in j['documentos'][o]]),
               'clausulas': np.mean([d['n_clausulas'] for d in j['documentos'][o]]),
               'or_sin_verbo': np.mean([d['oraciones_sin_verbo'] for d in j['documentos'][o]]),
               'cl_sin_verbo': np.mean([d['clausulas_sin_verbo'] for d in j['documentos'][o]])} for o in R}
    s0 = seg[next(iter(R))]
    n_docs = {o: len(j['documentos'][o]) for o in R}

    L = ["# Explicabilidad propia (MexGen por oraciones) frente a seis métodos de la literatura", "",
         "**Generado por:** `codigos/generar_reporte_comparacion_xai.py` a partir de `reportes/datos/metricas_comparacion_xai.json`  ",
         "**Cálculo:** `codigos/comparar_explicabilidad.py` y `codigos/xai_metodos.py` (caché por noticia en `codigos/cache/`)  ",
         "**Muestra:** `reportes/datos/muestra_comparacion_xai.csv`: " + ", ".join(f"{NOMBRE_OBJ[o]} {n} noticias" for o, n in n_docs.items()) +
         f"; textos únicos del corpus, estratificados por región y clase, con 3 o más oraciones y como máximo {meta['max_palabras']} palabras "
         "(para que todos los métodos vean la noticia completa dentro de 256 tokens).  ",
         f"**Equipo:** {EQUIPO} (los tiempos por noticia se midieron allí).", "",
         "## 1. Qué se compara", "",
         "Se explica la salida de cuatro modelos sobre las mismas noticias:", "",
         "| Modelo | Qué número se explica |", "|:---|:---|"]
    for o, d in meta['objetivos'].items():
        if o in R:
            L.append(f"| {NOMBRE_OBJ[o]} | {d}{' (de la clase predicha)' if o in ('jjneila', 'sabert', 'modelo_variables') else ''} |")
    L += ["", "Métodos:", "",
          "| Método | Referencia | Cómo asigna importancia | Granularidad nativa |", "|:---|:---|:---|:---|",
          f"| IG | Sundararajan, Taly y Yan (2017) | Integra el gradiente a lo largo del camino entre una referencia ([PAD]) y el texto ({meta['pasos_ig']} pasos) | Token |",
          "| IxG | Shrikumar et al. (2017) | Gradiente × entrada, en una sola pasada | Token |",
          "| Attention Rollout | Abnar y Zuidema (2020) | Multiplica las matrices de atención de todas las capas (con conexión residual) | Token |",
          "| LRP-Chefer | Chefer, Gur y Wolf (2021) | Propaga relevancia por las capas ponderando la atención por su gradiente positivo | Token |",
          f"| LIME | Ribeiro, Singh y Guestrin (2016) | Ajusta un modelo lineal local sobre perturbaciones ({meta['lime_palabra']} por noticia a nivel palabra, {meta['lime_oracion']} a nivel oración) | Cualquier unidad |",
          f"| SHAP (Partition) | Lundberg y Lee (2017) | Aproxima valores de Shapley con una partición jerárquica de las unidades ({meta['shap_palabra']} evaluaciones a nivel palabra) | Cualquier unidad |",
          "| **MexGen (propio)** | Adaptación de MExGen (Monteiro Paes et al., ACL 2025) | Quita una oración por vez y mide cuánto cambia la salida | Oración |", "",
          "## 2. Cómo funciona el MexGen adaptado", "",
          gc.diagrama_md('mexgen.png', REP, 'MexGen adaptado: leave-one-out por oraciones'),
          "MExGen (Monteiro Paes et al., 2025) es un marco de IBM Research para explicar modelos de lenguaje por **unidades lingüísticas** en varios niveles (párrafos, "
          "oraciones, frases y palabras) mediante perturbaciones que **eliminan** la unidad; propone algoritmos de complejidad lineal (*leave-one-out*, C-LIME y L-SHAP) "
          "y un refinamiento de lo grueso a lo fino para controlar el costo en textos largos. Segmenta con spaCy y obtiene las frases a partir del árbol de dependencias.", "",
          "La adaptación usada en esta tesis toma de MExGen tres decisiones y simplifica el resto:", "",
          "1. **Unidad = oración.** La noticia se divide en oraciones con Punkt para español (cierre en punto, signo de exclamación o de interrogación).",
          "2. **Algoritmo *leave-one-out*:** para cada oración i, Δᵢ = f(texto) − f(texto sin la oración i). Necesita k + 1 evaluaciones del modelo para k oraciones.",
          "3. **Perturbación por eliminación**, no por reemplazo con [MASK] (que genera entradas que el modelo nunca vio).",
          "4. **Sin «escalarizador»:** MExGen lo necesita porque explica texto generado; aquí la salida ya es un número (una probabilidad o una similitud).",
          "5. **Sin refinamiento a frases o palabras:** la explicación se queda en la oración, que es la unidad que un lector humano verifica.", "",
          "El resultado se lee directamente: «quitar esta oración baja la probabilidad de sensacionalista en 0,30».", "",
          "### Por qué se segmenta por puntos y no por comas", "",
          "La versión anterior de MexGen cortaba también en comas, punto y coma, dos puntos, comillas y paréntesis. Se cambió a oraciones porque el punto marca con mucha "
          "más certeza que **una idea terminó**: la coma suele unir una idea con su complemento (una aposición, una enumeración, una subordinada, un inciso), y al cortar "
          "allí se separa el sujeto de lo que se dice de él. Cada fragmento deja de ser una cláusula completa y la explicación señala pedazos sin sentido propio.", "",
          "Los datos lo confirman. En la muestra, una noticia tiene en promedio "
          f"{f(s0['oraciones'], 1)} oraciones y {f(s0['clausulas'], 1)} cláusulas cortadas por puntuación; **{pct(s0['cl_sin_verbo'])} de las cláusulas no tienen ningún verbo** "
          f"(son ideas incompletas), frente a {pct(s0['or_sin_verbo'])} de las oraciones (análisis morfosintáctico con spaCy). "
          "La fidelidad de ambas variantes se compara en la sección 5.", "",
          "## 3. Variantes por oración: qué se pudo hacer y qué no", "",
          "| Método | Granularidad nativa | Variante por oración | JJNeila / SaBERT | hiiamsid (redundancia) | Modelo sobre Variables |", "|:---|:---|:---|:---|:---|:---|"]
    for fila in COMPATIBILIDAD:
        L.append("| " + " | ".join(fila) + " |")
    L += ["", "**Dificultades encontradas al obtener la variante por oración:**", "",
          "- **LIME** la admite de forma nativa: basta con indicarle que los rasgos son las oraciones (`split_expression`) y que una oración ausente se borra (`mask_string=''`). "
          "Con pocas oraciones (3 a 8 por noticia) el modelo lineal local se ajusta con pocas perturbaciones.",
          "- **SHAP** también la admite, pero con dos «mañas»: (1) su enmascarador de texto exige un tokenizador que devuelva unidades y posiciones para *cualquier* cadena, incluida la "
          "vacía, de modo que hubo que escribir uno que segmente en oraciones; (2) por defecto reemplaza lo enmascarado por «...», que el modelo leería como texto, y además intenta "
          "tokenizar la máscara; se le pasó una máscara numérica y luego se vació para que la oración se elimine.",
          "- **IG, Input×Gradient, Rollout y LRP** no tienen variante por oración: producen un valor por token. La única forma de llevarlos a oraciones es **sumar** los tokens de "
          "cada oración, lo que (a) favorece a las oraciones largas, (b) puede cancelar valores positivos y negativos dentro de una oración (IG, IxG) y (c) en Rollout y LRP, que solo "
          "dan valores positivos, convierte la explicación en una medida de longitud. Por eso estos métodos se evalúan **en su nivel nativo (palabra)** y la suma por oración se "
          "informa solo como adaptación, para medir cuánto se pierde.",
          "- **hiiamsid** no es un clasificador: su salida (la similitud media entre oraciones) depende de varias codificaciones independientes. IG e IxG se pudieron adaptar "
          "derivando el coseno medio respecto de los embeddings de todas las oraciones a la vez; Rollout y LRP-Chefer no, porque la atención de cada oración no informa sobre su "
          "similitud con las demás.",
          "- **Modelo sobre Variables:** no es diferenciable (cuenta mayúsculas con expresiones regulares, usa percentiles), así que los métodos de gradiente y atención no aplican. "
          "LIME y SHAP a nivel palabra no se usan: además de su costo (cada evaluación ejecuta tres transformers y spaCy), borran palabras sueltas al azar y "
          "el modelo terminaría midiendo textos agramaticales (sección siguiente a los resultados). Quedan MexGen, LIME-oración y SHAP-oración. "
          "Su explicación global nativa son los coeficientes de la regresión (ver `Reporte_Seleccion_Modelo_sobre_Variables.md`).", ""] + TXT_EJEMPLO + [
          "## 4. Cómo se mide la fidelidad (y cómo se evita que la métrica favorezca a un tipo de método)", "",
          gc.diagrama_md('comparacion_xai.png', REP, 'Protocolo de la comparación de explicabilidad'),
          f"- **Curva de borrado:** se eliminan del texto las unidades más importantes según cada método hasta quitar el {int(meta['rejilla_q'][0] * 100)} %, "
          f"{int(meta['rejilla_q'][1] * 100)} % … {int(meta['rejilla_q'][-1] * 100)} % de las palabras y se mide la caída de la salida del modelo. "
          "Se usa la **fracción de palabras eliminadas** como eje común: así un método por oración y uno por palabra se comparan cuando han quitado la misma cantidad de texto.",
          "- **AOPC** (Samek et al., 2017; DeYoung et al., 2020): promedio de esa caída. Mayor es mejor.",
          "- **Ganancia sobre el azar:** AOPC del método menos AOPC de borrar unidades **al azar de la misma granularidad**. Este control es clave: borrar oraciones completas y borrar "
          "palabras sueltas alteran el texto de forma distinta (las palabras sueltas dejan frases agramaticales, fuera de distribución; Hooker et al., 2019). Sin el control, la "
          "métrica premiaría o castigaría la granularidad y no la explicación.",
          f"- **Suficiencia** (DeYoung et al., 2020): caída al conservar **solo** el {int(meta['q_suficiencia'] * 100)} % más importante. Menor es mejor.",
          "- **Cambio de clase:** proporción de noticias en que la clase predicha cambia al borrar el 20 % más importante.",
          "- **Prueba de Adebayo** (Adebayo et al., 2018), en 15 noticias: se reinician la cabeza y las 3 capas superiores del modelo y se correlaciona la explicación nueva con la "
          "original. Un método fiel debería cambiar (correlación cercana a 0); si no cambia, explica el texto y no el modelo.",
          "- **Acuerdo con MexGen:** correlación de Spearman entre el orden de las oraciones según cada método y según MexGen.", "",
          "Sesgos que quedan: (1) MexGen y la curva de borrado usan la misma operación (quitar oraciones), lo que le da ventaja a MexGen en la versión por oración de la curva; "
          "la ganancia sobre el azar de la misma granularidad reduce, pero no elimina, esa ventaja. (2) LIME y SHAP también se basan en borrar, por lo que comparten esa afinidad; "
          "los métodos de gradiente y atención no. (3) La prueba de Adebayo con reinicio parcial depende de cuántas capas se reinician.", ""]

    for k, o in enumerate(R, 5):
        r = R[o]
        al = r['_aleatorio']
        L += [f"## {k}. Resultados: {NOMBRE_OBJ[o]} ({n_docs[o]} noticias)", "",
              "| Método | Granularidad | AOPC | Ganancia sobre el azar [IC 95 %] | Supera al azar | Suficiencia@20 % ↓ | Cambio de clase@20 % | Acuerdo con MexGen (ρ) | Adebayo (ρ) ↓ | s / noticia | MexGen − método (p Holm) |",
              "|:---|:---|---:|:---:|---:|---:|---:|---:|---:|---:|:---|"]
        for m in [x for x in PALABRA + AGREGADOS + ORACION + ['MexGen-cláusulas'] if x in r]:
            x = r[m]
            comp = '—' if m == 'MexGen' else f"{f(x['mexgen_menos_metodo'])} ({fp(x['p_holm'])})"
            nm = f"**{m}**" if m == 'MexGen' else m
            L.append(f"| {nm} | {NOMBRE_GRAN[x['granularidad']]} | {f(x['aopc'][0])} | {f(x['ganancia'][0])} [{f(x['ganancia'][1][0])}; {f(x['ganancia'][1][1])}] | {pct(x['supera_azar'])} | "
                     f"{f(x['suficiencia'][0])} | {pct(x['cambio_clase']) if x['cambio_clase'] is not None else '—'} | {f(x['acuerdo_mexgen'], 2)} | {f(x['adebayo'], 2)} | "
                     f"{f(x['segundos'], 1)} | {comp} |")
        L.append("| *Azar* | " + " · ".join(f"{NOMBRE_GRAN[g]}: AOPC {f(v['aopc'])}" + (f", cambio de clase {pct(v['cambio_clase'])}" if v['cambio_clase'] is not None else '') for g, v in al.items()) + " | | | | | | | | | |")
        mejores = sorted([m for m in PALABRA + ORACION if m in r], key=lambda m: -r[m]['ganancia'][0])
        por_or = sorted([m for m in ORACION if m in r], key=lambda m: -r[m]['ganancia'][0])
        L += ["", f"Mayor ganancia sobre el azar: **{mejores[0]}** ({f(r[mejores[0]]['ganancia'][0])}), seguido de {mejores[1]} y {mejores[2]}. "
              f"MexGen queda en el puesto {mejores.index('MexGen') + 1} de {len(mejores)} entre todos los métodos nativos y en el {por_or.index('MexGen') + 1} de "
              f"{len(por_or)} entre los métodos por oración (la comparación justa, porque explican la misma unidad).", ""]

    L += [f"![Ganancia de fidelidad]({figs['ganancia']})", "", f"![Curvas de borrado]({figs['curvas']})", "", f"![Prueba de Adebayo]({figs['adebayo']})", "",
          f"![Acuerdo con MexGen]({figs['acuerdo']})", "",
          "Correlación de Spearman entre el orden de las oraciones según cada método y según MexGen, por modelo («no aplica»: el método no se puede usar en ese modelo). "
          "LIME-oración y SHAP-oración, que explican la misma unidad, son los que más coinciden con MexGen.", "",
          f"![Costo frente a ganancia]({figs['costo']})", "",
          f"Cada punto es un método: en horizontal, los segundos por noticia en el equipo de la ejecución (escala logarítmica); en vertical, la ganancia sobre el azar. "
          "MexGen aparece con borde oscuro. Arriba a la derecha quedan los métodos más fieles pero más costosos (LIME y SHAP por palabra); MexGen es el método "
          "por oración más barato.", ""]

    # Limitación de fondo de las perturbaciones por palabra
    L += ["## Limitación de los métodos que perturban palabras sueltas", "",
          "LIME y SHAP a nivel palabra explican a partir de cientos de versiones del texto en las que se borran palabras al azar. Esas versiones dejan de ser "
          "texto bien escrito: se rompen la concordancia, las construcciones verbales y las referencias. Cuando la salida del modelo cambia, no se sabe si cambió "
          "porque la palabra borrada era importante o porque el texto quedó agramatical y fuera de la distribución con la que se entrenó el modelo "
          "(Feng et al., 2018; Hooker et al., 2019; Hase, Xie y Bansal, 2021). La explicación puede entonces reflejar la sensibilidad del modelo a textos rotos y no "
          "lo que lo llevó a su decisión en el texto real.", "",
          "Los propios datos de esta comparación lo muestran: borrar **al azar** palabras sueltas, sin ningún criterio, ya hace caer la salida del modelo más que "
          "borrar al azar la misma cantidad de texto en oraciones completas.", "",
          "| Modelo | Caída media al borrar palabras al azar (AOPC) | Caída media al borrar oraciones al azar (AOPC) | Razón |", "|:---|---:|---:|---:|"]
    for o in R:
        ap, ao = R[o]['_aleatorio']['palabra']['aopc'], R[o]['_aleatorio']['oracion']['aopc']
        L.append(f"| {NOMBRE_OBJ[o]} | {f(ap)} | {f(ao)} | {f(ap / ao, 1) if ao > 0 else '—'} |")
    L += ["", f"![Borrado al azar: palabras frente a oraciones]({figs['azar']})", "",
          "Caída de la salida de cada modelo al borrar al azar palabras sueltas (azul) u oraciones completas (naranja), para la misma fracción de texto eliminada.", ""]
    L += ["", "Parte de la caída que se atribuye a las palabras se debe, por tanto, a haber dañado el texto. Por eso la fidelidad se mide como ganancia sobre el azar "
          "**de la misma granularidad** y no con la caída bruta. MexGen también perturba, pero quita oraciones completas: el texto que queda sigue siendo "
          "gramatical y se parece a una noticia más corta, aunque puede perder alguna referencia (un pronombre que remitía a la oración quitada).", ""]

    # Artefacto de la puntuación en los métodos por palabra
    con_cc = [o for o in R if 'conservando_cierre' in j['documentos'][o][0]]
    if con_cc:
        L += ["## Una «maña» de la evaluación por palabras: la puntuación de cierre", "",
              "Al borrar una palabra se borra también el punto que la acompaña («medidas.»). Si esa palabra cerraba una oración, dos oraciones se fusionan. "
              "Para un modelo que depende de la estructura en oraciones, como la redundancia (similitud media entre oraciones), esto cambia la salida por sí solo: "
              "un método puede «aprender» a señalar las palabras con punto final y obtener una fidelidad alta explicando la segmentación, no el contenido. "
              "Para medirlo se repitió la curva de borrado de los métodos por palabra **conservando la puntuación de cierre** de las palabras borradas, y se contó qué "
              "proporción de las palabras más importantes (el 10 % superior) lleva punto final.", "",
              "| Modelo | Método | Ganancia (borrado normal) | Ganancia (conservando el cierre) | Palabras con punto final en el 10 % más importante | En todo el texto |",
              "|:---|:---|---:|---:|---:|---:|"]
        for o in con_cc:
            docs = j['documentos'][o]
            for m in [x for x in PALABRA if x in docs[0]['conservando_cierre']]:
                g_norm = np.mean([d['metodos'][m]['aopc'] - d['aleatorio']['palabra']['aopc'] for d in docs])
                g_cc = np.mean([d['conservando_cierre'][m]['aopc'] - d['conservando_cierre']['_aleatorio']['aopc'] for d in docs])
                top = np.mean([d['cierre_top10'][m][0] for d in docs])
                todo = np.mean([d['cierre_top10'][m][1] for d in docs])
                L.append(f"| {NOMBRE_OBJ[o]} | {m} | {f(g_norm)} | {f(g_cc)} | {pct(top)} | {pct(todo)} |")
        L += ["", "Si un método concentra las palabras con punto final muy por encima de su proporción en el texto y su ganancia cae al conservar el cierre, su "
              "fidelidad aparente se debe a la segmentación. Esta «maña» no afecta a los métodos por oración, que borran oraciones completas.", ""]

    L += [
          "## Oraciones frente a cláusulas (segmentación de MexGen)", "",
          "| Modelo | Oraciones / noticia | Cláusulas / noticia | Oraciones sin verbo | Cláusulas sin verbo | Ganancia MexGen (oraciones) | Ganancia MexGen (cláusulas) |", "|:---|---:|---:|---:|---:|---:|---:|"]
    for o in R:
        L.append(f"| {NOMBRE_OBJ[o]} | {f(seg[o]['oraciones'], 1)} | {f(seg[o]['clausulas'], 1)} | {pct(seg[o]['or_sin_verbo'])} | {pct(seg[o]['cl_sin_verbo'])} | "
                 f"{f(R[o]['MexGen']['ganancia'][0])} | {f(R[o]['MexGen-cláusulas']['ganancia'][0])} |")
    L += ["", "Las cláusulas son unidades más pequeñas, por lo que una curva de borrado con ellas puede ser más fina; pero una gran parte no contiene un verbo y no expresa una "
          "idea completa, de modo que la explicación resultante es menos legible y menos verificable por un lector.", "",
          "## Síntesis: qué método es mejor y en qué sentido", ""]
    filas = []
    for m in PALABRA + ORACION:
        g = [R[o][m]['ganancia'][0] for o in R if m in R[o]]
        a = [R[o][m]['adebayo'] for o in R if m in R[o] and not np.isnan(R[o][m]['adebayo'])]
        t = [R[o][m]['segundos'] for o in R if m in R[o]]
        filas.append((m, np.mean(g), len(g), np.mean(a) if a else np.nan, np.mean(t)))
    L += ["| Método | Modelos donde aplica | Ganancia media | Adebayo medio (ρ) ↓ | s / noticia | Unidad de la explicación |", "|:---|---:|---:|---:|---:|:---|"]
    for m, g, n, a, t in sorted(filas, key=lambda x: -x[1]):
        L.append(f"| {'**' + m + '**' if m == 'MexGen' else m} | {n} de {len(R)} | {f(g)} | {f(a, 2)} | {f(t, 1)} | {'oración' if m in ORACION else 'palabra'} |")
    L += ["", "Criterios para elegir un método (cada uno se lee en la tabla anterior y en las secciones por modelo):", "",
          "1. **Fidelidad** (ganancia sobre el azar): qué tan bien señala el método lo que de verdad mueve la salida del modelo.",
          "2. **Dependencia del modelo** (Adebayo): un método cuya explicación no cambia al destruir el modelo no está explicando el modelo.",
          f"3. **Costo**: segundos por noticia ({EQUIPO}).",
          "4. **Cobertura**: en cuántos de los cuatro modelos se puede aplicar (MexGen, LIME y SHAP sirven para cualquier modelo; los métodos de gradiente y atención, solo para transformers).",
          "5. **Legibilidad**: una oración es una unidad que una persona puede leer y verificar; una lista de palabras sueltas, no.", ""]
    L += conclusion(R, j, filas)
    L += ["## Limitaciones", "",
          "1. Las métricas de borrado modifican el texto y llevan al modelo fuera de su distribución de entrenamiento; la ganancia sobre el azar de la misma granularidad lo mitiga, pero no lo elimina.",
          "2. La curva de borrado por oraciones comparte operación con MexGen, LIME-oración y SHAP-oración (quitar oraciones completas).",
          "3. LIME y SHAP se ejecutaron con un número limitado de perturbaciones por costo; con más perturbaciones su fidelidad podría mejorar.",
          "4. La prueba de Adebayo se hizo en 15 noticias por modelo y con reinicio de las 3 capas superiores.",
          "5. La muestra excluye noticias de más de " + str(meta['max_palabras']) + " palabras para que todos los métodos vean el texto completo.", "",
          "## Referencias", "",
          "- Abnar, S. y Zuidema, W. (2020). Quantifying Attention Flow in Transformers. *ACL 2020*.",
          "- Adebayo, J. et al. (2018). Sanity Checks for Saliency Maps. *NeurIPS 2018*.",
          "- Chefer, H., Gur, S. y Wolf, L. (2021). Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers. *ICCV 2021*.",
          "- DeYoung, J. et al. (2020). ERASER: A Benchmark to Evaluate Rationalized NLP Models. *ACL 2020*.",
          "- Feng, S. et al. (2018). Pathologies of Neural Models Make Interpretations Difficult. *EMNLP 2018*.",
          "- Hase, P., Xie, H. y Bansal, M. (2021). The Out-of-Distribution Problem in Explainability and Search Methods for Feature Importance Explanations. *NeurIPS 2021*.",
          "- Hooker, S. et al. (2019). A Benchmark for Interpretability Methods in Deep Neural Networks. *NeurIPS 2019*.",
          "- Lundberg, S. y Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS 2017*.",
          "- Monteiro Paes, L., Wei, D. et al. (2025). Multi-Level Explanations for Generative Language Models. *ACL 2025*, pp. 32291–32317.",
          "- Ribeiro, M. T., Singh, S. y Guestrin, C. (2016). «Why Should I Trust You?»: Explaining the Predictions of Any Classifier. *KDD 2016*.",
          "- Samek, W. et al. (2017). Evaluating the Visualization of What a Deep Neural Network Has Learned. *IEEE TNNLS*, 28(11).",
          "- Shrikumar, A. et al. (2017). Learning Important Features Through Propagating Activation Differences. *ICML 2017*.",
          "- Sundararajan, M., Taly, A. y Yan, Q. (2017). Axiomatic Attribution for Deep Networks. *ICML 2017*.", ""]
    salida = REP / 'Reporte_Comparativo_XAI_Literatura_vs_Propia.md'
    salida.write_text("\n".join(L) + "\n", encoding='utf-8')
    print(f"[OK] {salida}")


if __name__ == '__main__':
    main()
