"""
Recalcula los 4 casos de estudio (2 falsas y 2 verdaderas) con el IML y los estilos discursivos vigentes.
Entrada: codigos/casos_estudio_4_noticias.json. Salidas: reportes/casos_estilo/*.json,
reportes/Reporte_Casos_Estilos_Discursivos_4_Noticias.md y graficos/estilos/casos_*.png.
Cada caso se lee con los tres pilares del marco: estilo (cómo está escrito), Modelo sobre Variables (perfil) y
MexGen por oraciones (qué oración lo provoca).

Versión anterior de los casos. La versión final de la tesis (Tabla 4.20, textos originales del corpus y probabilidad
fuera de pliegue) se genera con Analisis_complementarios/ejecutar_casos_originales.py, que reutiliza las funciones de
este script (figura, figura_radar, figura_mexgen) y escribe en Analisis_complementarios/casos_estilo/.
"""

import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config_proyecto.py").exists())))
import config_proyecto as cfg
sys.path.insert(0, str(Path(__file__).resolve().parent))
from auditor_estilo import AuditorEstilo, informe_markdown
from estilos_discursivos import DIMENSIONES, dibujar_radar
import graficos_comunes as gc

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

SALIDA = cfg.EXPLICABILIDAD_REPORTES_DIR / 'casos_estilo'
FIG = cfg.EXPLICABILIDAD_GRAFICOS_DIR / 'estilos' / 'casos_dimensiones_iml.png'
FIG_RADAR = cfg.EXPLICABILIDAD_GRAFICOS_DIR / 'estilos' / 'casos_radar_estilo.png'
FIG_MEXGEN = cfg.EXPLICABILIDAD_GRAFICOS_DIR / 'estilos' / 'casos_mexgen_oraciones.png'


def es_falsa(c):
    return 'Falsa' in c['tipo_referencia']


def f(x, d=1):
    return '—' if x is None else f"{x:.{d}f}".replace('.', ',')


def fs(x, d=1):
    x = 0.0 if round(x, d) == 0 else x
    return (f"{x:+.{d}f}").replace('.', ',').replace('-', '−')


def figura_mexgen(casos, res):
    """Barras por oración: ΔIML (pilar 1) y ΔP(falsa) (pilar 2), una fila por caso."""
    plt = gc.pyplot()
    filas = [(c, r) for c, r in zip(casos, res) if r['mexgen'] and r['mexgen']['aplica']]
    alto = [max(2.0, 0.42 * r['mexgen']['n_oraciones'] + 0.9) for _, r in filas]
    fig, axes = plt.subplots(len(filas), 2, figsize=(12, sum(alto)), gridspec_kw={'height_ratios': alto}, squeeze=False)
    lim_i = max(abs(x['delta_iml']) for _, r in filas for x in r['mexgen']['oraciones']) * 1.35 + 0.5
    lim_p = max(abs(x['delta_p_falsa']) for _, r in filas for x in r['mexgen']['oraciones']) * 1.35 + 0.01
    for (c, r), (ax_i, ax_p) in zip(filas, axes):
        o = r['mexgen']['oraciones']
        y = np.arange(len(o))[::-1]
        di = np.array([x['delta_iml'] for x in o])
        dp = np.array([x['delta_p_falsa'] for x in o])
        col_e = gc.ESTILOS[r['perfil']['indice_estilo'] + 1]
        ax_i.barh(y, di, 0.62, color=[col_e if v >= 0 else gc.TINTA_SUAVE for v in di])
        ax_p.barh(y, dp, 0.62, color=[gc.CLASE[1] if v >= 0 else gc.CLASE[0] for v in dp])
        for ax, vals, lim, d in [(ax_i, di, lim_i, 1), (ax_p, dp, lim_p, 3)]:
            ax.axvline(0, color=gc.EJE, linewidth=1)
            ax.set_xlim(-lim, lim)
            ax.set_yticks(y)
            ax.set_yticklabels([f"O{x['indice']}" for x in o], fontsize=8.5)
            ax.grid(axis='y', visible=False)
            for yy, v in zip(y, vals):
                ax.text(v + (lim * 0.02 if v >= 0 else -lim * 0.02), yy, fs(v, d), va='center', ha='left' if v >= 0 else 'right',
                        fontsize=7.5, color=gc.TINTA_2)
        n = c['id'].split('_')[1]
        ax_i.set_title(f"{n}. {'Falsa' if es_falsa(c) else 'Verdadera'} · {r['perfil']['estilo']} · IML {r['perfil']['iml']:.0f}", fontsize=9)
        ax_p.set_title(f"{n}. P(falsa) = {f(r['modelo_sobre_variables']['p_falsa'], 2)} → {r['modelo_sobre_variables']['prediccion']}", fontsize=9)
        gc.coma(ax_i, 'x')
        gc.coma(ax_p, 'x')
    axes[-1][0].set_xlabel('ΔIML (puntos) · > 0: la oración sube el IML')
    axes[-1][1].set_xlabel('ΔP(falsa) · > 0: la oración empuja hacia «falsa»; < 0: hacia «real»')
    fig.suptitle('MexGen por oraciones en los 4 casos: cuánto cambia cada salida al quitar cada oración (O1, O2… en orden de aparición)',
                 fontsize=10, x=0.01, ha='left', color=gc.TINTA)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    gc.guardar(fig, FIG_MEXGEN)


def lectura_tres_pilares(r):
    """Párrafo que lee juntos el estilo, el perfil del clasificador y la oración que lo provoca."""
    p, ms, mg = r['perfil'], r['modelo_sobre_variables'], r['mexgen']
    nombres = {d: n for d, (n, _, _) in DIMENSIONES.items()}
    altas = sorted(((d, v) for d, v in p['dimensiones'].items() if v is not None and not np.isnan(v)), key=lambda x: -x[1])[:2]
    var = [c for c in ms['principales_contribuciones'] if c['empuja_hacia'] == ms['prediccion']][:2]
    t = (f"**Lectura de los tres pilares.** *Cómo está escrito:* {p['estilo'].split(': ', 1)[-1]} ({f(p['pertenencia_estilo'], 0)} % de pertenencia), "
         f"IML {f(p['iml'])}, con las dimensiones más altas en {nombres[altas[0][0]].lower()} ({f(altas[0][1], 0)}) y {nombres[altas[1][0]].lower()} ({f(altas[1][1], 0)}). "
         f"*Perfil:* el Modelo sobre Variables da P(falsa) = {f(ms['p_falsa'], 2)} (umbral {f(ms['umbral'], 2)}), es decir, perfil de noticia {ms['prediccion']}"
         + (f"; lo empujan sobre todo {' y '.join('`' + c['variable'] + '`' for c in var)}." if var else "."))
    if not mg or not mg['aplica']:
        return t + " *Qué oración:* no aplica (una sola oración)."
    o = {x['indice']: x for x in mg['oraciones']}
    e = mg['extremos']
    clave = 'hacia_falsa' if ms['prediccion'] == 'falsa' else 'hacia_real'

    def cita(i):
        s = ' '.join(o[i]['texto'].split())
        return f"la oración {i} («{s if len(s) <= 90 else s[:87] + '…'}»)"
    t += (f" *Qué oración:* {cita(e['sube_iml']['indice'])} es la que más sostiene el IML (ΔIML {fs(e['sube_iml']['delta'])}), y "
          + (f"la misma" if e[clave]['indice'] == e['sube_iml']['indice'] else cita(e[clave]['indice']))
          + f" es la que más sostiene la lectura «{ms['prediccion']}» del clasificador (ΔP(falsa) {fs(e[clave]['delta'], 3)}).")
    return t


def figura(casos, res):
    plt = gc.pyplot()
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.6), sharey=True)
    dims = list(DIMENSIONES)
    for ax, c, r in zip(axes, casos, res):
        vals = [r['perfil']['dimensiones'][d] for d in dims]
        ax.bar(range(5), vals, 0.65, color=gc.CLASE[1 if es_falsa(c) else 0], edgecolor='white', linewidth=1)
        ax.axhline(50, color=gc.EJE, linewidth=0.9, linestyle=':')
        for i, v in enumerate(vals):
            ax.text(i, v + 2, f'{v:.0f}', ha='center', fontsize=8, color=gc.TINTA)
        ax.set_xticks(range(5))
        ax.set_xticklabels(dims, fontsize=8.5)
        ax.set_ylim(0, 108)
        ax.grid(axis='x', visible=False)
        ax.set_title(f"{c['id'].split('_')[1]}. {'Falsa' if es_falsa(c) else 'Verdadera'} · IML {r['perfil']['iml']:.0f}\n"
                     f"{r['perfil']['estilo']} ({r['perfil']['pertenencia_estilo']:.0f} %)", fontsize=8.5)
    axes[0].set_ylabel('Percentil (0–100)')
    fig.suptitle('Dimensiones del IML en los 4 casos (D1 sensacionalismo · D2 emoción · D3 reiteración · D4 énfasis ortotipográfico · D5 aserción sin fuentes)',
                 fontsize=9.5, x=0.01, ha='left', color=gc.TINTA)
    fig.tight_layout()
    gc.guardar(fig, FIG)


def figura_radar(casos, res, ref):
    """Radar de cada caso frente al centro de su estilo dominante."""
    plt = gc.pyplot()
    fig = plt.figure(figsize=(14, 4.9))
    for k, (c, r) in enumerate(zip(casos, res)):
        ax = fig.add_subplot(1, 4, k + 1, projection='polar')
        i = r['perfil']['indice_estilo']
        dibujar_radar(ax, [
            {'valores': ref['centros'][i], 'color': gc.ESTILOS[i + 1], 'estilo_linea': '--', 'relleno': True},
            {'valores': np.nan_to_num([r['perfil']['dimensiones'][d] for d in DIMENSIONES], nan=50.0),   # D3 falta con una sola oración
             'color': gc.TINTA, 'relleno': False}],
            titulo=f"{c['id'].split('_')[1]}. {'Falsa' if es_falsa(c) else 'Verdadera'} · IML {r['perfil']['iml']:.0f}\n"
                   f"{r['perfil']['estilo']} ({r['perfil']['pertenencia_estilo']:.0f} %)", tam_etiquetas=7.5)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=gc.TINTA, linewidth=2, label='Noticia'),
                        *[Line2D([], [], color=gc.ESTILOS[i + 1], linewidth=2, linestyle='--', label=f"Centro del {n}")
                          for i, n in enumerate(ref['nombres_estilos'])]],
               loc='lower center', ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Cada caso frente al centro de su estilo dominante (percentil 0–100; línea punteada gris = mediana del corpus)',
                 fontsize=10, x=0.01, ha='left', color=gc.TINTA)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    gc.guardar(fig, FIG_RADAR)


def main():
    casos = json.loads((Path(__file__).resolve().parent / 'casos_estudio_4_noticias.json').read_text(encoding='utf-8'))
    aud = AuditorEstilo()
    SALIDA.mkdir(parents=True, exist_ok=True)
    res = []
    import time
    for c in casos:
        t0 = time.time()
        r = aud.auditar(c['texto'], c['titulo'])
        r['segundos_auditoria'] = round(time.time() - t0, 1)
        r.update({k: c[k] for k in ['id', 'tipo_referencia', 'origen', 'en_corpus']})
        (SALIDA / f"{c['id']}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding='utf-8')
        res.append(r)
        print(f"  {c['id']:<45} IML={r['perfil']['iml']:5.1f} | {r['perfil']['estilo']} ({r['perfil']['pertenencia_estilo']} %) | "
              f"Modelo sobre Variables: P(falsa)={r['modelo_sobre_variables']['p_falsa']} | {r['segundos_auditoria']} s", flush=True)
    ref = json.loads(cfg.REFERENCIA_ESTILOS_JSON.read_text(encoding='utf-8'))
    figura(casos, res)
    figura_radar(casos, res, ref)
    figura_mexgen(casos, res)
    L = ["# Casos de estudio: los tres pilares del marco en 4 noticias", "",
         "**Generado por:** `codigos/ejecutar_casos_estilo.py` (auditor: `codigos/auditor_estilo.py`) · datos en `reportes/casos_estilo/`  ",
         "**Construcción del IML y de los estilos:** [`Reporte_Construccion_Estilos_Discursivos.md`](Reporte_Construccion_Estilos_Discursivos.md)", "",
         "> Los estilos describen la **forma** del texto, no su veracidad. La lectura del Modelo sobre Variables se muestra aparte y es la de un "
         "clasificador con F1 macro de 0,63 en validación cruzada: no es un veredicto.", "",
         gc.diagrama_md('auditoria_texto_nuevo.png', cfg.EXPLICABILIDAD_REPORTES_DIR, 'Qué ocurre al auditar un texto'),
         "## Resumen", "",
         "| Caso | Referencia | IML | D1 | D2 | D3 | D4 | D5 | Estilo dominante (pertenencia) | Segundo estilo | Modelo sobre Variables: P(falsa) |",
         "|:---|:---|---:|---:|---:|---:|---:|---:|:---|:---|---:|"]
    for r in res:
        p, d = r['perfil'], r['perfil']['dimensiones']
        L.append(f"| {r['titulo']} | {r['tipo_referencia']} | **{f(p['iml'])}** | " + " | ".join(f(d[k], 0) for k in DIMENSIONES) +
                 f" | {p['estilo']} ({f(p['pertenencia_estilo'], 0)} %) | {p['estilo_secundario'].split(':')[0]} ({f(p['pertenencia_secundario'], 0)} %) | "
                 f"{f(r['modelo_sobre_variables']['p_falsa'], 2)} → {r['modelo_sobre_variables']['prediccion']} |")
    L += ["", f"![Dimensiones del IML en los 4 casos](../graficos/estilos/{FIG.name})", "",
          "D1 sensacionalismo · D2 carga emocional · D3 reiteración · D4 énfasis ortotipográfico · D5 aserción sin respaldo. "
          "Cada valor es un percentil respecto del corpus (50 = mediana).", "",
          f"![Cada caso frente al centro de su estilo](../graficos/estilos/{FIG_RADAR.name})", "",
          "En el radar, la línea negra es la noticia y la discontinua de color el centro de su estilo dominante: cuanto más se parecen las dos "
          "figuras, mayor es la pertenencia.", "",
          "Estilos del corpus:", ""] + [f"- **{n}**: {l}" for n, l in zip(ref['nombres_estilos'], ref['lemas_estilos'])] + [""]
    L += ["### MexGen por oraciones (pilar 3)", "",
          "Cada noticia se divide en oraciones con Punkt (la misma segmentación de la comparación de explicabilidad) y se quita una oración por vez. "
          "Para cada versión se recalculan las 43 variables y, con ellas, el IML y la P(falsa) del Modelo sobre Variables: "
          "Δᵢ = valor con el texto completo − valor sin la oración i. Un Δ positivo indica que la oración sostiene el valor; uno negativo, que lo reduce. "
          "Una misma extracción de variables sirve para los dos pilares (k + 1 extracciones para k oraciones).", "",
          f"![MexGen por oraciones en los 4 casos](../graficos/estilos/{FIG_MEXGEN.name})", "",
          "A la izquierda, ΔIML (en color del estilo dominante cuando la oración sube el IML, en gris cuando lo baja); a la derecha, ΔP(falsa) "
          "(rojo: la oración empuja hacia «falsa»; azul: hacia «real»). El ΔIML está en puntos de percentil (0–100) y el ΔP(falsa) en probabilidad, "
          "así que las dos escalas no se comparan entre sí; dentro de cada panel, la barra más larga es la oración que más pesa.", "",
          "| Caso | Oraciones | Oración que más sube el IML (ΔIML) | Oración que más sostiene la lectura del clasificador (ΔP(falsa)) | Segundos |",
          "|:---|---:|:---|:---|---:|"]
    for r in res:
        mg, ms = r['mexgen'], r['modelo_sobre_variables']
        if mg and mg['aplica']:
            e = mg['extremos']
            k = 'hacia_falsa' if ms['prediccion'] == 'falsa' else 'hacia_real'
            L.append(f"| {r['titulo']} | {mg['n_oraciones']} | O{e['sube_iml']['indice']} ({fs(e['sube_iml']['delta'])}) | "
                     f"O{e[k]['indice']} ({fs(e[k]['delta'], 3)}, hacia «{ms['prediccion']}») | {f(r['segundos_auditoria'])} |")
        else:
            L.append(f"| {r['titulo']} | 1 | no aplica | no aplica | {f(r['segundos_auditoria'])} |")
    L.append("")
    for i, r in enumerate(res, 1):
        L += [f"## {i}. {r['titulo']}", "", f"**Referencia:** {r['tipo_referencia']} · **Origen:** {r['origen']}"
              + (" · *(el texto forma parte del corpus de referencia)*" if r['en_corpus'] else ""), "",
              lectura_tres_pilares(r), ""]
        L += informe_markdown(r)
    L += ["## Notas", "",
          "- El caso 1 forma parte del corpus de referencia; la regresión logística se ajustó con ese corpus, por lo que su P(falsa) en ese caso no es una predicción fuera de muestra.",
          "- Los casos se eligieron en la versión anterior de la tesis como ejemplos ilustrativos; no son una muestra representativa.",
          "- La lectura de los tres pilares describe el estilo del texto y lo que mueve al clasificador; no verifica los hechos. "
          "El Modelo sobre Variables tiene F1 macro de 0,63 en validación cruzada.",
          "- MexGen quita oraciones completas: el texto que queda sigue siendo gramatical, pero puede perder alguna referencia (un pronombre que remitía a "
          "la oración quitada). Su fidelidad frente a otros métodos está en `XAI_Literatura_Comparacion/reportes/Reporte_Comparativo_XAI_Literatura_vs_Propia.md`.", ""]
    salida = cfg.EXPLICABILIDAD_REPORTES_DIR / 'Reporte_Casos_Estilos_Discursivos_4_Noticias.md'
    salida.write_text("\n".join(L) + "\n", encoding='utf-8')
    print(f"[OK] {salida}")


if __name__ == '__main__':
    main()
