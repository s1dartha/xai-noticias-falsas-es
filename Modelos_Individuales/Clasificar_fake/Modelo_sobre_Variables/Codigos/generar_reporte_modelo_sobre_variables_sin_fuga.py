"""
Genera Reportes/Reporte_Modelo_sobre_Variables_Sin_Fuga.md a partir de Reportes/metricas_modelo_sobre_variables_sin_fuga.json y de
Dataset/tabla_variables_4418_meta.json. Todas las cifras se leen de esos archivos; no hay valores escritos a mano.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config_proyecto.py").exists())))
import config_proyecto as cfg
import graficos_comunes as gc


def f(x, d=3):
    return '—' if x is None else f"{x:.{d}f}".replace('.', ',')


def pct(x, d=1):
    return f"{x * 100:.{d}f}".replace('.', ',') + " %"


def ic(v):
    return f"[{f(v[0])}; {f(v[1])}]"


ENC = "| Configuración | F1 macro [IC 95 %] | Exactitud | Precisión (falsa) | Recall (falsa) | Especificidad | FP | FN | Tasa FP | Tasa FN | ROC-AUC |"
SEP = "|:---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"


def fila(nombre, m, negrita=False):
    n = f"**{nombre}**" if negrita else nombre
    return (f"| {n} | {f(m['f1_macro'])} {ic(m['f1_macro_ic95'])} | {pct(m['exactitud'])} | {pct(m['precision_falsa'])} | {pct(m['recall_falsa'])} | "
            f"{pct(m['especificidad'])} | {m['FP']} | {m['FN']} | {pct(m['tasa_FP'])} | {pct(m['tasa_FN'])} | {f(m['roc_auc'])} |")


def dif(d):
    signo = "mejor" if d['ic95'][0] > 0 else ("peor" if d['ic95'][1] < 0 else "sin diferencia clara")
    return f"{f(d['dif_f1_macro'])} {ic(d['ic95'])} | {d['dif_FP']:+d} | {d['dif_FN']:+d} | {signo}"


IMG = cfg.MODELO_VARIABLES_IMG_DIR / 'sin_fuga'
# Configuraciones que se comparan en las figuras de curvas, regiones y generalización (color fijo por configuración)
COMPARADAS = ['SaBERT', 'S+R+E (tres pilares)', 'Completo (S+R+E+O+L)', 'Completo + SaBERT']
CORTO_FUENTE = {'FakeNewsCorpusSpanish': 'MEX-A3T', 'spanish-fake-news-fixed': 'Edds', 'Unified-and-Balanced-Spanish-Fake-News-Corpus': 'Unified',
                'Fake-news-latam-omdena': 'Omdena', 'fake_news_corpus_spanish': 'FakeDeS'}


def figuras(j):
    """Figuras del reporte. Devuelve rutas relativas a Reportes/."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
    sys.path.insert(0, str(cfg.ROOT))
    import graficos_comunes as gc
    plt = gc.pyplot()
    # En las figuras, «S+R+E (tres pilares)» se muestra solo como «S+R+E»
    j = json.loads(json.dumps(j, ensure_ascii=False).replace(' (tres pilares)', ''))
    COMPARADAS = [c.replace(' (tres pilares)', '') for c in globals()['COMPARADAS']]
    cv, P = j['validacion_cruzada'], j['metadata']['modelo_principal']
    color = {c: gc.SERIE[i] for i, c in enumerate(COMPARADAS)}
    out = {}

    # 1. F1 macro global con IC por configuración (grupos solos en gris, combinaciones en azul, SaBERT en naranja)
    conf = [c for c in cv if c != 'SaBERT (umbral nativo 0,5)']
    conf = sorted(conf, key=lambda c: cv[c]['Global']['f1_macro'])
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for i, c in enumerate(conf):
        g = cv[c]['Global']
        col = gc.TINTA_SUAVE if len(c.split(' (')[0]) == 1 else (gc.SERIE[1] if 'SaBERT' in c else gc.SERIE[0])
        ax.barh(i, g['f1_macro'], color=col, height=0.62)
        ax.errorbar(g['f1_macro'], i, xerr=[[g['f1_macro'] - g['f1_macro_ic95'][0]], [g['f1_macro_ic95'][1] - g['f1_macro']]],
                    color=gc.TINTA, capsize=3, lw=1)
        ax.text(g['f1_macro_ic95'][1] + 0.01, i, f(g['f1_macro']), va='center', fontsize=8.5, color=gc.TINTA_2)
    ax.axvline(0.5, color=gc.TINTA_SUAVE, ls=':', lw=1)
    ax.text(0.5, len(conf) - 0.4, ' azar', fontsize=8, color=gc.TINTA_SUAVE)
    ax.set_yticks(range(len(conf)), conf)
    ax.set_xlim(0.4, 0.9)
    gc.coma(ax, 'x', 1)
    ax.set_xlabel('F1 macro (validación cruzada agrupada, IC 95 %)')
    ax.grid(axis='y', visible=False)
    ax.set_title('F1 macro por configuración: grupos solos (gris), combinaciones (azul) y con SaBERT (naranja)')
    out['f1'] = gc.guardar(fig, IMG / 'f1_macro_por_configuracion.png')

    # 2. Falsos positivos y falsos negativos por configuración
    fig, ax = plt.subplots(figsize=(8, 4.4))
    y = np.arange(len(conf))
    fp_ = [cv[c]['Global']['FP'] for c in conf]
    fn_ = [cv[c]['Global']['FN'] for c in conf]
    ax.barh(y - 0.18, fp_, height=0.34, color=gc.SERIE[6], label='Falsos positivos (real marcada como falsa)')
    ax.barh(y + 0.18, fn_, height=0.34, color=gc.SERIE[1], label='Falsos negativos (falsa que pasa como real)')
    for yi, a, b in zip(y, fp_, fn_):
        ax.text(a + 8, yi - 0.18, str(a), va='center', fontsize=7.5, color=gc.TINTA_2)
        ax.text(b + 8, yi + 0.18, str(b), va='center', fontsize=7.5, color=gc.TINTA_2)
    ax.set_yticks(y, conf)
    ax.set_xlabel('Número de errores (sobre los textos únicos)')
    ax.set_xlim(0, max(fp_ + fn_) * 1.15)
    ax.grid(axis='y', visible=False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.4, -0.13), ncol=2, fontsize=8.5)
    ax.set_title('Falsos positivos y falsos negativos por configuración')
    out['errores'] = gc.guardar(fig, IMG / 'falsos_positivos_negativos.png')

    # 3. Curvas ROC y PR con las probabilidades fuera de pliegue
    pr = pd.read_csv(cfg.PREDICCIONES_MODELO_VARIABLES_SIN_FUGA_CSV)
    pr.columns = [c.replace(' (tres pilares)', '') for c in pr.columns]
    yv = pr['label_num'].values
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    for c in COMPARADAS:
        p = pr[f'p__{c}'].values
        fpr, tpr, _ = roc_curve(yv, p)
        pre, rec, _ = precision_recall_curve(yv, p)
        a1.plot(fpr, tpr, color=color[c], label=f"{c} (AUC {f(roc_auc_score(yv, p))})")
        a2.plot(rec, pre, color=color[c], label=f"{c} (AP {f(average_precision_score(yv, p))})")
    a1.plot([0, 1], [0, 1], color=gc.TINTA_SUAVE, ls=':', lw=1)
    a2.axhline(yv.mean(), color=gc.TINTA_SUAVE, ls=':', lw=1)
    a1.set(xlabel='Tasa de falsos positivos', ylabel='Tasa de verdaderos positivos (falsas detectadas)', title='Curva ROC')
    a2.set(xlabel='Recall (falsas detectadas)', ylabel='Precisión', title='Curva precisión–recall')
    for a in (a1, a2):
        a.legend(fontsize=7.5, loc='lower right' if a is a1 else 'lower left')
        gc.coma(a, 'x', 1)
        gc.coma(a, 'y', 1)
    out['curvas'] = gc.guardar(fig, IMG / 'curvas_roc_pr.png')

    # 4. F1 macro por región y subconjunto
    regiones = ['España', 'América Latina', 'América Latina: Omdena', 'América Latina: sin Omdena']
    fig, ax = plt.subplots(figsize=(9, 3.8))
    w = 0.8 / len(COMPARADAS)
    for k, c in enumerate(COMPARADAS):
        v = [cv[c][r]['f1_macro'] for r in regiones]
        x = np.arange(len(regiones)) + (k - (len(COMPARADAS) - 1) / 2) * w
        ax.bar(x, v, width=w * 0.92, color=color[c], label=c)
        for xi, vi in zip(x, v):
            ax.text(xi, vi + 0.01, f(vi, 2), ha='center', fontsize=7, color=gc.TINTA_2)
    ax.axhline(0.5, color=gc.TINTA_SUAVE, ls=':', lw=1)
    ax.set_xticks(range(len(regiones)), [f"{r.replace(': ', ':\n')}\n(n = {cv[P][r]['n']})" for r in regiones])
    ax.set_ylim(0, 1.05)
    gc.coma(ax, 'y', 1)
    ax.set_ylabel('F1 macro')
    ax.grid(axis='x', visible=False)
    ax.legend(fontsize=8, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.27))
    ax.set_title('F1 macro por región (validación cruzada)')
    out['regiones'] = gc.guardar(fig, IMG / 'f1_macro_por_region.png')

    # 5. Generalización: transferencia entre regiones y exclusión de una fuente
    esc = [(e, j['transferencia'][e]) for e in j['transferencia']] + \
          [(f"Sin ver {CORTO_FUENTE.get(e, e)}", j['exclusion_fuente'][e]) for e in j['exclusion_fuente']]
    fig, ax = plt.subplots(figsize=(9.5, 4))
    for k, c in enumerate(COMPARADAS):
        v = [r[c]['f1_macro'] for _, r in esc]
        x = np.arange(len(esc)) + (k - (len(COMPARADAS) - 1) / 2) * w
        ax.bar(x, v, width=w * 0.92, color=color[c], label=c)
    ax.axhline(0.5, color=gc.TINTA_SUAVE, ls=':', lw=1)
    ax.axvline(1.5, color=gc.EJE, lw=1)
    ax.set_xticks(range(len(esc)), [e.replace(' → ', '\n→ ') for e, _ in esc], fontsize=8.5)
    ax.set_ylim(0, 1)
    gc.coma(ax, 'y', 1)
    ax.set_ylabel('F1 macro en la parte no vista')
    ax.grid(axis='x', visible=False)
    ax.legend(fontsize=8, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.2))
    ax.set_title('Generalización: transferencia entre regiones (izquierda) y fuente excluida (derecha)')
    out['generalizacion'] = gc.guardar(fig, IMG / 'generalizacion_region_fuente.png')
    return {k: 'Imagenes/sin_fuga/' + v.name for k, v in out.items()}


def main():
    j = json.load(open(cfg.METRICAS_MODELO_VARIABLES_SIN_FUGA_JSON, encoding='utf-8'))
    figs = figuras(j)
    tv = json.load(open(cfg.TABLA_VARIABLES_META_JSON, encoding='utf-8'))
    m, cv, comp = j['metadata'], j['validacion_cruzada'], j['comparaciones_pareadas']
    P = m['modelo_principal']
    G, NG = m['grupos_variables'], m['nombres_grupos']

    L = ["# Modelo sobre Variables explicable sin fuga de información", "",
         "**Generado por:** `generar_reporte_modelo_sobre_variables_sin_fuga.py` a partir de `Reportes/metricas_modelo_sobre_variables_sin_fuga.json`  ",
         "**Evaluación:** `evaluar_modelo_sobre_variables_sin_fuga.py` · **Variables:** `construir_tabla_variables.py` → `Dataset/tabla_variables_4418.csv`  ",
         "**Etiqueta:** 1 = falsa (clase positiva), 0 = real", "",
         gc.diagrama_md('evaluacion_sin_fuga.png', cfg.MODELO_VARIABLES_REPORTES_DIR, 'Esquema de la evaluación sin fuga'),
         "## 1. Diseño", "",
         f"- **Datos:** {m['n_noticias']} noticias; al deduplicar por texto quedan **{m['n_textos_unicos']} textos únicos** "
         f"({m['falsas']} falsas, {m['reales']} reales), que son la unidad de evaluación. Ningún texto está a la vez en entrenamiento y prueba.",
         f"- **Validación:** {m['validacion']}.",
         "- **Sin fuga:** el modelo, el tratamiento de valores faltantes y el umbral se aprenden solo con el entrenamiento de cada pliegue. "
         "No hay imputaciones, cortes ni umbrales calculados sobre el corpus completo. Los hiperparámetros están fijos y no se ajustaron.",
         "- **Un solo punto de operación:** cada configuración usa su umbral anidado. SaBERT recibe el mismo procedimiento; su umbral nativo de 0,5 se muestra solo como referencia.",
         f"- **Clasificador:** {m['nombre_modelo']} ({m['descripcion_modelo']}), elegido entre tres familias en "
         "[`Reporte_Seleccion_Modelo_sobre_Variables.md`](Reporte_Seleccion_Modelo_sobre_Variables.md). Todas las configuraciones entrenadas usan este clasificador.", "",
         "### Variables", "", "| Grupo | Variables |", "|:---|:---|"]
    for k, v in G.items():
        L.append(f"| **{k}** · {NG[k]} | " + ", ".join(f"`{x}`" for x in v) + " |")
    vr = tv['verificacion_variables_reutilizadas']
    L += ["", f"Las variables S y R se reutilizan de la extracción anterior tras comprobar, en {vr['n_noticias']} noticias, que coinciden con un recálculo "
          "con la revisión fija de cada modelo (diferencia absoluta máxima: " + ", ".join(f"`{k}` {v:.1e}" for k, v in vr['diferencia_absoluta_maxima'].items()) +
          "). Las emociones se calculan oración por oración con RoBERTuito y se agregan por noticia (media y máximo). "
          "La redundancia se usa como similitud continua, sin umbrales.", "",
          "## 2. Resultados globales (validación cruzada)", "", ENC, SEP]
    for nombre, r in cv.items():
        L.append(fila(nombre, r['Global'], negrita=nombre in (P, 'SaBERT')))
    L += ["", "Las filas S, R, E, O y L usan un único grupo de variables (ablación). Los umbrales por pliegue de cada configuración están en el JSON.", "",
          f"![F1 macro por configuración]({figs['f1']})", "",
          "F1 macro de cada configuración con su intervalo de confianza: en gris, un solo grupo de variables; en azul, las combinaciones; en naranja, las que usan SaBERT.", "",
          f"![Falsos positivos y negativos]({figs['errores']})", "",
          "Número de noticias reales marcadas como falsas (falsos positivos) y de falsas que pasan como reales (falsos negativos) en cada configuración.", "",
          f"![Curvas ROC y PR]({figs['curvas']})", "",
          "Curvas ROC y precisión–recall con las probabilidades fuera de pliegue (cada texto se predice con un modelo que no lo vio). "
          "La línea punteada es el azar.", "",
          "## 3. Por región", "",
          f"![F1 macro por región]({figs['regiones']})", "",
          "F1 macro de las cuatro configuraciones principales en cada región y en América Latina con y sin Omdena.", ""]
    for region in ['España', 'América Latina', 'América Latina: Omdena', 'América Latina: sin Omdena']:
        n = cv[P][region]
        L += [f"### {region} (n = {n['n']}: {n['falsas']} falsas, {n['reales']} reales)", "", ENC, SEP]
        for nombre in ['SaBERT', 'SaBERT (umbral nativo 0,5)', 'S+R+E (tres pilares)', P, 'Completo + SaBERT']:
            L.append(fila(nombre, cv[nombre][region], negrita=nombre == P))
        L.append("")
    L += ["### Por fuente (modelo principal frente a SaBERT)", "",
          "| Fuente | n | F1 macro principal | F1 macro SaBERT | FP principal / SaBERT | FN principal / SaBERT |", "|:---|---:|---:|---:|---:|---:|"]
    for fu, r in cv[P]['por_fuente'].items():
        s = cv['SaBERT']['por_fuente'][fu]
        L.append(f"| {fu} | {r['n']} | {f(r['f1_macro'])} | {f(s['f1_macro'])} | {r['FP']} / {s['FP']} | {r['FN']} / {s['FN']} |")

    L += ["", "## 4. Diferencias pareadas (bootstrap)", "",
          "Diferencia de F1 macro (A − B) con IC 95 %, y diferencia en el número de falsos positivos y falsos negativos (negativo = A comete menos errores).", "",
          "| Comparación | Dif. F1 macro [IC 95 %] | Dif. FP | Dif. FN | Lectura |", "|:---|:---:|---:|---:|:---|"]
    for k, d in comp.items():
        L.append(f"| {k.replace(' | ', ' · ')} | {dif(d)} |")

    for titulo, clave in [("## 5. Transferencia entre regiones", 'transferencia'), ("## 6. Exclusión de una fuente por vez", 'exclusion_fuente')]:
        L += ["", titulo, "",
              "Se entrena con una parte y se evalúa en la otra; el umbral se elige solo con la parte de entrenamiento." if clave == 'transferencia'
              else "Se entrena con cuatro fuentes y se evalúa en la quinta, que el modelo nunca vio.", ""]
        for esc, r in j[clave].items():
            n = next(iter(r.values()))
            L += [f"### {esc if clave == 'transferencia' else 'Prueba en ' + esc} (n = {n['n']}: {n['falsas']} falsas, {n['reales']} reales)", "",
                  ENC.replace("| ROC-AUC |", "| ROC-AUC | θ |"), SEP + "---:|"]
            for nombre, v in r.items():
                L.append(fila(nombre, v, negrita=nombre == P) + f" {f(v['umbral'], 2)} |")
            L += ["", "| Frente a SaBERT | Dif. F1 macro [IC 95 %] | Dif. FP | Dif. FN | Lectura |", "|:---|:---:|---:|---:|:---|"]
            for nombre, v in r.items():
                if 'vs_SaBERT' in v:
                    L.append(f"| {nombre} | {dif(v['vs_SaBERT'])} |")
            L.append("")

    L += [f"![Generalización]({figs['generalizacion']})", "",
          "F1 macro en la parte que el modelo nunca vio: a la izquierda, entrenando en una región y evaluando en la otra; a la derecha, "
          "excluyendo una fuente del entrenamiento y evaluando en ella.", ""]
    g, s = cv[P]['Global'], cv['SaBERT']['Global']
    L += ["## 7. Lectura", "",
          f"- **Validación cruzada:** el modelo principal obtiene F1 macro {f(g['f1_macro'])} (FP {g['FP']}, FN {g['FN']}) frente a {f(s['f1_macro'])} de SaBERT "
          f"(FP {s['FP']}, FN {s['FN']}). La significancia de cada diferencia está en la sección 4.",
          "- **Generalización:** las secciones 5 y 6 miden el desempeño en regiones y fuentes no vistas. Es la evidencia más exigente, porque elimina "
          "los atajos propios de cada fuente (formato, máscaras de anonimización, temas, fechas).",
          "- **Aporte de cada pilar:** las filas S, R, E, O y L de la sección 2 muestran cuánto discrimina cada grupo por separado; "
          "la comparación «S+R+E+O» y «Completo» frente a «tres pilares» de la sección 4 indica cuánto añaden los marcadores ortotipográficos y lingüísticos.",
          "- **SaBERT:** sus probabilidades vienen de un modelo ajustado por terceros; si su corpus de entrenamiento comparte fuentes con este, sus cifras pueden estar infladas (ver auditoría de SaBERT).", ""]

    salida = cfg.MODELO_VARIABLES_REPORTES_DIR / "Reporte_Modelo_sobre_Variables_Sin_Fuga.md"
    salida.write_text("\n".join(L) + "\n", encoding='utf-8')
    print(f"[OK] {salida}")


if __name__ == '__main__':
    main()
