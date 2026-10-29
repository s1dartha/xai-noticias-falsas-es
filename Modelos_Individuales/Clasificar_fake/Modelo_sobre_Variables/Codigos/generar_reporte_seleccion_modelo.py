"""
Genera Reportes/Reporte_Seleccion_Modelo_sobre_Variables.md a partir de Reportes/metricas_seleccion_modelo_sobre_variables.json.
Las cifras se leen del JSON. La sección de antecedentes resume la literatura de la carpeta Literatura/ (con página).
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config_proyecto.py").exists())))
import config_proyecto as cfg
import graficos_comunes as gc

NOMBRE = {'lr': 'Regresión logística', 'rf': 'Bosque aleatorio', 'gbdt': 'Gradient boosting', 'sabert': 'SaBERT (referencia)'}


def f(x, d=3):
    return '—' if x is None else f"{x:.{d}f}".replace('.', ',')


def pct(x):
    return f"{x * 100:.1f}".replace('.', ',') + " %"


def ic(v):
    return f"[{f(v[0])}; {f(v[1])}]"


ANTECEDENTES = [
    gc.diagrama_md('tabla_variables.png', cfg.MODELO_VARIABLES_REPORTES_DIR, 'De dónde salen las 43 variables del Modelo sobre Variables'),
    "## 1. Antecedentes: qué clasificadores usa la literatura con variables de estilo", "",
    "El Modelo sobre Variables no clasifica texto crudo: recibe una tabla de 43 variables numéricas por noticia "
    "(sensacionalismo, redundancia, emociones, marcadores ortotipográficos y rasgos lingüísticos). Es un problema de "
    "**datos tabulares con variables con significado**, y la literatura sobre detección de noticias falsas con variables "
    "de estilo converge en tres familias de clasificadores:", "",
    "| Trabajo | Variables | Clasificadores comparados | Resultado relevante |", "|:---|:---|:---|:---|",
    "| Muñoz e Iglesias (2024), *Exploiting Content Characteristics for Explainable Detection of Fake News*, Big Data and Cognitive Computing 8, 129 "
    "(`Literatura/Antecedentes/`) | Más de 100 variables lingüísticas, afectivas (emociones NRC, VADER), de legibilidad y LIWC (pp. 5–6); es el antecedente más parecido a esta tesis | "
    "Regresión logística, LinearSVC, árbol de decisión, bosque aleatorio, XGBoost, CatBoost; validación cruzada de 5 pliegues (p. 10) | "
    "F1 medio en 9 corpus: CatBoost 0,812, XGBoost 0,792, bosque aleatorio 0,788, regresión logística 0,770, árbol 0,723 (Tabla 2, p. 11). "
    "Los clasificadores tradicionales logran «competitive results with significantly lower computational requirements» (p. 1) |",
    "| Ortega Riveros y Quintero Perozo (2020), *Detección automática de noticias falsas en español con técnicas de machine learning*, Universidad de los Andes "
    "(`Literatura/Antecedentes/`) | Variables de estilo en español: porcentaje de mayúsculas, signos de interrogación, etc. (p. 2) | "
    "Bosque aleatorio, gradient boosting, BiLSTM (pp. 13–15) | Bosque aleatorio 76,3 % de exactitud, gradient boosting 73,9 %, BiLSTM 59,0 % (Tabla 4, p. 16) |",
    "| Horne y Adalı (2017), *This Just In: Fake News Packs a Lot in Title, Uses Simpler, Repetitive Content…*, ICWSM "
    "(`Literatura/patrones o caracteristicas/Sensacionalismo/`) | Mayúsculas, puntuación, exclamaciones, legibilidad, redundancia léxica, sentimiento (pp. 3–4) | "
    "SVM lineal con pocas variables, para «avoid the over-fitting» (p. 5) | 71 % de exactitud con el cuerpo de la noticia y 78 % con el titular (Tabla 6, p. 6) |",
    "| Flores Quinayás y Montaño Morcillo (2022), detección de noticias falsas en Twitter, elecciones de Colombia 2022 (`Literatura/Antecedentes/`) | TF-IDF (no de estilo) | "
    "Bosque aleatorio, SVC, Naive Bayes, regresión logística, XGBoost, BERT (p. 36) | Bosque aleatorio y regresión logística empatan (0,898 de exactitud, Tabla 25, pp. 41–42) |",
    "| Grinsztajn, Oyallon y Varoquaux (2022), *Why do tree-based models still outperform deep learning on typical tabular data?*, NeurIPS | 45 conjuntos tabulares de dominios variados | "
    "Redes neuronales frente a XGBoost y bosque aleatorio | Los modelos de árboles siguen siendo el estado del arte en datos tabulares de tamaño medio (~10.000 filas), como el de esta tesis |",
    "| Rudin (2019), *Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead*, Nature Machine Intelligence "
    "(`Literatura/Explicabilidad_No/`) | Argumento metodológico | — | «It is a myth that there is necessarily a trade-off between accuracy and interpretability» (p. 2); "
    "con «structured data with meaningful features, there is often no significant difference in performance between more complex classifiers (… boosted decision trees, random forests) "
    "and much simpler classifiers (logistic regression, decision lists)» (p. 2) |", "",
    "**Por qué estas tres familias:**", "",
    "1. **Regresión logística.** Es el modelo interpretable de referencia: cada coeficiente indica cuánto y en qué dirección empuja una variable hacia «falsa». "
    "Rudin (2019) argumenta que en datos estructurados con variables con significado suele rendir igual que modelos complejos, y en Muñoz e Iglesias (2024) "
    "queda a unos 2 puntos de F1 de los ensambles de árboles.",
    "2. **Bosque aleatorio.** Es el mejor clasificador en los dos antecedentes en español (Ortega Riveros y Quintero Perozo, 2020; Flores Quinayás y Montaño Morcillo, 2022) "
    "y es robusto a variables poco informativas y a relaciones no lineales.",
    "3. **Gradient boosting.** Es la familia con mejor F1 sobre variables de contenido en Muñoz e Iglesias (2024) y la que Grinsztajn et al. (2022) señalan como estado del arte en datos tabulares.", "",
    "Se descartan el SVM lineal (cumple el mismo papel que la regresión logística, pero sin probabilidades calibradas, que el umbral anidado necesita), "
    "el árbol de decisión simple (el peor en Muñoz e Iglesias, 2024) y las redes neuronales (Grinsztajn et al., 2022, muestran que no superan a los árboles en tablas de este tamaño).", "",
]


def main():
    j = json.load(open(cfg.SELECCION_MODELO_JSON, encoding='utf-8'))
    m, mod, sab, ext, comp = j['metadata'], j['modelos'], j['sabert_referencia'], j['escenarios_externos'], j['comparaciones_pareadas']
    tipos = list(mod)
    el = j['modelo_elegido']
    fig = j['figuras']

    L = ["# Selección del clasificador del Modelo sobre Variables", "",
         "**Generado por:** `generar_reporte_seleccion_modelo.py` a partir de `Reportes/metricas_seleccion_modelo_sobre_variables.json`  ",
         "**Evaluación:** `seleccionar_modelo_sobre_variables.py`  ",
         f"**Datos:** {m['n_textos_unicos']} textos únicos y {m['n_variables']} variables (S+R+E+O+L) · **Etiqueta:** 1 = falsa", ""]
    L += ANTECEDENTES
    L += ["## 2. Protocolo de comparación", "",
          f"- **Validación:** {m['validacion']}; los tres modelos ven exactamente los mismos pliegues, variables y procedimiento de umbral.",
          "- **Sin fuga:** imputación, estandarización, modelo y umbral se ajustan solo con el entrenamiento de cada pliegue.",
          "- **Hiperparámetros fijos** (valores habituales, no ajustados con estos datos), para que la comparación mida la familia de modelos y no el esfuerzo de ajuste "
          "(Rudin, 2019, Apéndice B, advierte que el ajuste de hiperparámetros puede pesar más que la elección del algoritmo):", ""]
    for t in tipos:
        L.append(f"  - **{NOMBRE[t]}:** {m['modelos'][t]['configuracion']}.")
    L += ["- **Escenarios:** validación cruzada (global y por región), transferencia entre regiones y exclusión de una fuente por vez.",
          f"- **Regla de decisión (fijada antes de ver resultados):** {m['regla']}.",
          "- SaBERT se incluye solo como referencia, con el mismo procedimiento de umbral.", "",
          "## 3. Resultados en validación cruzada", "",
          "| Modelo | F1 macro [IC 95 %] | Exactitud | Precisión (falsa) | Recall (falsa) | Especificidad | FP | FN | Tasa FP | Tasa FN | ROC-AUC | PR-AUC | Tiempo (s) |",
          "|:---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for t in tipos + ['sabert']:
        r = (sab if t == 'sabert' else mod[t])['validacion_cruzada']['Global']
        seg = f(mod[t]['segundos_validacion_cruzada'], 0) if t != 'sabert' else '—'
        n = f"**{NOMBRE[t]}**" if t == el else NOMBRE[t]
        L.append(f"| {n} | {f(r['f1_macro'])} {ic(r['f1_macro_ic95'])} | {pct(r['exactitud'])} | {pct(r['precision_falsa'])} | {pct(r['recall_falsa'])} | "
                 f"{pct(r['especificidad'])} | {r['FP']} | {r['FN']} | {pct(r['tasa_FP'])} | {pct(r['tasa_FN'])} | {f(r['roc_auc'])} | {f(r['pr_auc'])} | {seg} |")
    L += ["", f"![F1 macro por escenario]({fig['f1_por_escenario']})", "",
          f"![Falsos positivos y negativos]({fig['fp_fn']})", "",
          f"![Curvas ROC y precisión-recall]({fig['roc_pr']})", "",
          "### Por región (F1 macro [IC 95 %] · FP · FN)", "",
          "| Modelo | España | América Latina | América Latina: Omdena | América Latina: sin Omdena |", "|:---|:---:|:---:|:---:|:---:|"]
    for t in tipos + ['sabert']:
        r = (sab if t == 'sabert' else mod[t])['validacion_cruzada']
        L.append(f"| {NOMBRE[t]} | " + " | ".join(f"{f(r[k]['f1_macro'])} {ic(r[k]['f1_macro_ic95'])} · {r[k]['FP']} · {r[k]['FN']}"
                                                   for k in ['España', 'América Latina', 'América Latina: Omdena', 'América Latina: sin Omdena']) + " |")
    L += ["", "### Diferencias pareadas frente a la regresión logística (bootstrap)", "",
          "| Comparación | Dif. F1 macro [IC 95 %] | Dif. FP | Dif. FN | Lectura |", "|:---|:---:|---:|---:|:---|"]
    for k, d in comp.items():
        lect = "mejor que la regresión logística" if d['ic95'][0] > 0 else ("peor que la regresión logística" if d['ic95'][1] < 0 else "sin diferencia clara")
        L.append(f"| {k} | {f(d['dif_f1_macro'])} {ic(d['ic95'])} | {d['dif_FP']:+d} | {d['dif_FN']:+d} | {lect} |")

    L += ["", "## 4. Generalización: regiones y fuentes no vistas", "",
          "| Escenario | n | " + " | ".join(NOMBRE[t] for t in tipos + ['sabert']) + " |", "|:---|---:|" + "---:|" * (len(tipos) + 1)]
    for esc, r in ext.items():
        mejor = max(tipos, key=lambda t: r[t]['f1_macro'])
        celdas = []
        for t in tipos + ['sabert']:
            c = f"{f(r[t]['f1_macro'])} (FP {r[t]['FP']}, FN {r[t]['FN']})"
            celdas.append(f"**{c}**" if t == mejor else c)
        L.append(f"| {esc} | {r['lr']['n']} | " + " | ".join(celdas) + " |")
    L.append("| **Media de exclusión por fuente** | | " + " | ".join(f(j['resumen_exclusion_fuente'][t]) for t in tipos + ['sabert']) + " |")
    L += ["", "Cada celda es el F1 macro en la parte no vista; en negrita, el mejor de los tres clasificadores sobre variables.", "",
          f"![Exclusión de una fuente]({fig['exclusion_fuente']})", "",
          "## 5. Decisión", "", f"- **Modelo elegido: {NOMBRE[el]}.** {j['motivo']}."]
    cvl = mod['lr']['validacion_cruzada']['Global']
    L += [f"- La regresión logística obtiene F1 macro {f(cvl['f1_macro'])} con {cvl['FP']} falsos positivos y {cvl['FN']} falsos negativos, "
          f"y es además la más rápida ({f(mod['lr']['segundos_validacion_cruzada'], 0)} s frente a "
          + ", ".join(f"{f(mod[t]['segundos_validacion_cruzada'], 0)} s del {NOMBRE[t].lower()}" for t in tipos if t != 'lr') + ").",
          "- **No gana en todos los escenarios:** " + "; ".join(
              f"{NOMBRE[t].lower()} es el mejor en {sum(1 for r in ext.values() if max(tipos, key=lambda u: r[u]['f1_macro']) == t)} de {len(ext)} escenarios externos"
              for t in tipos) + f"; en la validación cruzada de América Latina el mejor es el "
          f"{NOMBRE[max(tipos, key=lambda t: mod[t]['validacion_cruzada']['América Latina']['f1_macro'])].lower()}. Las diferencias entre familias son pequeñas frente a la "
          "brecha con SaBERT, lo que coincide con Rudin (2019): con variables con significado, ningún algoritmo domina con claridad y un modelo interpretable no pierde "
          "de forma sistemática frente a los ensambles de árboles.",
          "- **Consecuencia para la tesis:** el Modelo sobre Variables se implementa con regresión logística; sus coeficientes son la explicación global del modelo y "
          "la base del IML (Fase 5)." if el == 'lr' else f"- **Consecuencia para la tesis:** el Modelo sobre Variables se implementa con {NOMBRE[el].lower()}.", "",
          "## 6. Coeficientes de la regresión logística", "",
          "Ajuste descriptivo con los 4.100 textos únicos (las métricas anteriores son todas fuera de pliegue). Las variables están estandarizadas: el coeficiente es "
          "el cambio en log-odds de «falsa» por cada desviación estándar, y el odds ratio, el factor por el que se multiplican las probabilidades relativas.", "",
          f"![Coeficientes de la regresión logística]({fig['coeficientes_lr']})", "",
          "| # | Variable | Grupo | Coeficiente | Odds ratio por 1 DE | Dirección |", "|---:|:---|:---:|---:|---:|:---|"]
    for i, r in enumerate(j['coeficientes_regresion_logistica'][:15], 1):
        L.append(f"| {i} | `{r['variable']}` | {r['grupo']} | {f(r['coeficiente_estandarizado'])} | {f(r['odds_ratio_por_1_DE'], 2)} | "
                 f"{'hacia falsa' if r['coeficiente_estandarizado'] > 0 else 'hacia real'} |")
    L += ["", "Los coeficientes reflejan asociación condicionada a las demás variables, no causalidad; variables correlacionadas (por ejemplo, las de mayúsculas) se reparten el peso.", "",
          "## 7. Limitaciones", "",
          "1. Los hiperparámetros no se ajustaron; un ajuste anidado podría mejorar algo a los modelos de árboles, aunque Rudin (2019) señala que esa ganancia suele ser comparable a la diferencia entre algoritmos.",
          "2. Ningún clasificador sobre variables alcanza a SaBERT en España; la comparación con SaBERT está en `Reporte_Modelo_sobre_Variables_Sin_Fuga.md`.",
          "3. Las cifras de la literatura citadas provienen de otros corpus y métricas (a menudo exactitud) y no son comparables directamente con las de esta tesis.", "",
          "## Referencias", "",
          "- Flores Quinayás, J. E. y Montaño Morcillo, J. G. (2022). *Modelo para la detección de noticias falsas en formato texto en la red social Twitter, aplicado al contexto político colombiano de las elecciones presidenciales de 2022*. Proyecto de grado, Maestría en Ciencia de Datos.",
          "- Grinsztajn, L., Oyallon, E. y Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *Advances in Neural Information Processing Systems 35* (NeurIPS 2022, Datasets and Benchmarks).",
          "- Horne, B. D. y Adalı, S. (2017). This Just In: Fake News Packs a Lot in Title, Uses Simpler, Repetitive Content in Text Body, More Similar to Satire than Real News. *ICWSM 2017 Workshops*, AAAI.",
          "- Muñoz, S. e Iglesias, C. Á. (2024). Exploiting Content Characteristics for Explainable Detection of Fake News. *Big Data and Cognitive Computing*, 8, 129.",
          "- Ortega Riveros, J. A. y Quintero Perozo, D. Y. (2020). *Detección automática de noticias falsas en español con técnicas de machine learning*. Trabajo de grado, Universidad de los Andes.",
          "- Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 1, 206–215.", ""]
    salida = cfg.MODELO_VARIABLES_REPORTES_DIR / "Reporte_Seleccion_Modelo_sobre_Variables.md"
    salida.write_text("\n".join(L) + "\n", encoding='utf-8')
    print(f"[OK] {salida}")


if __name__ == '__main__':
    main()
