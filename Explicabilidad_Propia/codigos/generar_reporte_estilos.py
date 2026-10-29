"""
Genera reportes/Reporte_Construccion_Estilos_Discursivos.md a partir de reportes/metricas_iml_estilos_discursivos.json.
Todas las cifras se leen del JSON.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config_proyecto.py").exists())))
import config_proyecto as cfg
import graficos_comunes as gc

NOMBRES_GRUPO = {'sens': 'S', 'emo': 'E', 'red': 'R', 'orto': 'O', 'ling': 'L'}


def f(x, d=3):
    return '—' if x is None else f"{x:.{d}f}".replace('.', ',')


def pct(x, d=1):
    return f"{x * 100:.{d}f}".replace('.', ',') + " %"


def main():
    j = json.load(open(cfg.METRICAS_ESTILOS_DISCURSIVOS_JSON, encoding='utf-8'))
    dims, est, fig = j['dimensiones'], j['estilos'], j['figuras']
    k = j['k_elegido']
    usadas, excl = j['variables_usadas'], j['variables_excluidas']
    de = j['direccion_empirica']
    coinciden = sum(v['coincide'] for v in de.values())
    L = ["# Construcción del IML y de los estilos discursivos", "",
         "**Generado por:** `codigos/generar_reporte_estilos.py` a partir de `reportes/metricas_iml_estilos_discursivos.json`  ",
         "**Construcción:** `codigos/estilos_discursivos.py` · **Variables:** `Modelo_sobre_Variables/Dataset/tabla_variables_4418.csv`  ",
         f"**Corpus de referencia:** {j['n_textos_unicos']} textos únicos ({j['n_noticias']} noticias)", "",
         gc.diagrama_md('iml_y_estilos.png', cfg.EXPLICABILIDAD_REPORTES_DIR, 'Cómo se construyen el IML y los estilos discursivos'),
         "## 1. Qué son y qué cambió", "",
         "Los **estilos discursivos** son perfiles de la **forma** en que está escrita una noticia: cuánto sensacionaliza, cuánta emoción carga, cuánto repite, "
         "cuánto enfatiza con mayúsculas y signos, y cuánto afirma sin atribuir a fuentes. **No son categorías de veracidad**: una noticia verdadera puede tener un "
         "estilo sensacionalista y una falsa puede estar escrita con sobriedad.", "",
         "| Versión anterior | Versión vigente (estilos discursivos) |", "|:---|:---|",
         "| Solo variables de sensacionalismo y redundancia | Las cinco familias: sensacionalismo, emociones, redundancia, marcadores ortotipográficos y rasgos lingüísticos |",
         "| Dimensiones con fórmulas y constantes ad hoc (por ejemplo, `(p_max − p_mean) × 150`) | Cada variable se expresa como percentil del corpus; cada dimensión es un promedio |",
         "| Pesos fijos del IML (35 %, 20 %, 15 %, 20 %, 10 %) | El IML es el promedio simple de las cinco dimensiones |",
         "| Cinco perfiles definidos a mano con rampas y umbrales (`D1 > 40…75`) | Los estilos salen de los datos (k-medias); su número se elige con silueta y estabilidad |",
         "| Cada noticia recibe un perfil fijo | Cada noticia recibe un grado de pertenencia a cada estilo (estilo dominante y secundario) |",
         "| Umbrales de redundancia (GMM, P95) dentro del índice | La redundancia entra como similitud continua, sin umbrales |", "",
         "## 2. Las cinco dimensiones y de dónde sale cada una", "",
         f"Entran **{len(usadas)} de las 43 variables** de la tabla. A cada una se le fijó una dirección **antes de mirar las etiquetas** "
         "(↑: más valor = estilo más marcado; ↓: menos valor = estilo más marcado).", "",
         "| Dimensión | Variables (dirección) | Fuente | Por qué |", "|:---|:---|:---|:---|"]
    fuentes = {'D1': 'JJNeila (sensacionalismo), por documento y por oración', 'D2': 'RoBERTuito (emociones), por oración',
               'D3': 'hiiamsid (SBERT), similitud entre oraciones', 'D4': 'Expresiones regulares sobre el texto',
               'D5': 'Expresiones regulares y spaCy'}
    for d, info in dims.items():
        vs = ", ".join(f"`{v}` {'↑' if s > 0 else '↓'}" for v, s in info['variables'].items())
        L.append(f"| **{d} {info['nombre']}** | {vs} | {fuentes[d]} | {info['justificacion']} |")
    L += ["", "**Variables que no entran al IML** (siguen en el Modelo sobre Variables):", "", "| Variable | Motivo |", "|:---|:---|"]
    for v, m in excl.items():
        L.append(f"| `{v}` | {m} |")
    L += ["", "## 3. Cómo se calcula", "",
          "1. **Percentil de cada variable** respecto de los textos únicos del corpus (0–100), invertido si la dirección es ↓.",
          "2. **Dimensión** = percentil, dentro del corpus, del promedio de los percentiles de sus variables. Este segundo paso hace que las cinco dimensiones "
          "tengan la misma escala: sin él, las dimensiones con muchas variables poco correlacionadas (énfasis ortotipográfico, aserción) quedan comprimidas cerca de 50 "
          "y no pesan en los estilos.",
          "3. **IML** = promedio de D1…D5 (0–100). Un IML de 70 indica una noticia con rasgos de estilo más marcados que la mayoría del corpus.",
          "4. **Estilos**: k-medias sobre (D1…D5) de los textos únicos. Una noticia de una sola oración no tiene D3 y se le asigna el valor neutro 50 para el agrupamiento.",
          "5. **Pertenencia** a cada estilo = inverso de la distancia al centro del estilo, normalizado para que sume 100 %.", "",
          "La etiqueta de veracidad no interviene en ninguno de estos pasos.", "",
          "## 4. Cuántos estilos", "",
          f"Regla: {j['regla_k']}. Con k = 2 los estilos solo separan «más» de «menos» a lo largo de un único eje (sensacionalismo y emoción) y no describen "
          "combinaciones de rasgos, por eso el rango empieza en 3.", "",
          "| k | Silueta | Estabilidad (ARI medio) | ARI percentil 5 |", "|---:|---:|---:|---:|"]
    for kk, v in j['eleccion_k'].items():
        n = f"**{kk}**" if int(kk) == k else kk
        L.append(f"| {n} | {f(v['silueta'])} | {f(v['estabilidad_ari_media'])} | {f(v['estabilidad_ari_p5'])} |")
    L += ["", f"![Elección del número de estilos](../{fig['eleccion_k']})", "",
          f"Se eligen **{k} estilos**. Las siluetas son bajas (entre {f(min(v['silueta'] for v in j['eleccion_k'].values()), 2)} y "
          f"{f(max(v['silueta'] for v in j['eleccion_k'].values()), 2)}): los estilos son **zonas de un continuo**, no grupos separados. "
          "Por eso cada noticia recibe un grado de pertenencia y no una etiqueta única.", "",
          "## 5. Los estilos", "",
          "| Estilo | D1 | D2 | D3 | D4 | D5 | IML del centro | Noticias (únicas) | % | Pertenencia media al estilo dominante |", "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for e in est:
        c = e['centro']
        L.append(f"| {e['nombre']} | " + " | ".join(f(c[d], 0) for d in dims) + f" | {f(e['iml_centro'], 0)} | {e['n']} | {pct(e['proporcion'])} | {f(e['pertenencia_media'], 0)} % |")
    L += ["", f"![Centros de los estilos](../{fig['centros']})", "",
          f"![Radar de los estilos](../{fig['radar']})", "",
          "### Cómo se nombraron", "",
          "Los nombres se pusieron **después** del agrupamiento y se asignan por el **perfil del centro**, no por el número del grupo de k-medias: "
          f"{j['regla_nombres']} Si al reconstruir los estilos los centros dejaran de cumplir estas condiciones, el script se detiene en lugar de "
          "asignar un nombre que no corresponde. El «perfil» es la descripción automática del centro (un rasgo se nombra si está al menos 10 puntos "
          "por encima o por debajo de la mediana del corpus) y es la evidencia que respalda cada nombre.", "",
          "| Estilo | Lema | Perfil del centro (automático) | Equivalente en inglés |", "|:---|:---|:---|:---|"]
    for e in est:
        L.append(f"| **{e['nombre']}** | {e['lema']} | {e['perfil']} | *{e['nombre_en']}* |")
    L += ["", "- **Institucional / Técnico:** el sensacionalismo y la emoción más bajos del corpus y la aserción sin respaldo más baja (más verbos de "
          "atribución, cifras y fechas), con reiteración de las ideas clave y énfasis ortotipográfico por encima de la mediana (los más altos de los tres estilos).",
          "- **Opinión / Crónica:** sensacionalismo y emoción algo por debajo de la mediana y aserción algo por encima, pero con la menor reiteración "
          "(el texto avanza) y casi sin mayúsculas ni signos enfáticos (escritura estándar).",
          "- **Alarmista / Viral:** el más alto en sensacionalismo, emoción y aserción sin respaldo; algo por encima de la mediana en énfasis ortotipográfico "
          "y algo por debajo en reiteración.", ""]
    pr = j['proyecciones']
    var, vec = pr['pca']['varianza_explicada'], pr['vecindad']
    cargas = pr['pca']['cargas']
    def cargas_txt(pc):
        return ", ".join(f"{d} {f(v, 2)}" for d, v in sorted(cargas[pc].items(), key=lambda x: -abs(x[1])))
    L += ["### Mapa de las noticias: PCA, t-SNE y UMAP", "",
          "Cada punto es una noticia (textos únicos) ubicada según sus cinco dimensiones y proyectada a dos ejes con tres técnicas:", "",
          f"- **PCA** (lineal) es la proyección que mejor conserva las distancias globales. Los dos primeros componentes explican {pct(var[0], 0)} y {pct(var[1], 0)} de la varianza "
          f"({pct(var[0] + var[1], 0)} en total), así que el plano muestra algo más de la mitad de la información. Cargas: PC1 = {cargas_txt('PC1')}; "
          f"PC2 = {cargas_txt('PC2')}.",
          f"- **t-SNE** (perplejidad {pr['tsne']['perplejidad']}) y **UMAP** ({pr['umap']['n_vecinos']} vecinos) son no lineales: preservan qué noticias son "
          "**vecinas**, pero no las distancias globales; el tamaño de las zonas y la separación entre ellas en el dibujo no significan nada por sí mismos.", "",
          f"![Proyecciones por estilo](../{fig['proyecciones_estilo']})", "",
          f"Los tres estilos ocupan **zonas contiguas de una misma nube**, sin huecos entre ellas: es lo que anticipaba la silueta baja. "
          f"En el espacio original de cinco dimensiones, de los {vec['k']} vecinos más cercanos de cada noticia, en promedio {pct(vec['mismo_estilo'], 0)} "
          f"pertenecen a su mismo estilo (frente a {pct(vec['mismo_estilo_azar'], 0)} si fuera al azar): las zonas son coherentes, aunque los bordes son difusos. "
          "Por eso cada noticia lleva un grado de pertenencia y no una etiqueta rígida.", "",
          f"![Proyecciones por clase](../{fig['proyecciones_clase']})", "",
          f"Coloreadas por clase, falsas y reales se **mezclan** en todo el mapa: entre los vecinos de una noticia, {pct(vec['misma_clase'], 0)} comparten su "
          f"clase, apenas por encima de {pct(vec['misma_clase_azar'], 0)} esperado al azar. Las falsas se concentran algo más en la zona del estilo "
          "Alarmista / Viral, pero el estilo de escritura no separa la veracidad; esa es la función del Modelo sobre Variables y de SaBERT.", "",
          "### Distribución por región", "", "| Estilo | " + " | ".join(sorted({r for e in est for r in e['por_region']})) + " |", "|:---|" + "---:|" * len({r for e in est for r in e['por_region']})]
    regiones = sorted({r for e in est for r in e['por_region']})
    for e in est:
        L.append(f"| {e['nombre'].split(':')[0]} | " + " | ".join(str(e['por_region'].get(r, 0)) for r in regiones) + " |")
    L += ["", "## 6. Relación con la veracidad (solo descriptiva)", "",
          f"La proporción global de noticias falsas es {pct(j['tasa_falsas_global'])}. Por estilo:", "",
          "| Estilo | Falsas | % de falsas |", "|:---|---:|---:|"]
    for e in est:
        L.append(f"| {e['nombre']} | {e['falsas']} de {e['n']} | {pct(e['tasa_falsas'])} |")
    L += ["", f"![Tamaño de los estilos y proporción de falsas](../{fig['tamanos']})", "",
          "| Indicador | AUC (falsa) | Media en falsas | Media en reales |", "|:---|---:|---:|---:|"]
    for c, v in j['asociacion_veracidad'].items():
        L.append(f"| {c} | {f(v['auc_falsa'])} | {f(v['media_falsas'], 1)} | {f(v['media_reales'], 1)} |")
    L += ["", f"![Distribución del IML por clase](../{fig['iml_clase']})", "",
          "El IML y los estilos se asocian con la veracidad de forma **débil**: el estilo más marcado concentra más falsas, pero hay noticias verdaderas "
          "en todos los estilos y la AUC del IML está lejos de un detector. Esto es coherente con su propósito: describir cómo está escrita una noticia, "
          "no decidir si es falsa (esa es la función del Modelo sobre Variables y de SaBERT).", "",
          "### Comprobación de las direcciones fijadas a priori", "",
          f"En {coinciden} de {len(de)} variables la dirección fijada a priori coincide con la diferencia observada entre falsas y reales (AUC por encima o por "
          "debajo de 0,5). Las que no coinciden se mantienen: la dirección define el constructo de estilo, no se ajusta a las etiquetas.", "",
          "| Variable | Dirección a priori | AUC (falsa) | Coincide |", "|:---|:---:|---:|:---:|"]
    for v, x in de.items():
        L.append(f"| `{v}` | {'↑' if x['direccion_a_priori'] > 0 else '↓'} | {f(x['auc_falsa'])} | {'sí' if x['coincide'] else 'no'} |")
    corr = j['correlacion_dimensiones_spearman']
    L += ["", "### Correlación entre dimensiones (Spearman)", "", "| | " + " | ".join(dims) + " |", "|:---|" + "---:|" * len(dims)]
    for a in dims:
        L.append(f"| {a} | " + " | ".join(f(corr[a][b], 2) for b in dims) + " |")
    L += ["", "## 7. Perfilar una noticia nueva", "",
          "`codigos/auditor_estilo.py` calcula las 43 variables de un texto nuevo con exactamente el mismo procedimiento que el corpus "
          "(`codigos/variables_texto.py`, verificado en 30 noticias del corpus con diferencia máxima de 10⁻⁶), lo ubica en los percentiles guardados en "
          "`codigos/referencia_iml_estilos.json` y devuelve las dimensiones, el IML, el estilo con sus pertenencias y las oraciones donde se concentra cada rasgo. "
          "Ejemplos: [`Reporte_Casos_Estilos_Discursivos_4_Noticias.md`](Reporte_Casos_Estilos_Discursivos_4_Noticias.md).", "",
          "## 8. Limitaciones", "",
          "1. Los percentiles son relativos a este corpus (España y América Latina); un IML de 70 significa «más marcado que el 70 % de estas noticias», no un valor absoluto.",
          "2. Las direcciones de las variables se apoyan en la literatura, pero algunas no coinciden con lo observado en este corpus (sección 6).",
          "3. Las siluetas bajas indican que no hay grupos naturales bien separados; los estilos son una partición útil para describir, no tipos discretos.",
          "4. Los nombres (Institucional / Técnico, Opinión / Crónica, Alarmista / Viral) son una interpretación de los centros: describen el estilo "
          "típico de la zona, no el género periodístico real de cada noticia (no se anotó si una noticia es un boletín oficial o una columna de opinión).",
          "5. Las variables de sensacionalismo y emociones heredan las limitaciones de sus modelos (F1 macro por debajo de 0,65 en sus validaciones).", ""]
    salida = cfg.EXPLICABILIDAD_REPORTES_DIR / 'Reporte_Construccion_Estilos_Discursivos.md'
    salida.write_text("\n".join(L) + "\n", encoding='utf-8')
    print(f"[OK] {salida}")


if __name__ == '__main__':
    main()
