# Análisis complementarios

Esta carpeta reúne los análisis realizados durante la revisión final del trabajo para comprobar la solidez de algunas decisiones del protocolo y para presentar los resultados con la nomenclatura definitiva de la tesis. Ninguno reentrena los modelos de la literatura ni modifica los scripts de los módulos principales: los reutilizan (importándolos o leyendo sus resultados) y escriben sus salidas solo en esta carpeta. La excepción es `corregir_nfc.py`, que corrige filas de dos tablas de entrada (ver abajo).

| Análisis | Script | Uso en la tesis |
|:---|:---|:---|
| Sensibilidad del clasificador interpretable a casi duplicados, imputación y reasignación de 23 textos | `sensibilidad_clasificador.py` | Subsección 3.2.3 (pares casi duplicados) y Subsección 4.4.2, Tabla 4.15 |
| Solapamiento de MEX-A3T con FakeDeS y con Unified | `solapamiento_mexa3t.py` | Subsección 3.2.3 |
| IC por *bootstrap* de la prueba de Adebayo | `adebayo_ic.py` | Subsección 4.3.3 |
| Casos de estudio con los textos originales y probabilidad fuera de pliegue | `ejecutar_casos_originales.py` | Subsección 4.4.6, Tabla 4.20, Figuras 4.25 y 4.26 |
| Figuras del OE3 con la etiqueta «Expl. por oraciones» | `figuras_oe3_renombradas.py` | Figuras 4.14 a 4.19 |
| Tres cálculos complementarios de la revisión final | `calculos_revision_tutor.py` | Complemento de las Secciones 3.3, 3.4 y de la Tabla 4.18; no figuran en el texto |
| Normalización Unicode (NFD → NFC) de 1.252 textos, casi todos de Omdena | `corregir_nfc.py` | Corrección de datos; las cifras de este repositorio ya la incorporan |

## Ejecución

Desde la raíz del repositorio. Salvo `ejecutar_casos_originales.py`, `corregir_nfc.py` sin su caché `datos_nfc/` y las tareas 1 y 2 de `calculos_revision_tutor.py`, ningún script carga modelos de lenguaje ni requiere GPU.

```bash
python Analisis_complementarios/corregir_nfc.py                   # ya aplicado; con la caché de datos_nfc/ no carga modelos
python Analisis_complementarios/sensibilidad_clasificador.py      # CPU, ~1 min; debe ir antes que ejecutar_casos_originales.py
python Analisis_complementarios/solapamiento_mexa3t.py            # CPU, segundos
python Analisis_complementarios/adebayo_ic.py                     # CPU, segundos (lee el JSON de XAI_Literatura_Comparacion)
python Analisis_complementarios/figuras_oe3_renombradas.py        # CPU, segundos (lee el JSON y la caché de XAI_Literatura_Comparacion)
python Analisis_complementarios/ejecutar_casos_originales.py      # carga los modelos del auditor de estilo; GPU opcional
python Analisis_complementarios/calculos_revision_tutor.py [1 2 3] # 3: segundos, sin modelos; 1 y 2: descargan JJNeila y el tokenizador de SaBERT
```

Todos los resultados están incluidos, de modo que pueden consultarse sin ejecutar nada.

## Scripts y decisiones metodológicas

**`corregir_nfc.py`.** 1.252 textos del corpus ampliado (1.249 de Omdena y 3 de Unified) traían las tildes en Unicode descompuesto (NFD: «explico» + acento suelto), de modo que los léxicos con tilde, spaCy y los tokenizadores de JJNeila, `hiiamsid` y SaBERT no reconocían esas palabras. La corrección normaliza a NFC (no quita tildes) en el momento de medir, dentro de `variables_texto.extraer()`, `auditor_estilo.auditar()` y `SaBERT.p_falsa()` de `auditoria_censura_vs_domain_shift.py`. El script recalcula las 43 variables y la P(falsa) de SaBERT solo en esas filas (caché en `datos_nfc/`) y las sustituye en `tabla_variables_4418.csv` y `predicciones_sabert_ampliado_4418.csv`; el resumen queda en `corregir_nfc.json` (59 predicciones de SaBERT cambian con umbral 0,5). El corpus, la columna `Text` e `id_texto` no se modifican, por lo que los pliegues de validación cruzada son los mismos y la diferencia en las métricas se debe solo a la corrección. Las copias previas (`*_pre_nfc.csv`) que guarda el script no se incluyen en el repositorio. Después de ejecutarlo hay que repetir, en este orden: `auditoria_censura_vs_domain_shift.py --recalcular`, `generar_reporte_sabert.py`, `evaluar_modelo_sobre_variables_sin_fuga.py`, `seleccionar_modelo_sobre_variables.py`, `generar_reporte_modelo_sobre_variables_sin_fuga.py`, `estilos_discursivos.py`, `generar_reporte_estilos.py`, `sensibilidad_clasificador.py` y `ejecutar_casos_originales.py`.

**`sensibilidad_clasificador.py`.** Repite la validación cruzada sin fuga del clasificador interpretable (43 variables) y de SaBERT con umbral anidado, reutilizando sin cambios las funciones de `Modelos_Individuales/Clasificar_fake/Modelo_sobre_Variables/Codigos/evaluar_modelo_sobre_variables_sin_fuga.py`. Cambia solo una de tres decisiones por escenario: (b) el grupo de `StratifiedGroupKFold`, externo e interno, pasa a ser el clúster de casi duplicados (componentes conexas de pares con coseno TF-IDF ≥ 0,90); (c) las dos variables de similitud entre oraciones consecutivas se dejan vacías en los textos de una sola oración y se imputan dentro del pliegue; (d) los 23 textos compartidos entre Unified y FakeDeS se asignan a FakeDeS / América Latina; (e) combina las tres. El escenario (a) reproduce el protocolo original y el script se detiene si no obtiene las cifras publicadas; la reproducción da predicciones idénticas a las guardadas por el script original. Guarda también las predicciones fuera de pliegue para las 4.418 filas.

**`solapamiento_mexa3t.py`.** Compara cada texto de MEX-A3T con los de FakeDeS y Unified mediante coseno TF-IDF ajustado sobre la unión de cada par de fuentes. Antes de comparar, las máscaras `*NUMBER*`, `*URL*` y `*PHONE*` se sustituyen por un mismo símbolo y se eliminan las cifras en ambos lados, para que un texto enmascarado y su original coincidan.

**`adebayo_ic.py`.** IC 95 % percentil (2.000 réplicas, semilla 42) de la media de Spearman por método y modelo, remuestreando las 15 noticias con prueba de Adebayo guardadas en `metricas_comparacion_xai.json`.

**`ejecutar_casos_originales.py`.** Toma el texto exacto de las noticias 2357, 3071, 3653 y 2 del corpus ampliado, con la etiqueta de su corpus, ejecuta el auditor de `Explicabilidad_Propia/codigos/` sin modificarlo y añade la P(falsa) fuera de pliegue del escenario (a). Regenera las figuras con las funciones de `ejecutar_casos_estilo.py`, sustituyendo en memoria el título «MexGen por oraciones» por «Expl. por oraciones».

**`figuras_oe3_renombradas.py`.** Ejecuta las funciones de figuras de `generar_reporte_comparacion_xai.py` sobre los mismos datos, renombrando en memoria «MexGen» → «Expl. por oraciones» y «MexGen-cláusulas» → «Expl. por segmentos». No recalcula explicaciones ni modifica el script original.

**`calculos_revision_tutor.py`.** Lee sus entradas desde las rutas del repositorio (`config_proyecto.py`) y escribe en `revision_tutor/`: (1) JJNeila con 128 subpalabras y umbral calibrado τ = 0,85 sobre los 70 textos sintéticos de control (Subsección 3.4.1), comparando con la caché de la evaluación original; (2) longitud del corpus ampliado con el tokenizador de SaBERT y en caracteres; (3) *bootstrap* pareado por noticia (10.000 réplicas, semilla 42) de la diferencia de ganancia sobre el azar entre la explicación por oraciones y LIME-oración / SHAP-oración. Admite elegir tareas (`python ... 3`); solo actualiza en el JSON las claves de las tareas ejecutadas.

## Resultados principales

**Casi duplicados y solapamiento.** Entre los 4.100 textos únicos hay 448 pares con coseno TF-IDF ≥ 0,90; 442 están dentro de Unified y 440 tienen etiqueta opuesta (versiones real y falsa de una misma noticia). Forman 444 clústeres de más de un texto (442 de dos y 2 de tres; 890 textos). Entre MEX-A3T (566 filas) y FakeDeS (248) la similitud máxima es 0,373, y entre MEX-A3T y Unified (2.060), 0,448: no hay ningún par ≥ 0,8 (`solapamiento_mexa3t_pares.csv` queda vacío).

**Sensibilidad del clasificador interpretable** (F1 macro; Tabla 4.15):

| Escenario | Global | ROC-AUC | España | América Latina | Esp. → AL | Sin ver Unified |
|:---|---:|---:|---:|---:|---:|---:|
| (a) Protocolo original | 0,625 | 0,676 | 0,584 | 0,663 | 0,579 | 0,505 |
| (b) Casi duplicados agrupados | 0,626 | 0,680 | 0,590 | 0,657 | 0,576 | 0,506 |
| (c) Imputación dentro del pliegue (132 textos) | 0,625 | 0,676 | 0,584 | 0,663 | 0,581 | 0,505 |
| (d) 23 textos a FakeDeS (14 falsos, 9 reales) | 0,625 | 0,676 | 0,582 | 0,664 | 0,577 | 0,511 |
| (e) Las tres correcciones | 0,625 | 0,679 | 0,589 | 0,658 | 0,575 | 0,498 |

El F1 global cambia menos de 0,001 y ningún otro escenario más de 0,01. SaBERT, con las tres correcciones, obtiene 0,782 (0,905 en España y 0,647 en América Latina). Por ello la tesis conserva las cifras del protocolo original.

**Prueba de Adebayo.** Los IC son amplios salvo en Attention Rollout: por ejemplo, [0,06; 0,65] para la explicación por oraciones en sensacionalismo, [−0,52; −0,03] en veracidad y [0,986; 0,991] para Attention Rollout en sensacionalismo.

**Casos de estudio** (Tabla 4.20; P(falsa) del modelo ajustado sobre todo el corpus y, entre corchetes, fuera de pliegue): caso 1, IML 72,9, 0,77 [0,79]; caso 2, 39,6, 0,48 [0,49]; caso 3, 35,1, 0,15 [0,16]; caso 4, 57,2, 0,28 [0,26].

**Cálculos complementarios** (`revision_tutor/`):

| Cálculo | Resultado |
|:---|:---|
| JJNeila en 70 textos sintéticos (τ = 0,85) | 70/70 aciertos, F1 macro = 1,000; P mínima de los sensacionalistas 0,907 y máxima de los no sensacionalistas 0,469; diferencia con la caché original < 10⁻⁷ |
| Noticias del corpus ampliado (4.418) con más de 256 subpalabras (SaBERT) | 1.504 (34,0 %) contando [CLS] y [SEP], es decir, truncadas con `max_length = 256`; 1.479 (33,5 %) sin ellos (texto en NFC) |
| Noticias con más de 1.000 caracteres | 1.863 (42,2 %); mediana 844 caracteres y 179 subpalabras |

*Bootstrap* pareado de la diferencia de ganancia (explicación por oraciones − método; media e IC 95 %):

| Modelo | − LIME-oración | − SHAP-oración |
|:---|:---|:---|
| Sensacionalismo | −0,005 [−0,015; 0,003] | −0,004 [−0,016; 0,009] |
| Veracidad (SaBERT) | −0,005 [−0,014; 0,003] | −0,006 [−0,016; 0,003] |
| Redundancia | +0,007 [0,002; 0,011] | +0,007 [0,003; 0,011] |
| Clasificador interpretable | −0,003 [−0,007; −0,0004] | 0,000 [−0,003; 0,004] |

Las medias reproducen las de la Tabla 4.18 y siete de los ocho intervalos coinciden con la conclusión del Wilcoxon-Holm. La excepción es el clasificador interpretable frente a LIME-oración: el intervalo queda justo por debajo de cero, mientras que la prueba de Wilcoxon con corrección de Holm da p = 0,130. La diferencia (−0,003) es mínima y no cambia la conclusión de que los tres métodos por oración tienen una fidelidad equivalente.

## Figuras

| Carpeta | Contenido |
|:---|:---|
| [`figuras_oe3/`](figuras_oe3/) | Versión de la tesis de las Figuras 4.14 a 4.19 (ganancia, curvas de borrado, Adebayo, borrado aleatorio, ejemplo y acuerdo), con «Expl. por oraciones». `costo_frente_a_ganancia.png` no se incluyó en la tesis. Las versiones con la etiqueta interna «MexGen» están en `Explicabilidad_Propia/XAI_Literatura_Comparacion/reportes/imagenes/` (ver su [README](../Explicabilidad_Propia/XAI_Literatura_Comparacion/README.md)). |
| [`casos_estilo/`](casos_estilo/) | [`casos_mexgen_oraciones.png`](casos_estilo/casos_mexgen_oraciones.png) (Figura 4.25), [`casos_radar_estilo.png`](casos_estilo/casos_radar_estilo.png) (Figura 4.26) y `casos_dimensiones_iml.png` (no incluida), más un JSON por caso con la salida completa del auditor. |

## Limitaciones

- Los casi duplicados se detectan con similitud léxica (TF-IDF) y un umbral fijo de 0,90; las paráfrasis con otro vocabulario no se detectan.
- La sensibilidad se evaluó para el modelo completo y para SaBERT, no para cada configuración de variables.
- Los IC de Adebayo se basan en 15 noticias y en una prueba parcial.
- Los cuatro casos son ilustrativos, no una muestra representativa.
- Los 70 textos sintéticos son contrastivos y sirven como control de cordura, no como medida de validez; el F1 de 1,000 no se extiende a noticias reales.
- Un tercio de las noticias supera las 256 subpalabras y SaBERT solo ve su comienzo; la muestra de la comparación de explicabilidad evita este efecto al limitarse a 200 palabras.

## Archivos generados

| Archivo | Script |
|:---|:---|
| `corregir_nfc.json`, `datos_nfc/variables_nfc.csv`, `datos_nfc/sabert_nfc.csv` | `corregir_nfc.py` |
| `sensibilidad_clasificador.json`, `predicciones_oof_original.csv`, `clusteres_casi_duplicados.csv`, `pares_casi_duplicados_090.csv` | `sensibilidad_clasificador.py` |
| `solapamiento_mexa3t.json`, `solapamiento_mexa3t_pares.csv` | `solapamiento_mexa3t.py` |
| `adebayo_ic.json` | `adebayo_ic.py` |
| `casos_originales.json`, `casos_estilo/*.json`, `casos_estilo/*.png` | `ejecutar_casos_originales.py` |
| `figuras_oe3/*.png` | `figuras_oe3_renombradas.py` |
| `revision_tutor/resultados_revision_tutor.json`, `tarea1_predicciones_sinteticos.csv`, `tarea2_longitudes_corpus.csv`, `tarea3_bootstrap_pareado.csv` | `calculos_revision_tutor.py` |
