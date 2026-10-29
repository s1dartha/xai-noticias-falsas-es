# Comparación de la explicación por oraciones con seis métodos de la literatura

Esta carpeta contiene la evaluación del objetivo específico 3 (OE3) de la tesis: qué tan fieles y confiables son las explicaciones de los métodos de atribución de la literatura y cómo se comparan con la explicación por oraciones del marco. La metodología se describe en la Sección 3.5 (y en la Subsección 3.6.3 para la explicación por oraciones); los resultados, en la Sección 4.3 y en la Subsección 4.4.5 (Tabla 4.18).

**Nomenclatura.** En el código y en los JSON, la explicación por oraciones conserva su nombre interno `MexGen` (adaptación de MExGen, Monteiro Paes et al., 2025, con *leave-one-out* por oraciones), y su variante que corta también en comas se llama `MexGen-cláusulas`. En la tesis aparecen como «Expl. por oraciones» y «Expl. por segmentos». Las claves no se renombraron para no invalidar la caché.

## Diseño

| Elemento | Decisión |
|:---|:---|
| Modelos explicados | Sensacionalismo (JJNeila, P de la clase predicha), veracidad (SaBERT, P de la clase predicha), redundancia (`hiiamsid`, similitud media entre oraciones) y clasificador interpretable sobre las 43 variables («Modelo sobre Variables», P de la clase predicha). |
| Muestra | 100 textos únicos por modelo, 25 por estrato región × clase (semilla 42); al menos 3 oraciones y como máximo 200 palabras, para que todos los métodos vean el texto completo dentro de 256 subpalabras. |
| Métodos por palabra | Gradientes integrados (`IG`, 32 pasos, referencia `[PAD]`), gradiente por entrada (`IxG`), Attention Rollout, LRP-Chefer, LIME (200 perturbaciones) y SHAP de partición (300 evaluaciones). La unidad perturbada se borra, no se enmascara. |
| Métodos por oración | Explicación por oraciones (*leave-one-out*, determinista), LIME-oración (50 perturbaciones; 40 en el clasificador interpretable) y SHAP-oración. Los métodos por palabra se suman además por oración (`IG→oración`, etc.) para medir lo que se pierde con la agregación. |
| Aplicabilidad | Sin métodos de atención en redundancia; en el clasificador interpretable, no diferenciable, solo métodos por oración (Tabla 3.6 de la tesis). |
| Fidelidad | Curva de borrado en fracción de palabras retiradas q ∈ {0,1; …; 0,5}; AOPC; **ganancia sobre el azar de la misma granularidad** (AOPC del método menos la media de tres órdenes aleatorios de las mismas unidades); suficiencia y cambio de clase al 20 %. |
| Control de puntuación | Repetición de la curva de los métodos por palabra conservando la puntuación de cierre de las palabras borradas (`complementar()`). |
| Confiabilidad | Prueba de Adebayo parcial: reinicio de la cabeza y de las tres capas superiores (solo las capas en redundancia), Spearman entre explicaciones, 15 noticias por modelo; no aplica al clasificador interpretable. |
| Inferencia estadística | IC 95 % por *bootstrap* (2.000 réplicas); explicación por oraciones frente a cada método con Wilcoxon pareado y corrección de Holm **dentro de cada modelo**. |

## Archivos

| Archivo | Contenido |
|:---|:---|
| `codigos/xai_metodos.py` | Objetivos explicados (`Clasificador`, `Redundancia`, `ModeloVariables`), segmentación (palabras, oraciones Punkt, segmentos), los métodos de atribución y las métricas de borrado. Dispositivo: GPU si existe; `XAI_DISPOSITIVO=cpu` la desactiva. |
| `codigos/comparar_explicabilidad.py` | Construye la muestra, calcula atribuciones y métricas por noticia (caché reanudable en `codigos/cache/<modelo>/<id>.json`), la prueba de Adebayo (`adebayo_<id>.json`), el control de puntuación y consolida todo en `reportes/datos/metricas_comparacion_xai.json`. |
| `codigos/generar_reporte_comparacion_xai.py` | Calcula medias, IC y pruebas a partir del JSON; escribe `reportes/Reporte_Comparativo_XAI_Literatura_vs_Propia.md` y las figuras de `reportes/imagenes/`. |
| `codigos/cache/` | 100 noticias por modelo y 15 archivos de Adebayo en cada modelo neuronal; permite reproducir todo sin recalcular explicaciones. |
| `reportes/datos/muestra_comparacion_xai.csv` | Las 100 noticias de la muestra. |
| `reportes/datos/metricas_comparacion_xai.json` | Resultados consolidados por noticia (fuente de las Tablas 4.9 a 4.12 y 4.18). |

## Ejecución

Todos los comandos se lanzan desde la raíz del repositorio, con las dependencias de `requirements.txt`, el modelo `es_core_news_sm` de spaCy y Punkt de NLTK.

```bash
# 1) Cálculo completo (GPU recomendada). Reanuda desde la caché; con la caché incluida no recalcula nada.
python Explicabilidad_Propia/XAI_Literatura_Comparacion/codigos/comparar_explicabilidad.py [jjneila sabert redundancia modelo_variables]
#    Opciones: --solo-complementar (solo control de puntuación), --solo-consolidar (solo reconstruye el JSON desde la caché)
#    Variables de entorno: XAI_DISPOSITIVO=cpu (forzar CPU), XAI_HILOS=<n> (hilos de torch, para correr dos modelos en paralelo)

# 2) Reporte y figuras (sin GPU, segundos)
python Explicabilidad_Propia/XAI_Literatura_Comparacion/codigos/generar_reporte_comparacion_xai.py

# 3) Figuras con la nomenclatura de la tesis (sin GPU)
python Analisis_complementarios/figuras_oe3_renombradas.py
```

**Qué se reproduce sin GPU.** Con la caché y el JSON incluidos, los pasos 2 y 3 (y `--solo-consolidar`) regeneran todas las cifras y figuras de la Sección 4.3 sin cargar ningún modelo. Recalcular la caché exige cargar los cuatro modelos: el cálculo original se hizo en una GPU NVIDIA GeForce GTX 1650 (4 GB). Solo el tiempo de atribución registrado en el JSON suma unos 20 a 25 minutos por modelo neuronal; las curvas de borrado, los controles aleatorios y la prueba de Adebayo añaden muchas más evaluaciones del modelo. En el clasificador interpretable cada evaluación vuelve a extraer las 43 variables con tres modelos de lenguaje (con caché por texto).

**Nota sobre la corrección de tildes (NFC).** La caché y el JSON se calcularon antes de normalizar a NFC los 1.252 textos con tildes descompuestas (casi todos de Omdena; ver `Analisis_complementarios/corregir_nfc.py`). Los pasos 2 y 3 reproducen exactamente las cifras de la tesis; recalcular la caché con el código actual daría valores algo distintos, sobre todo en el clasificador interpretable, que se reentrena con la tabla de variables corregida.

## Resultados principales

Ganancia sobre el azar de la misma granularidad (media; Tabla 4.9 de la tesis, con IC en ella):

| Método | Sensacionalismo | Veracidad | Redundancia | Clasif. interpretable |
|:---|---:|---:|---:|---:|
| Gradientes integrados (palabra) | 0,208 | 0,174 | 0,020 | — |
| Gradiente por entrada (palabra) | 0,008 | 0,067 | −0,003 | — |
| Attention Rollout (palabra) | 0,092 | 0,110 | — | — |
| LRP-Chefer (palabra) | 0,245 | 0,158 | — | — |
| LIME (palabra) | 0,342 | 0,216 | 0,138 | — |
| SHAP (palabra) | 0,388 | 0,192 | 0,144 | — |
| LIME (oración) | 0,151 | 0,083 | 0,036 | 0,075 |
| SHAP (oración) | 0,150 | 0,084 | 0,036 | 0,072 |
| Expl. por oraciones | 0,146 | 0,077 | 0,043 | 0,072 |

1. **Por oración, la fidelidad de la explicación por oraciones no se distingue de la de LIME y SHAP** (Tabla 4.18): la diferencia no es significativa en sensacionalismo, veracidad y el clasificador interpretable (p de Holm entre 0,130 y 0,991), y es ligeramente favorable y significativa en redundancia (+0,007; p = 0,004 y 0,019). No encontrar diferencia no demuestra equivalencia. El *bootstrap* pareado de 10.000 réplicas ([`Analisis_complementarios/`](../../Analisis_complementarios/README.md)) coincide en siete de los ocho contrastes; la excepción es el clasificador interpretable frente a LIME-oración, con un intervalo apenas por debajo de 0 ([−0,0067; −0,0004]) aunque Wilcoxon-Holm da p = 0,130. Con 4,6 oraciones por noticia, los tres métodos son casi el mismo estimador. El acuerdo de orden con LIME y SHAP por oración va de ρ = 0,58 a 0,86 (Figura 4.19). Detalle en el [informe de explicabilidad](../../Informes_complementarios/Informe_explicabilidad.md).
2. **Sumar atribuciones por palabra a oraciones pierde fidelidad** (Tabla 4.10): los gradientes integrados bajan de 0,208 a 0,004 en sensacionalismo; la explicación por oraciones supera a la versión agregada en las 16 comparaciones (Wilcoxon-Holm, p < 0,05).
3. **Borrar palabras sueltas daña el texto** (Tabla 4.11): el borrado aleatorio de palabras reduce la salida 2,2 veces (sensacionalismo) y 3,6 veces (veracidad) más que el de oraciones, lo que justifica comparar cada método con el azar de su propia granularidad.
4. **Parte de la fidelidad por palabra en redundancia proviene de fusionar oraciones** (Tabla 4.12): al conservar la puntuación de cierre, LIME baja de 0,138 a 0,058 y SHAP de 0,144 a 0,097.
5. **Prueba de Adebayo** (Figura 4.16): Attention Rollout (ρ = 0,99 y 0,98) y LRP-Chefer (0,77 y 0,79) casi no cambian al aleatorizar el modelo en los dos clasificadores; los gradientes, LIME y SHAP por palabra sí cambian. Con 15 noticias los IC son amplios (p. ej., [0,06; 0,65] para la explicación por oraciones en sensacionalismo; ver `Analisis_complementarios/adebayo_ic.json`).
6. **Oraciones frente a segmentos** (Subsección 4.4.5): cortar también en comas da 12,6 segmentos por noticia frente a 4,6 oraciones y +0,044 de ganancia media, pero el 35 % de los segmentos no tiene verbo, frente al 4 % de las oraciones.

## Figuras: versión del repositorio y versión de la tesis

`reportes/imagenes/` contiene las figuras que genera `generar_reporte_comparacion_xai.py`, con la etiqueta interna «MexGen». Las incluidas en la tesis son las de [`Analisis_complementarios/figuras_oe3/`](../../Analisis_complementarios/figuras_oe3/), generadas con las mismas funciones y datos pero con la etiqueta «Expl. por oraciones» / «Expl. por segmentos». Ambas se conservan: la primera la produce el script de esta carpeta y la segunda la lee la tesis.

| Figura de la tesis | Versión de la tesis | Versión con etiqueta interna |
|:---|:---|:---|
| 4.14 Ganancia por método | [figuras_oe3](../../Analisis_complementarios/figuras_oe3/ganancia_fidelidad_por_metodo.png) | [imagenes](reportes/imagenes/ganancia_fidelidad_por_metodo.png) |
| 4.15 Curvas de borrado | [figuras_oe3](../../Analisis_complementarios/figuras_oe3/curvas_borrado.png) | [imagenes](reportes/imagenes/curvas_borrado.png) |
| 4.16 Prueba de Adebayo | [figuras_oe3](../../Analisis_complementarios/figuras_oe3/prueba_adebayo.png) | [imagenes](reportes/imagenes/prueba_adebayo.png) |
| 4.17 Borrado aleatorio | [figuras_oe3](../../Analisis_complementarios/figuras_oe3/borrado_azar_palabras_vs_oraciones.png) | [imagenes](reportes/imagenes/borrado_azar_palabras_vs_oraciones.png) (idéntica: no lleva nombres de métodos) |
| 4.18 Ejemplo oración frente a palabra | [figuras_oe3](../../Analisis_complementarios/figuras_oe3/ejemplo_oracion_vs_palabra.png) | [imagenes](reportes/imagenes/ejemplo_oracion_vs_palabra.png) |
| 4.19 Acuerdo de orden | [figuras_oe3](../../Analisis_complementarios/figuras_oe3/acuerdo_con_mexgen.png) | [imagenes](reportes/imagenes/acuerdo_con_mexgen.png) |
| No incluida (costo frente a ganancia) | [figuras_oe3](../../Analisis_complementarios/figuras_oe3/costo_frente_a_ganancia.png) | [imagenes](reportes/imagenes/costo_frente_a_ganancia.png) |

## Limitaciones

- Las métricas de borrado sacan al modelo de su distribución; la ganancia sobre el azar de la misma granularidad lo atenúa, pero no lo elimina. Miden de qué partes del texto depende la salida, no si la salida es correcta.
- La muestra excluye noticias de más de 200 palabras y de menos de tres oraciones, por lo que no cubre todo el rango de longitudes del corpus.
- LIME y SHAP se ejecutaron con un número limitado de perturbaciones; con más, su fidelidad podría variar.
- La prueba de Adebayo es parcial (un único reinicio de las capas superiores, no en cascada) y usa 15 noticias por modelo: sus resultados indican dirección, no magnitudes precisas.
- La curva de borrado por oraciones comparte la operación de los tres métodos por oración (retirar oraciones completas).

## Archivos generados

`codigos/cache/<modelo>/*.json`, `reportes/datos/muestra_comparacion_xai.csv`, `reportes/datos/metricas_comparacion_xai.json`, `reportes/Reporte_Comparativo_XAI_Literatura_vs_Propia.md` (se crea al ejecutar el paso 2; no se incluye en el repositorio) y las siete figuras de `reportes/imagenes/`.
