# Convención de etiquetas del repositorio

Verificada cruzando cada archivo con la verdad del corpus (unión por texto normalizado u orden de filas comprobado). La función de referencia es `config_proyecto.etiqueta_binaria()`.

## 1. Veracidad (clasificación de noticias falsas)

**Regla única: `1` = noticia falsa (clase positiva), `0` = noticia real.**

| Archivo | Columna(s) | Codificación | Verificación |
|:---|:---|:---|:---|
| `Clasificar_fake/Dataset/Noticias_entre_70_y_370_palabras (1).xlsx` (2.604) | `class` (bool) | `True` = **real**, `False` = falsa | 1.345 reales / 1.259 falsas |
| ídem | `label_num` · `label_name` · `categoria` | `0`/`REAL`/`VERDADERA` = real; `1`/`FAKE`/`FALSA` = falsa | 100 % |
| `Noticias_entre_70_y_370_palabras (1).csv` (2.602) | `clase_num` · `categoria` | `1`/`FALSA` = falsa | 100 % |
| `…_AMPLIADO_LATAM.csv/.xlsx` (4.418) | `clase_num` · `categoria` | `1`/`FALSA` = falsa | 2.457 reales / 1.961 falsas (tras corregir Omdena) |
| `Dataset/predicciones_sabert_ampliado_4418.csv` | `clase_num`; `prob_fake` = P(falsa) | `pred_sabert` = (`prob_fake` > 0,5) | 100 % |
| `Modelo_sobre_Variables/Dataset/tabla_variables_4418.csv` | `label_num` | `1` = falsa (desde `clase_num` con `etiqueta_binaria`) | 100 % |
| `Modelo_sobre_Variables/Codigos/features_modelo_sobre_variables_*.csv` | `label_num` (= `clase_num`) | `1` = falsa | 100 % |
| `Modelo_sobre_Variables/Reportes/predicciones_modelo_sobre_variables_sin_fuga.csv` | `label_num` | `1` = falsa | 100 % |
| `Modelos_Literatura/Reportes/predicciones_3_modelos_2604.csv` | `y_true`; `prob_m*` = P(falsa) | `1` = falsa | 100 % |
| `Redundancia/Dataset/redundancia_corpus_hiiamsid.csv` | `label_num` | `1` = falsa | 100 % |
| `Explicabilidad_Propia/reportes/iml_estilos_discursivos_4418.csv` | `label_num` | `1` = falsa | 100 % |
| `XAI_Literatura_Comparacion/reportes/datos/muestra_comparacion_xai.csv` | `label_num` | `1` = falsa | 100 % |

> **Corrección 2026-10-24 (Omdena).** Las 1.250 noticias de Omdena estaban etiquetadas con la columna `Prediction` del dataset original, que es la salida de un clasificador del proyecto Omdena. Se reemplazó por `Corrected_label`: 138 noticias cambian de clase (92 de falsa a real y 46 de real a falsa). La comparación fila a fila está en `Clasificar_fake/Dataset/omdena_etiquetas_prediction_vs_corrected.csv`.

## 2. Sensacionalismo

**Regla única: `1` = sensacionalista (o clickbait en TA1C y NoticIA), `0` = no sensacionalista.**

| Archivo | Columna | Codificación | Verificación |
|:---|:---|:---|:---|
| `Sensacionalismo/Dataset/trainset_realnews.csv`, `trainset_paper_and_corpus.csv`, `externos/echucompa/data_testset_realnews.csv` | `Amarillismo` | `Amarillista` = 1 | Mapeo explícito; falla ante valores desconocidos |
| `dataset_Salud.csv` (2.200) | `clase` | `1` = sensacionalista | Contenido y AUC de JJNeila |
| `dataset_IA_sintetico_70.csv` | `Clase` | `Sensacionalista` = 1 | Mapeo explícito |
| `externos/ta1c/TA1C_dataset_*_complete.csv` | `Tag Value` | `Clickbait` = 1 | Mapeo explícito |
| `externos/noticia/` (NoticIA) | — | todas clickbait (1) | Solo se usa para recall |

## 3. Emociones

| Archivo | Columna | Codificación | Verificación |
|:---|:---|:---|:---|
| `Emociones/Dataset/externos/emoevent/test.tsv` | `emotion` | `others`, `joy`, `sadness`, `anger`, `surprise`, `disgust`, `fear` | **La columna está desplazada 30 filas en el archivo publicado**: la etiqueta se toma de `emoevent_es.csv` cruzando por texto (ver `Modelos_Individuales/Emociones/README.md`) |

## 4. Modelos de Hugging Face (índice de salida)

| Modelo | `id2label` | Salida usada | Verificación |
|:---|:---|:---|:---|
| `Narrativaai/fake-news-detection-spanish` (RoBERTa-large-BNE) | `{0: REAL, 1: FAKE}` | `softmax[:, 1]` = P(falsa) | config.json |
| `VerificadoProfesional/SaBERT-Spanish-Fake-News` | `{0: False, 1: True}` | `softmax[:, 0]` = P(falsa) | config.json + AUC |
| `Juanillaberia/spanish-fake-news-classifier` | no definido | `softmax[:, 0]` = P(falsa) | Empírica: ROC-AUC 0,86 frente a 0,14 con el índice 1 |
| `JJNeila/bert-spanish-sensationalism-oss` | no definido | `softmax[:, 1]` = P(sensacionalista) | Tarjeta del modelo + AUC en IA sintético, Salud y Amarillismo humano |
| `pysentimiento/robertuito-emotion-analysis` | `{0: others, 1: joy, 2: sadness, 3: anger, 4: surprise, 5: disgust, 6: fear}` | se lee por nombre de etiqueta | config.json |

Los índices y las revisiones fijas de cada modelo están en `config_proyecto.MODELOS_HF`.

## 5. Reglas para el código

- Para convertir una etiqueta de veracidad, usar siempre `cfg.etiqueta_binaria(valor)`: lanza un error ante valores desconocidos en lugar de asignar una clase por defecto.
- Fijar siempre el orden de clases: `confusion_matrix(y, pred, labels=[0, 1])`.
- `precision_recall_curve` devuelve `(precision, recall, umbrales)`: desempaquetar en ese orden.
- Al unir archivos por posición, comprobar antes que el texto coincide fila a fila.
- En los modelos multiclase, leer las salidas por el nombre de la etiqueta (`id2label`), no por el índice.
