# Datos: dónde se descargan y cómo se procesaron

El repositorio **ya incluye los datos procesados** con los que se obtuvieron los resultados de la tesis, así que no hace falta descargar nada para reproducirlos. Este archivo sirve para otra cosa: ver de dónde viene cada conjunto, descargarlo desde su fuente original y saber qué procesamiento le aplicamos. Si se parte de los archivos originales, hay que repetir ese procesamiento para obtener exactamente nuestras cifras.

Los conjuntos son de sus autores y conservan sus licencias. Cítense las fuentes originales al reutilizarlos. La licencia MIT del repositorio ([`LICENSE`](LICENSE)) cubre solo nuestro código.

Convención común: **1 = clase positiva** (falsa, sensacionalista, *clickbait* o paráfrasis, según el conjunto) y semilla 42 en todo muestreo.

---

## 1. Corpus de noticias falsas y reales

Archivos incluidos en `Modelos_Individuales/Clasificar_fake/Dataset/`:

- `Noticias_entre_70_y_370_palabras (1).xlsx`: corpus base, 2.604 noticias.
- `Noticias_entre_70_y_370_palabras_AMPLIADO_LATAM.csv`: corpus ampliado, 4.418 noticias. Es el archivo canónico.

| Conjunto | Descarga | Qué se toma | Etiqueta usada | Noticias en el corpus |
|:---|:---|:---|:---|---:|
| Unified (Freiren) | <https://huggingface.co/datasets/Freiren/Unified-and-Balanced-Spanish-Fake-News-Corpus> | Todo el conjunto | `class` (True = real → 0) | 2.060 |
| Edds | <https://huggingface.co/datasets/Edds/spanish-fake-news-fixed> | Todo el conjunto | `class` (True = real → 0) | 294 |
| FakeDeS (mariagrandury) | <https://huggingface.co/datasets/mariagrandury/fake_news_corpus_spanish> | Particiones publicadas en Hugging Face | `CATEGORY` | 248 |
| Kaggle | <https://www.kaggle.com/datasets/arseniitretiakov/noticias-falsas-en-espaol> | 2 registros (falsos) | `class` | 2 (solo en el corpus base) |
| MEX-A3T | <https://github.com/jpposadas/FakeNewsCorpusSpanish> | `train.xlsx` y `development.xlsx` | `Category` (Fake → 1) | 566 |
| Omdena LatAm | <https://huggingface.co/datasets/IsaacRodgz/Fake-news-latam-omdena> | Particiones `train` y `test`, solo `Type == "Web Article"` | `Corrected_label` (no `Prediction`) | 1.250 |

**Procesamiento aplicado, en orden:**

1. **Filtro de longitud.** Se conservan las noticias de 70 a 370 palabras, contando palabras separadas por espacios (`conteo_palabras_text`). Es el mismo criterio para todas las fuentes.
2. **Etiqueta binaria.** Se convierte con `config_proyecto.etiqueta_binaria`, que rechaza cualquier valor no reconocido en lugar de asignar una clase por defecto.
3. **Normalización mínima.** Solo se quitan los espacios de los extremos. La puntuación, las mayúsculas, las tildes y la ortografía se dejan intactas, porque se miden como variables de estilo. Las máscaras de MEX-A3T (`*NUMBER*`, `*URL*`, `*PHONE*`) se conservan.
4. **Región por fuente.** Unified y Edds se asignan a España; FakeDeS, MEX-A3T y Omdena, a América Latina.
5. **Ampliación sin redundancia.** Solo se añaden noticias de MEX-A3T y Omdena cuyo texto normalizado (minúsculas, sin espacios en los extremos) no esté ya en el corpus base. Los 2 registros de Kaggle no entran en el corpus ampliado.
6. **Duplicados.** El archivo canónico **conserva** los duplicados entre fuentes: 318 registros, porque los 294 textos de Edds y 23 de FakeDeS también están en Unified. Cada análisis que entrena o elige parámetros los elimina conservando la primera aparición, lo que deja **4.100 textos únicos**.
7. **Tildes en NFC al medir.** Casi todos los textos de Omdena (1.249 de 1.250) y 3 de Unified vienen con las tildes en Unicode descompuesto (NFD). El archivo se conserva tal cual, pero la extracción de variables, el auditor y SaBERT normalizan el texto a NFC antes de medir ([`Analisis_complementarios/corregir_nfc.py`](Analisis_complementarios/README.md)).

**Qué está automatizado.** `Modelos_Individuales/Clasificar_fake/Dataset/construir_corpus_latam.py` descarga MEX-A3T y Omdena y realiza los pasos 1, 2, 3 y 5 para esas fuentes. En cambio, la unión de Unified, Edds, FakeDeS y Kaggle y la asignación de región se hicieron a mano (Apéndice A.3 de la tesis).

Además, el script no deja el archivo idéntico al publicado: conserva columnas de trabajo, incluye los registros de Kaggle y **sobrescribe** el CSV. Por eso, para obtener nuestros resultados hay que usar el archivo incluido, y el script solo sirve para verificar la ampliación. Si se ejecuta, conviene hacerlo sobre una copia del repositorio.

Detalles y conteos por fuente: [`Dataset/README.md`](Modelos_Individuales/Clasificar_fake/Dataset/README.md).

## 2. Conjuntos para validar el instrumento de sensacionalismo

Carpeta: `Modelos_Individuales/Sensacionalismo/Dataset/`.

| Conjunto | Descarga | Archivo incluido | Procesamiento |
|:---|:---|:---|:---|
| TA1C (Mordecki y Moncecchi, 2025) | <https://github.com/gmordecki/TA1C> | `externos/ta1c/TA1C_dataset_validation_complete.csv` y `..._test_complete.csv` | Particiones `validation` (calibración del umbral) y `test` (referencia), 700 cada una. Se evalúan dos entradas: solo titular (128 subpalabras) y titular + cuerpo (256). |
| Titulares de salud (Díaz Román et al.) | <https://github.com/gonzalezheber/Text-classifier-of-sensationalist-headlines-in-Spanish-using-BERT-based-models> | `dataset_Salud.csv` (2.200) | Partición estratificada 50/50 con semilla 42: 1.100 de calibración y 1.100 de prueba. |
| Amarillismo (EchuCompa, UBA) | <https://github.com/EchuCompa/Sensationalism-Detector> | `trainset_realnews.csv` (14), `trainset_paper_and_corpus.csv` (59) y `externos/echucompa/data_testset_realnews.csv` (41) | Se unen los tres archivos de noticias reales (114). Se **excluye** `trainset_gptnews.csv` (textos generados con GPT). `Amarillista` → 1. |
| NoticIA (García-Ferrero y Altuna, 2024) | <https://huggingface.co/datasets/Iker/NoticIA> | `externos/noticia/test-00000-of-00001.parquet` | Partición `test` (100). Todas las noticias son *clickbait*, así que solo se mide la exhaustividad. |
| Textos sintéticos de control | Propios (no se descargan) | `dataset_IA_sintetico_70.csv` | 70 textos contrastivos (35 + 35), generados para este trabajo como control de cordura. |

## 3. Conjuntos para validar la redundancia semántica

Carpeta: `Modelos_Individuales/Redundancia/Dataset/benchmarks_externos/`.

| Conjunto | Descarga | Archivo incluido | Procesamiento |
|:---|:---|:---|:---|
| STS Benchmark en español | <https://huggingface.co/datasets/PhilipMay/stsb_multi_mt> (configuración `es`) | `stsb_es_dev.parquet` (1.500), `stsb_es_test.parquet` (1.379) | `dev` calibra los umbrales y `test` evalúa. Se binariza con puntaje ≥ 1, ≥ 3 o ≥ 4, según el nivel. |
| PAWS-X en español | <https://huggingface.co/datasets/google-research-datasets/paws-x> (configuración `es`) | `pawsx_es_validation.parquet`, `pawsx_es_test.parquet` (2.000 cada uno) | Validación externa; `label = 1` significa paráfrasis. |

## 4. Conjunto para validar las emociones

Carpeta: `Modelos_Individuales/Emociones/Dataset/externos/emoevent/`.

| Conjunto | Descarga | Archivos incluidos | Procesamiento |
|:---|:---|:---|:---|
| EmoEvent (Plaza-del-Arco et al., 2020) | <https://github.com/fmplaza/EmoEvent>, partición `splits/es` (descargada el 24/10/2026) | `test.tsv`, `train.tsv`, `dev.tsv`, `emoevent_es.csv` | **Corrección necesaria:** en `test.tsv` la columna `emotion` está desplazada 30 filas respecto de los textos. La etiqueta de cada tuit de prueba se toma del corpus completo (`emoevent_es.csv`), cruzando por el texto normalizado (`etiquetas_test_corregidas`). El cruce se valida con `train` y `dev`. Quedan **1.615 tuits** evaluados de 1.656 filas. Sin esta corrección, los resultados no coinciden con los de la tesis. |

## 5. Modelos preentrenados

No hay que descargarlos a mano. Cada script los obtiene de Hugging Face la primera vez, en la revisión (*commit*) fijada en `config_proyecto.MODELOS_HF` (Tabla A.2 de la tesis). Cambiar la revisión puede cambiar los resultados.

Los cuatro modelos de *clickbait* que se comparan en la selección del instrumento de sensacionalismo no tienen revisión fija en el código. La revisión que realmente se usó quedó registrada en `Modelos_Individuales/Sensacionalismo/reportes/metricas_modelos_sensacionalismo.json`:

| Modelo | Revisión usada |
|:---|:---|
| `ggcastle/beto-clickbait-es` | `03c31ac` |
| `dcere/ta1c-Clickbait-Detector-es` | `8e4fd58` |
| `dcere/ta1c-Clickbait-Detector-es-large` | `25e713a` |
| `christinacdl/XLM_RoBERTa-Multilingual-Clickbait-Detection` | `1a02f7a` |

## 6. Cómo comprobar que se parte de los mismos datos

| Archivo | Filas esperadas |
|:---|---:|
| `Noticias_entre_70_y_370_palabras (1).xlsx` | 2.604 |
| `Noticias_entre_70_y_370_palabras (1).csv` (sin Kaggle) | 2.602 |
| `Noticias_entre_70_y_370_palabras_AMPLIADO_LATAM.csv` | 4.418 (4.100 textos únicos) |
| `dataset_Salud.csv` | 2.200 |
| Amarillismo (3 archivos) | 114 (58 amarillistas) |
| TA1C validación / prueba | 700 / 700 |
| EmoEvent prueba corregida | 1.615 |

`python Modelos_Individuales/Clasificar_fake/Dataset/generar_reporte_corpus.py` vuelve a calcular la composición del corpus (Tabla 3.2), los duplicados y las longitudes, y permite compararlos con la tesis.
