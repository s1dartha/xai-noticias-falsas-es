# ¿Basta con leer la noticia? Detección de desinformación con Transformers, patrones lingüísticos y explicabilidad

Repositorio de código y resultados del trabajo de grado:

> Alba Tobías, N. y Camacho Gutiérrez, D. S. (2026). *¿Basta con leer la noticia? Detección de desinformación con Transformers, patrones lingüísticos, explicabilidad y los límites del texto aislado*. Trabajo de grado, Programa de Ciencia de Datos, Departamento de Matemáticas, Universidad Externado de Colombia. Tutor: Alber Ferney Montenegro, Ph.D.

El trabajo estudia la detección de noticias falsas en español con fuentes de España y de América Latina, y propone un marco de explicabilidad que trabaja sobre 43 variables lingüísticas de estilo. Este repositorio contiene todo lo necesario para reproducir las cifras, tablas y figuras de la tesis. Cada carpeta principal tiene un `README.md` que funciona como reporte del módulo: explica qué hace cada script, las decisiones metodológicas y su relación con el documento.

## Marco propuesto

![Marco propuesto](Diagramas/marco_propuesto.png)

Una noticia se describe con 43 variables de estilo, que se obtienen con tres modelos de lenguaje usados sin ajuste fino (sensacionalismo, similitud semántica y emociones), con spaCy y con reglas. Sobre esas variables trabajan tres componentes:

- **Índice de Manipulación Lingüística (IML) y estilos discursivos:** describen *cómo* está escrito el texto.
- **Clasificador interpretable (Modelo sobre Variables):** estima si el texto tiene un perfil de noticia falsa.
- **Explicación por oraciones:** adaptación de MExGen por *leave-one-out*; señala qué oración sostiene cada resultado. En el código y en los diagramas aparece con el nombre corto «MexGen».

## Correspondencia entre la tesis y el repositorio

| Objetivo / parte de la tesis | Carpeta | Reporte |
|:---|:---|:---|
| Corpus base (2.604) y ampliado (4.418), deduplicación y etiquetas | `Modelos_Individuales/Clasificar_fake/Dataset/` | [README](Modelos_Individuales/Clasificar_fake/Dataset/README.md) |
| OE1: tres clasificadores Transformer de la literatura | `Modelos_Individuales/Clasificar_fake/Modelos_Literatura/` | [README](Modelos_Individuales/Clasificar_fake/Modelos_Literatura/README.md) |
| OE1: SaBERT por región y fuente; máscaras frente a cambio de dominio | `Modelos_Individuales/Clasificar_fake/SABERT_Evaluacion/` | [README](Modelos_Individuales/Clasificar_fake/SABERT_Evaluacion/README.md) |
| OE2: instrumento de sensacionalismo | `Modelos_Individuales/Sensacionalismo/` | [README](Modelos_Individuales/Sensacionalismo/README.md) |
| OE2: instrumento de redundancia semántica | `Modelos_Individuales/Redundancia/` | [README](Modelos_Individuales/Redundancia/README.md) |
| OE2: instrumento de emociones | `Modelos_Individuales/Emociones/` | [README](Modelos_Individuales/Emociones/README.md) |
| OE2: 43 variables y clasificador interpretable evaluado sin fuga | `Modelos_Individuales/Clasificar_fake/Modelo_sobre_Variables/` | [README](Modelos_Individuales/Clasificar_fake/Modelo_sobre_Variables/README.md) |
| OE2 y OE4: IML, estilos discursivos y auditoría de un texto nuevo | `Explicabilidad_Propia/` | [README](Explicabilidad_Propia/README.md) |
| OE3 y OE4: comparación con seis métodos de explicabilidad de la literatura | `Explicabilidad_Propia/XAI_Literatura_Comparacion/` | [README](Explicabilidad_Propia/XAI_Literatura_Comparacion/README.md) |
| Análisis de robustez y casos de estudio finales (OE1 a OE4) | `Analisis_complementarios/` | [README](Analisis_complementarios/README.md) |
| Diagramas de flujo del marco y de cada proceso | `Diagramas/` | [README](Diagramas/README.md) |
| Informes que amplían la tesis: explicabilidad, IML y estilos discursivos | `Informes_complementarios/` | [Explicabilidad](Informes_complementarios/Informe_explicabilidad.md) · [Estilos discursivos](Informes_complementarios/Informe_estilos_discursivos.md) |

## Estructura

```text
.
├── config_proyecto.py        # rutas, etiquetas (falsa = 1), modelos de Hugging Face con revisión fija, semilla 42
├── graficos_comunes.py       # estilo común de las figuras
├── ETIQUETAS.md              # convención de etiquetas por archivo y por modelo
├── DATOS_Y_DESCARGAS.md      # enlaces de descarga de cada conjunto y procesamiento aplicado
├── LICENSE                   # MIT (solo el código)
├── requirements.txt          # dependencias principales
├── requirements_lock.txt     # entorno exacto con el que se obtuvieron los resultados
├── Modelos_Individuales/
│   ├── Clasificar_fake/
│   │   ├── Dataset/                 # corpus
│   │   ├── Modelos_Literatura/      # 3 clasificadores de la literatura
│   │   ├── SABERT_Evaluacion/       # auditoría de SaBERT
│   │   └── Modelo_sobre_Variables/  # 43 variables y clasificador interpretable
│   ├── Sensacionalismo/
│   ├── Redundancia/
│   └── Emociones/
├── Explicabilidad_Propia/
│   ├── codigos/                     # variables de un texto, IML, estilos, auditor de estilo
│   └── XAI_Literatura_Comparacion/  # explicación por oraciones frente a IG, Gradiente por entrada,
│                                    # Attention Rollout, LRP-Chefer, LIME y SHAP
├── Analisis_complementarios/        # sensibilidad, casi duplicados, IC de Adebayo, casos finales
├── Informes_complementarios/        # informes ampliados de explicabilidad y de estilos discursivos
└── Diagramas/
```

## Instalación

Se usó Python 3.12.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps pysentimiento==0.7.3
python -m spacy download es_core_news_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

`torch` debe instalarse en la variante que corresponda al equipo (CPU o CUDA). Para reproducir exactamente el entorno verificado puede usarse `requirements_lock.txt`. Los modelos de Hugging Face se descargan la primera vez que se usan, siempre en la revisión fijada en `config_proyecto.py`.

Todos los scripts se ejecutan desde la raíz del repositorio, por ejemplo:

```bash
python Modelos_Individuales/Sensacionalismo/Evaluacion_Modelos/evaluar_modelos_sensacionalismo.py
```

## Niveles de reproducción

1. **Sin GPU y en minutos.** Los JSON de resultados, las predicciones y las cachés de inferencia (`cache/`) están incluidos. Los scripts `generar_reporte_*.py`, los análisis de `Analisis_complementarios/` y la evaluación del clasificador interpretable regeneran las cifras y figuras de la tesis a partir de ellos.
2. **Completa.** La inferencia con los modelos de lenguaje (clasificadores, extracción de variables y métodos de explicabilidad) se puede repetir desde cero. Tarda horas en CPU y se recomienda GPU. Los scripts de explicabilidad usan la GPU si está disponible; `XAI_DISPOSITIVO=cpu` lo evita.

El orden completo es el de los capítulos de la tesis:

| Paso | Script principal | Detalle |
|:---:|:---|:---|
| 1 | `Clasificar_fake/Dataset/construir_corpus_latam.py` | ampliación del corpus con fuentes de América Latina |
| 2 | `Clasificar_fake/Modelos_Literatura/Codigos/run_3_modelos_fakenews.py` | benchmark de 3 clasificadores |
| 3 | `Clasificar_fake/SABERT_Evaluacion/Codigos/auditoria_censura_vs_domain_shift.py` | SaBERT por región y fuente |
| 4 | `Sensacionalismo/…/evaluar_modelos_sensacionalismo.py`, `Redundancia/pipeline_redundancia.py`, `Emociones/…/evaluar_modelos_emociones.py` | selección y validación de instrumentos |
| 5 | `Clasificar_fake/Modelo_sobre_Variables/Codigos/` (extracción → tabla → selección → evaluación sin fuga) | variables y clasificador interpretable |
| 6 | `Explicabilidad_Propia/codigos/estilos_discursivos.py` | IML y estilos discursivos |
| 7 | `Explicabilidad_Propia/XAI_Literatura_Comparacion/codigos/comparar_explicabilidad.py` | comparación de métodos de explicabilidad |
| 8 | `Analisis_complementarios/` | robustez y casos de estudio |

Los comandos exactos y los tiempos de cada paso están en el README de cada carpeta.

## Convenciones

- En todo el repositorio la noticia **falsa es la clase positiva (1)** y la real es 0 (ver [`ETIQUETAS.md`](ETIQUETAS.md)).
- La semilla global es 42, y cada modelo de Hugging Face se fija por *commit* en `config_proyecto.MODELOS_HF`.
- Los scripts no escriben en la carpeta LaTeX de la tesis salvo que se defina la variable de entorno `TESIS_IMG_DIR`.

## Datos de terceros

Los corpus y bancos de evaluación externos (FakeDeS, Unified, Edds, MEX-A3T, Omdena, TA1C, NoticIA, EmoEvent, STS Benchmark, PAWS-X y titulares de salud, entre otros) conservan la licencia y las condiciones de uso de sus autores. Se incluyen ya procesados, solo para reproducir los resultados. [`DATOS_Y_DESCARGAS.md`](DATOS_Y_DESCARGAS.md) reúne, para cada conjunto, el enlace de descarga original, qué parte se usó y el procesamiento que hay que aplicarle para obtener exactamente nuestras cifras. Quien los reutilice debe citar las fuentes originales.

## Licencia

El código se publica bajo licencia MIT ([`LICENSE`](LICENSE)). La licencia no se extiende a los datos de terceros.

## Autores

- Natali Alba Tobías ([@s1dartha](https://github.com/s1dartha))
- Daniel Santiago Camacho Gutiérrez ([@danisanty06](https://github.com/danisanty06))
