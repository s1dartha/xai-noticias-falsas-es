"""
Configuración central del repositorio XAI-para-Noticias-falsas.

Define en un solo lugar:
  1. Las rutas del proyecto, relativas a la raíz del repositorio.
  2. La codificación canónica de etiquetas (FAKE = 1, REAL = 0).
  3. Los modelos de Hugging Face con su revisión fija y el índice de salida de la clase positiva.
  4. La semilla global y la exportación opcional de figuras a la carpeta de la tesis.

Uso desde cualquier script del repositorio:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config_proyecto.py").exists())))
    import config_proyecto as cfg
"""

import os
import shutil
from pathlib import Path

# =============================================================================
# 1. RUTAS DEL PROYECTO
# =============================================================================
ROOT = Path(__file__).resolve().parent

MODELOS_DIR = ROOT / "Modelos_Individuales"
DIAGRAMAS_DIR = ROOT / "Diagramas"   # diagramas de flujo del marco y de cada proceso (Diagramas/generar_diagramas.py)

# Clasificación de noticias falsas
CLASIFICAR_DIR = MODELOS_DIR / "Clasificar_fake"
DATASET_FAKE_DIR = CLASIFICAR_DIR / "Dataset"
DATASET_INTERMEDIOS_DIR = DATASET_FAKE_DIR / "archivos_intermedios_extraccion"

LITERATURA_DIR = CLASIFICAR_DIR / "Modelos_Literatura"
LITERATURA_CODIGOS_DIR = LITERATURA_DIR / "Codigos"
LITERATURA_REPORTES_DIR = LITERATURA_DIR / "Reportes"
LITERATURA_IMG_DIR = LITERATURA_REPORTES_DIR / "Imagenes"

MODELO_VARIABLES_DIR = CLASIFICAR_DIR / "Modelo_sobre_Variables"
MODELO_VARIABLES_CODIGOS_DIR = MODELO_VARIABLES_DIR / "Codigos"
MODELO_VARIABLES_REPORTES_DIR = MODELO_VARIABLES_DIR / "Reportes"
MODELO_VARIABLES_IMG_DIR = MODELO_VARIABLES_REPORTES_DIR / "Imagenes"

SABERT_DIR = CLASIFICAR_DIR / "SABERT_Evaluacion"
SABERT_CODIGOS_DIR = SABERT_DIR / "Codigos"
SABERT_REPORTES_DIR = SABERT_DIR / "Reportes"
SABERT_IMG_DIR = SABERT_REPORTES_DIR / "Imagenes"

# Redundancia semántica
REDUNDANCIA_DIR = MODELOS_DIR / "Redundancia"
REDUNDANCIA_DATASET_DIR = REDUNDANCIA_DIR / "Dataset"
REDUNDANCIA_REPORTES_DIR = REDUNDANCIA_DIR / "Reportes"

# Sensacionalismo
SENSACIONALISMO_DIR = MODELOS_DIR / "Sensacionalismo"
SENSACIONALISMO_DATASET_DIR = SENSACIONALISMO_DIR / "Dataset"
SENSACIONALISMO_REPORTES_DIR = SENSACIONALISMO_DIR / "reportes"

# Emociones
EMOCIONES_DIR = MODELOS_DIR / "Emociones"
EMOCIONES_DATASET_DIR = EMOCIONES_DIR / "Dataset"
EMOCIONES_REPORTES_DIR = EMOCIONES_DIR / "Reportes"

# Explicabilidad propia (MEXGEN, IML y estilos discursivos)
EXPLICABILIDAD_DIR = ROOT / "Explicabilidad_Propia"
EXPLICABILIDAD_CODIGOS_DIR = EXPLICABILIDAD_DIR / "codigos"
EXPLICABILIDAD_REPORTES_DIR = EXPLICABILIDAD_DIR / "reportes"
EXPLICABILIDAD_GRAFICOS_DIR = EXPLICABILIDAD_DIR / "graficos"

# -----------------------------------------------------------------------------
# Archivos de datos canónicos (una sola copia por archivo)
# -----------------------------------------------------------------------------
CORPUS_BASE_XLSX = DATASET_FAKE_DIR / "Noticias_entre_70_y_370_palabras (1).xlsx"
CORPUS_BASE_CSV = DATASET_FAKE_DIR / "Noticias_entre_70_y_370_palabras (1).csv"
CORPUS_AMPLIADO_CSV = DATASET_FAKE_DIR / "Noticias_entre_70_y_370_palabras_AMPLIADO_LATAM.csv"
CORPUS_AMPLIADO_XLSX = DATASET_FAKE_DIR / "Noticias_entre_70_y_370_palabras_AMPLIADO_LATAM.xlsx"
PREDICCIONES_SABERT_4418_CSV = DATASET_FAKE_DIR / "predicciones_sabert_ampliado_4418.csv"

REDUNDANCIA_BENCH_DIR = REDUNDANCIA_DATASET_DIR / "benchmarks_externos"                # STS-B es y PAWS-X es
REDUNDANCIA_CORPUS_CSV = REDUNDANCIA_DATASET_DIR / "redundancia_corpus_hiiamsid.csv"   # umbrales calibrados con STS
METRICAS_REDUNDANCIA_JSON = REDUNDANCIA_REPORTES_DIR / "metricas_redundancia_hiiamsid.json"

# Modelo sobre Variables (Fases 4 y 5)
FEATURES_AVANZADO_4418_CSV = MODELO_VARIABLES_CODIGOS_DIR / "features_modelo_sobre_variables_ampliado_avanzado_4418.csv"  # extracción de origen de S, R, O y L
MODELO_VARIABLES_DATASET_DIR = MODELO_VARIABLES_DIR / "Dataset"
TABLA_VARIABLES_CSV = MODELO_VARIABLES_DATASET_DIR / "tabla_variables_4418.csv"            # variables S, R, E, O, L por noticia
TABLA_VARIABLES_META_JSON = MODELO_VARIABLES_DATASET_DIR / "tabla_variables_4418_meta.json"
SELECCION_MODELO_JSON = MODELO_VARIABLES_REPORTES_DIR / "metricas_seleccion_modelo_sobre_variables.json"
METRICAS_MODELO_VARIABLES_SIN_FUGA_JSON = MODELO_VARIABLES_REPORTES_DIR / "metricas_modelo_sobre_variables_sin_fuga.json"
PREDICCIONES_MODELO_VARIABLES_SIN_FUGA_CSV = MODELO_VARIABLES_REPORTES_DIR / "predicciones_modelo_sobre_variables_sin_fuga.csv"

METRICAS_SENSACIONALISMO_JSON = SENSACIONALISMO_REPORTES_DIR / "metricas_modelos_sensacionalismo.json"
METRICAS_EMOCIONES_JSON = EMOCIONES_REPORTES_DIR / "metricas_modelos_emociones.json"

ESTILOS_DISCURSIVOS_CSV = EXPLICABILIDAD_REPORTES_DIR / "iml_estilos_discursivos_4418.csv"          # Fase 5: IML y estilo por noticia
METRICAS_ESTILOS_DISCURSIVOS_JSON = EXPLICABILIDAD_REPORTES_DIR / "metricas_iml_estilos_discursivos.json"
REFERENCIA_ESTILOS_JSON = EXPLICABILIDAD_CODIGOS_DIR / "referencia_iml_estilos.json"            # percentiles y centroides para perfilar textos nuevos
XAI_COMPARACION_DIR = EXPLICABILIDAD_DIR / "XAI_Literatura_Comparacion"
XAI_COMPARACION_DATOS_DIR = XAI_COMPARACION_DIR / "reportes" / "datos"

# Benchmark de clasificadores de la literatura (3 modelos)
CACHE_PREDICCIONES_3_MODELOS_NPZ = LITERATURA_CODIGOS_DIR / "predictions_cache_2604.npz"
PREDICCIONES_3_MODELOS_CSV = LITERATURA_REPORTES_DIR / "predicciones_3_modelos_2604.csv"
METRICAS_3_MODELOS_JSON = LITERATURA_REPORTES_DIR / "metricas_3_modelos_fakenews.json"

# =============================================================================
# 2. ETIQUETAS CANÓNICAS
# =============================================================================
# Convención única para todo el repositorio: la noticia falsa es la clase positiva.
FAKE = 1
REAL = 0
NOMBRES_CLASE = {REAL: "Real (0)", FAKE: "Falsa (1)"}

_VALORES_REAL = {"TRUE", "REAL", "VERDADERA", "VERDADERO", "0"}
_VALORES_FAKE = {"FALSE", "FAKE", "FALSA", "FALSO", "1"}


def etiqueta_binaria(valor):
    """Convierte cualquier codificación de veracidad de los corpus a FAKE (1) / REAL (0).

    Acepta booleanos (True = noticia verdadera, como en la columna `class` del corpus base),
    enteros 0/1 (ya canónicos) y textos como REAL, VERDADERA, FAKE o FALSA.
    Lanza ValueError ante valores desconocidos en lugar de asignar una clase por defecto.
    """
    if isinstance(valor, bool) or type(valor).__name__ == "bool_":
        return REAL if bool(valor) else FAKE
    texto = str(valor).strip().upper()
    if texto in _VALORES_REAL:
        return REAL
    if texto in _VALORES_FAKE:
        return FAKE
    raise ValueError(f"Etiqueta de veracidad no reconocida: {valor!r}")


# =============================================================================
# 3. MODELOS DE HUGGING FACE (revisión fija e índice de la clase positiva)
# =============================================================================
# `idx_positivo` es la posición de la salida softmax que corresponde a la clase positiva
# (FAKE en los clasificadores de veracidad; sensacionalista en JJNeila). Verificado con
# el `id2label` de cada config.json el 2026-10-24. En Juanillaberia el config no define
# etiquetas; el índice 0 como FAKE se confirmó empíricamente (ROC-AUC 0,86 frente a 0,14
# con la asignación opuesta sobre el corpus base).
MODELOS_HF = {
    "M1_Narrativaai": {
        "id": "Narrativaai/fake-news-detection-spanish",
        "revision": "0eab6d956dbfc035dfd4ef2d0d40700876401ad3",
        "idx_positivo": 1,  # id2label = {0: REAL, 1: FAKE}
    },
    "M2_SaBERT": {
        "id": "VerificadoProfesional/SaBERT-Spanish-Fake-News",
        "revision": "c092c970eda10216ea2da846813745c0074a6ca2",
        "idx_positivo": 0,  # id2label = {0: False (noticia falsa), 1: True}
    },
    "M3_Juanillaberia": {
        "id": "Juanillaberia/spanish-fake-news-classifier",
        "revision": "163af625e4695e5d73256775bf8d9f65c8c28c13",
        "idx_positivo": 0,  # sin id2label; verificado empíricamente
    },
    "Sensacionalismo_JJNeila": {
        "id": "JJNeila/bert-spanish-sensationalism-oss",
        "revision": "1b8140efe548c876cfc03357c86100ede9118440",
        "idx_positivo": 1,  # sin id2label en el config; la tarjeta del modelo indica 1 = Sensationalism.
                            # Verificado (Fase 3): AUC de P[:,1] = 1,00 en IA sintético, 0,69 en Salud (prueba)
                            # y 0,64 en Amarillismo humano (titular + copete + cuerpo).
        # Modelo de sensacionalismo elegido en la Fase 3 entre 5 candidatos (ver
        # Modelos_Individuales/Sensacionalismo/README.md). Umbral calibrado en
        # TA1C validation + mitad de Salud; no alcanza F1 macro >= 0,65 y se declara como limitación.
        "umbral_calibrado": 0.85,
    },
    "SBERT_Redundancia": {
        "id": "hiiamsid/sentence_similarity_spanish_es",
        "revision": "66ab46adac3910bb6ea6085b962a25e49513b981",
    },
    "Emociones_RoBERTuito": {
        "id": "pysentimiento/robertuito-emotion-analysis",
        "revision": "bcd6835f4d1ab1a061bd7437c9d762623c8437ad",
        # Modelo de emociones elegido en la Fase 3b entre 3 candidatos (ver
        # Modelos_Individuales/Emociones/README.md). id2label = {0: others, 1: joy, 2: sadness,
        # 3: anger, 4: surprise, 5: disgust, 6: fear}; leer las salidas por nombre, no por índice. Requiere el
        # preprocesamiento de pysentimiento y admite como máximo 128 tokens.
        "max_len": 128,
    },
}

# =============================================================================
# 4. SEMILLA Y EXPORTACIÓN OPCIONAL A LA TESIS
# =============================================================================
SEED = 42

# La copia de figuras a la carpeta LaTeX de la tesis está desactivada por defecto.
# Para activarla, defina la variable de entorno TESIS_IMG_DIR con la ruta de destino.
TESIS_IMG_DIR = Path(os.environ["TESIS_IMG_DIR"]) if os.environ.get("TESIS_IMG_DIR") else None


def exportar_figuras_tesis(rutas_figuras, subcarpeta=""):
    """Copia figuras a TESIS_IMG_DIR/subcarpeta solo si la variable de entorno está definida."""
    if TESIS_IMG_DIR is None:
        return []
    destino = TESIS_IMG_DIR / subcarpeta
    destino.mkdir(parents=True, exist_ok=True)
    copiadas = []
    for ruta in rutas_figuras:
        ruta = Path(ruta)
        if ruta.exists():
            shutil.copy2(ruta, destino / ruta.name)
            copiadas.append(destino / ruta.name)
    return copiadas
