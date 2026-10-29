"""
Genera los diagramas de flujo del marco propuesto y de cada proceso del repositorio (PNG en Diagramas/).

    python Diagramas/generar_diagramas.py            # todos
    python Diagramas/generar_diagramas.py mexgen     # uno o varios por nombre

Las cifras que aparecen en los diagramas se copian de los reportes generados (cada función indica de cuál);
si un reporte cambia, hay que actualizar el diagrama correspondiente.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config_proyecto.py").exists())))
import config_proyecto as cfg
import graficos_comunes as gc

OUT = cfg.DIAGRAMAS_DIR
ESTILOS = ['Institucional / Técnico', 'Opinión / Crónica', 'Alarmista / Viral']


def alto(n, esc, tam=10, tam_det=8.5, titulo=1):
    """Alto (en unidades del lienzo) de una caja_lista con `n` líneas de detalle."""
    return (7 + titulo * tam * 1.3 + 4 + n * tam_det * 1.35 + 9) / 72 / esc


# =============================================================================
# 1. MARCO PROPUESTO (tres pilares)
# =============================================================================
def marco_propuesto():
    """Fuentes: variables_texto.py, estilos_discursivos.py, auditor_estilo.py, xai_metodos.py,
    Modelos_Individuales/Clasificar_fake/Modelo_sobre_Variables/README.md y Explicabilidad_Propia/README.md."""
    E = 0.62
    fig, ax = gc.lienzo_completo(17, 20.5, E)
    gc.titulo_diagrama(ax, 0.3, 19.6, 'Marco propuesto: estilo, perfil y explicación de una noticia',
                       'Tres pilares sobre las mismas 43 variables de estilo; ninguno usa el contenido factual para decidir.')

    gc.caja_lista(ax, 8.5, 17.6, 7, alto(1, E), 'Noticia en texto plano', ['español; título y cuerpo como un solo texto'], 'entrada')
    gc.flecha(ax, (8.5, 17.6 - alto(1, E) / 2), (8.5, 16.55))
    h = alto(4, E)
    gc.caja_lista(ax, 8.5, 16.5 - h / 2, 15.6, h, 'Extracción de variables (variables_texto.py)', [
        'Segmentación en oraciones (Punkt, español)',
        'JJNeila → sensacionalismo (S, 4) · hiiamsid → redundancia (R, 6) · RoBERTuito → emociones (E, 12)',
        'spaCy y expresiones regulares → marcadores ortotipográficos (O, 6) y rasgos lingüísticos (L, 15)',
        'Los tres transformers se usan sin ajuste fino y con revisión fija'], 'modelo')
    y_var = 16.5 - h - 0.95
    gc.flecha(ax, (8.5, 16.5 - h), (8.5, y_var + 0.38))
    gc.caja(ax, 8.5, y_var, 5.2, 0.75, '43 variables por noticia', tipo='entrada')

    # Pilares 1 y 2
    top, bot = y_var - 0.95, 5.15
    gc.grupo(ax, 0.3, bot, 8.2, top, 'PILAR 1 · Estilos discursivos', 'no supervisado · descriptivo', color='#2a78d6')
    gc.grupo(ax, 8.8, bot, 16.7, top, 'PILAR 2 · Modelo sobre Variables', 'supervisado · predictivo', color='#eb6834', lado='der')
    gc.flecha(ax, (6.9, y_var - 0.38), (6.9, top - 1.05))
    gc.flecha(ax, (10.1, y_var - 0.38), (10.1, top - 1.05))
    ax.text(6.75, top + 0.3, 'se usan 28', ha='right', va='center', fontsize=8.5, color=gc.TINTA_2, style='italic')
    ax.text(10.25, top + 0.3, 'se usan las 43', ha='left', va='center', fontsize=8.5, color=gc.TINTA_2, style='italic')

    x1, x2, w = 4.25, 12.75, 7.2
    y = top - 1.05
    h1 = alto(5, E)
    gc.caja_lista(ax, x1, y - h1 / 2, w, h1, '5 dimensiones (percentiles del corpus)', [
        'D1 Sensacionalismo · D2 Carga emocional', 'D3 Reiteración · D4 Énfasis ortotipográfico',
        'D5 Aserción sin respaldo', 'cada variable con dirección fijada a priori',
        'IML = media de D1–D5 (0–100)'], 'proceso')
    h2 = alto(5, E)
    gc.caja_lista(ax, x2, y - h2 / 2, w, h2, 'Regresión logística (L2, C = 1)', [
        'imputación por mediana + estandarización', 'evaluada sin fuga: 4.100 textos únicos,',
        'StratifiedGroupKFold(5) y umbral anidado', 'F1 macro 0,625 (SaBERT: 0,780;',
        'variables + SaBERT: 0,798)'], 'modelo')
    y_b = y - max(h1, h2) - 0.45
    h3 = alto(3, E)
    gc.flecha(ax, (x1, y - h1), (x1, y_b))
    gc.caja_lista(ax, x1, y_b - h3 / 2, w, h3, 'k-medias sobre D1–D5 (k = 3)', [
        'k elegido por silueta y estabilidad (ARI ≥ 0,75)', 'pertenencia a cada estilo = inverso de la',
        'distancia a su centro, normalizado (%)'], 'proceso')
    gc.flecha(ax, (x2, y - h2), (x2, y_b))
    gc.caja_lista(ax, x2, y_b - h3 / 2, w, h3, 'Lectura del modelo para la noticia', [
        'P(falsa) frente al umbral (mediana de los anidados)', 'aporte de cada variable en log-odds:',
        'coeficiente × valor estandarizado (z)'], 'proceso')
    y_c = y_b - h3 - 0.45
    h4 = alto(3, E, tam=11)
    gc.flecha(ax, (x1, y_b - h3), (x1, y_c))
    gc.caja_lista(ax, x1, y_c - h4 / 2, w, h4, '¿CÓMO está escrito el texto?', [
        'radar de 5 dimensiones, IML y estilo dominante:', ' · '.join(ESTILOS[:2]) + ' ·', ESTILOS[2]], 'salida', tam=11)
    gc.flecha(ax, (x2, y_b - h3), (x2, y_c))
    gc.caja_lista(ax, x2, y_c - h4 / 2, w, h4, '¿TIENE perfil de noticia falsa?', [
        'probabilidad, clase predicha y las variables', 'que más la empujan hacia «falsa» o «real»',
        '(perfil de estilo, no verificación de hechos)'], 'salida', tam=11)

    # Pilar 3
    h5 = alto(3, E, tam=11)
    top3 = bot - 0.9
    y5 = top3 - 1.0 - h5 / 2
    gc.grupo(ax, 0.3, y5 - h5 / 2 - 0.35, 16.7, top3, 'PILAR 3 · Explicación local',
             'MexGen adaptado, por oraciones', color='#1baf7a')
    gc.flecha(ax, (6.9, y_c - h4), (6.9, y5 + h5 / 2))
    gc.flecha(ax, (10.1, y_c - h4), (10.1, y5 + h5 / 2))
    ax.text(6.75, bot - 0.45, 'f = IML: oraciones que suben o bajan el IML', ha='right', va='center', fontsize=8.5, color=gc.TINTA_2, style='italic')
    ax.text(10.25, bot - 0.45, 'f = P(falsa): oraciones que la mueven', ha='left', va='center', fontsize=8.5, color=gc.TINTA_2, style='italic')
    gc.caja_lista(ax, 8.5, y5, 15.6, h5, '¿QUÉ oración exacta lo provocó?', [
        'Δᵢ = f(texto) − f(texto sin la oración i), para cada oración i; la noticia se corta en puntos (Punkt), no en comas',
        'xai_metodos.mexgen; en el auditor (AuditorEstilo.mexgen) una extracción de variables por versión da ΔIML y ΔP(falsa)',
        'fidelidad que no se distingue de la de LIME y SHAP por oración; más barato y determinista (Informes_complementarios/Informe_explicabilidad.md)'],
        'salida', tam=11)
    ax.set_ylim(y5 - h5 / 2 - 0.6, 19.75)
    return fig


# =============================================================================
# 2. PIPELINE GENERAL
# =============================================================================
def pipeline_general():
    E = 0.6
    fig, ax = gc.lienzo_completo(20, 8.8, E)
    gc.titulo_diagrama(ax, 0.3, 8.6, 'Fases del proyecto de punta a punta',
                       'Cada fase tiene un script que produce un JSON y un reporte generado a partir de ese JSON.')
    fases = [
        ('1 · Corpus', ['4.418 noticias, 5 fuentes', 'España y América Latina', '4.100 textos únicos'],
         'construir_corpus_latam.py', 'entrada'),
        ('2 · Clasificadores', ['3 transformers de la literatura', 'SaBERT: auditoría por región,', 'fuente y máscaras'],
         'run_3_modelos_fakenews.py', 'modelo'),
        ('3 · Instrumentos', ['sensacionalismo: 5 modelos → JJNeila', 'redundancia: hiiamsid (STS-B)', 'emociones: 3 modelos → RoBERTuito'],
         'evaluar_modelos_*.py, pipeline_redundancia.py', 'modelo'),
        ('4 · Tabla de variables', ['43 variables por noticia', 'grupos S, R, E, O y L'],
         'construir_tabla_variables.py', 'proceso'),
        ('5 · Modelo sobre Variables', ['regresión logística elegida', 'entre 3 familias; evaluación', 'sin fuga de información'],
         'seleccionar_… / evaluar_…_sin_fuga.py', 'modelo'),
        ('6 · IML y estilos', ['28 variables → 5 dimensiones', 'IML y k-medias (k = 3)', '3 estilos discursivos'],
         'estilos_discursivos.py', 'proceso'),
        ('7 · Explicabilidad', ['MexGen por oraciones frente a', 'IG, IxG, Rollout, LRP, LIME y SHAP', 'en 4 modelos × 100 noticias'],
         'comparar_explicabilidad.py', 'proceso'),
        ('8 · Dashboard', ['auditoría de un texto nuevo:', 'estilo, perfil y explicación', '(pendiente de actualizar)'],
         'dashboard/server.py', 'salida'),
    ]
    w, h = 4.3, alto(3, E)
    xs = [2.55, 7.45, 12.35, 17.25]
    filas = [6.3, 2.55]
    pos = [(xs[i % 4], filas[i // 4]) for i in range(8)]
    for (t, lineas, script, tipo), (x, y) in zip(fases, pos):
        gc.caja_lista(ax, x, y, w, h, t, lineas, tipo)
        ax.text(x, y - h / 2 - 0.22, script, ha='center', va='top', fontsize=7.5, color=gc.TINTA_SUAVE, family='monospace')
    for i in range(3):
        for f in filas:
            gc.flecha(ax, (xs[i] + w / 2, f), (xs[i + 1] - w / 2, f))
    gc.flecha_ruta(ax, [(xs[3], filas[0] - h / 2 - 0.6), (xs[3], 4.45), (xs[0], 4.45), (xs[0], filas[1] + h / 2)])
    gc.leyenda_tipos(ax, 0.4, 0.35, paso=2.6)
    return fig


# =============================================================================
# 3. PROCESOS
# =============================================================================
def construccion_corpus():
    """Fuentes: construir_corpus_latam.py y Modelos_Individuales/Clasificar_fake/Dataset/README.md."""
    E = 0.62
    fig, ax = gc.lienzo_completo(17, 13.2, E)
    gc.titulo_diagrama(ax, 0.3, 13.0, 'Construcción del corpus ampliado', 'construir_corpus_latam.py → Noticias_entre_70_y_370_palabras_AMPLIADO_LATAM.csv')
    h = alto(3, E)
    gc.caja_lista(ax, 4.3, 10.0, 7.6, h, 'Corpus base: 2.602 noticias', [
        'Unified 2.060 · Edds 294 · FakeDeS (mariagrandury) 248',
        '(el benchmark de 3 modelos usa además',
        '2 registros de Kaggle sin región: 2.604)'], 'entrada')
    gc.caja_lista(ax, 12.7, 10.0, 7.6, h, 'Fuentes de América Latina', [
        'MEX-A3T (train y dev) y FakeDeS 2021 (jpposadas)', 'Omdena LATAM: etiqueta Corrected_label',
        '(antes Prediction: 138 noticias cambiaron de clase)'], 'entrada')
    h2 = alto(2, E)
    gc.flecha(ax, (12.7, 10.0 - h / 2), (12.7, 7.6 + h2 / 2))
    gc.caja_lista(ax, 12.7, 7.6, 7.6, h2, 'Unificar, deduplicar y filtrar', [
        'texto idéntico → se conserva una copia', 'solo noticias de 70 a 370 palabras'], 'proceso')
    gc.flecha(ax, (12.7, 7.6 - h2 / 2), (12.7, 5.3 + h2 / 2))
    gc.caja_lista(ax, 12.7, 5.3, 7.6, alto(3, E), 'Agregar solo lo nuevo: 1.816 noticias', [
        'las que no estaban en el corpus base (por texto)', 'Omdena 1.250 · MEX-A3T 566',
        '(FakeDeS 2021 no aporta noticias nuevas)'], 'proceso')
    h3 = alto(5, E)
    y3 = 2.3
    gc.flecha_ruta(ax, [(4.3, 10.0 - h / 2), (4.3, y3 + h3 / 2)])
    gc.flecha(ax, (12.7, 5.3 - alto(3, E) / 2), (12.7, y3 + h3 / 2))
    ax.text(4.45, 7.0, 'se conserva completo', ha='left', va='center', fontsize=8.5, color=gc.TINTA_2, style='italic')
    gc.caja_lista(ax, 8.5, y3, 16, h3, 'Corpus ampliado: 4.418 noticias (2.457 reales, 1.961 falsas)', [
        'Unified 2.060 · Edds 294 · FakeDeS (mariagrandury) 248 · FakeNewsCorpusSpanish (MEX-A3T) 566 · Omdena 1.250',
        'España 2.354 (46,3 % falsas) · América Latina 2.064 (42,2 % falsas)',
        '4.100 textos únicos: 317 textos aparecen en más de una fuente, ninguno con etiquetas contradictorias',
        'las evaluaciones del Modelo sobre Variables, el IML y la explicabilidad usan los textos únicos',
        'etiqueta canónica: 1 = falsa, 0 = real (ETIQUETAS.md)'], 'salida')
    return fig


def evaluacion_sensacionalismo():
    """Fuente: Modelos_Individuales/Sensacionalismo/README.md."""
    E = 0.62
    fig, ax = gc.lienzo_completo(17, 12.6, E)
    gc.titulo_diagrama(ax, 0.3, 12.4, 'Elección del instrumento de sensacionalismo', 'evaluar_modelos_sensacionalismo.py (sin ajuste fino)')
    h = alto(4, E)
    gc.caja_lista(ax, 4.3, 9.3, 7.6, h, 'Calibración del umbral', [
        'TA1C validation (titular + cuerpo, clickbait)', 'Salud: mitad de calibración',
        'un umbral por modelo: maximiza F1 macro', 'los conjuntos de prueba no se usan aquí'], 'entrada')
    gc.caja_lista(ax, 12.7, 9.3, 7.6, h, 'Conjuntos de prueba', [
        'Salud (mitad de prueba) · Amarillismo humano (114)', 'TA1C test (clickbait, referencia externa)',
        'IA sintético (control) · NoticIA (solo recall)', 'no hay corpus público de sensacionalismo anotado'], 'entrada')
    h2 = alto(3, E)
    gc.flecha(ax, (4.3, 9.3 - h / 2), (7.0, 6.4 + h2 / 2 - 0.05))
    gc.flecha(ax, (12.7, 9.3 - h / 2), (10.0, 6.4 + h2 / 2 - 0.05))
    gc.caja_lista(ax, 8.5, 6.4, 9.5, h2, '5 modelos de Hugging Face', [
        'JJNeila (sensacionalismo) y 4 detectores de clickbait:', 'ggcastle BETO, dcere TA1C base y large, XLM-R (christinacdl)',
        'revisión fija; índice de la clase positiva verificado'], 'modelo')
    h3 = alto(2, E)
    gc.flecha(ax, (8.5, 6.4 - h2 / 2), (8.5, 4.1 + h3 / 2))
    gc.caja_lista(ax, 8.5, 4.1, 9.5, h3, 'Criterio: F1 macro ≥ 0,65 en sensacionalismo', [
        'ninguno lo alcanza; el mejor es JJNeila (0,557 de media)', 'los de clickbait quedan cerca del azar (otro constructo)'], 'decision')
    gc.flecha(ax, (8.5, 4.1 - h3 / 2), (8.5, 1.8 + alto(1, E) / 2))
    gc.caja_lista(ax, 8.5, 1.8, 12, alto(1, E), 'JJNeila como instrumento (umbral calibrado 0,85)', [
        'P(sensacionalista) del documento y de cada oración → variables S y dimensión D1'], 'salida')
    return fig


def evaluacion_redundancia():
    """Fuente: Modelos_Individuales/Redundancia/README.md."""
    E = 0.62
    fig, ax = gc.lienzo_completo(17, 13.4, E)
    gc.titulo_diagrama(ax, 0.3, 13.2, 'Validación del instrumento de redundancia', 'pipeline_redundancia.py · hiiamsid/sentence_similarity_spanish_es')
    h = alto(1, E)
    gc.caja_lista(ax, 8.5, 10.6, 9, h, 'Noticia → oraciones (Punkt) → embeddings hiiamsid', [
        'similitud coseno entre todos los pares de oraciones del documento'], 'modelo')
    h2 = alto(4, E)
    gc.flecha(ax, (6.5, 10.6 - h / 2), (3.5, 7.6 + h2 / 2))
    gc.flecha(ax, (8.5, 10.6 - h / 2), (8.5, 7.6 + h2 / 2))
    gc.flecha(ax, (10.5, 10.6 - h / 2), (13.5, 7.6 + h2 / 2))
    gc.caja_lista(ax, 3.0, 7.6, 5.4, h2, 'STS-B en español', [
        'pares anotados por personas (0–5)', 'Spearman con el juicio humano 0,823',
        'umbrales por Youden en dev:', 'τ tema 0,41 · paráfrasis 0,59 · redundante 0,72'], 'entrada')
    gc.caja_lista(ax, 8.5, 7.6, 5.2, h2, 'PAWS-X en español', [
        'pares adversariales con las', 'mismas palabras', 'ROC-AUC 0,56: el coseno no', 'los separa → limitación'], 'entrada')
    gc.caja_lista(ax, 14.0, 7.6, 5.4, h2, 'Corpus (textos únicos)', [
        'falsas frente a reales', '(U de Mann-Whitney, Holm)', 'AUC entre 0,486 y 0,503:', 'no discrimina la veracidad'], 'entrada')
    h3 = alto(3, E)
    for x in (3.0, 8.5, 14.0):
        gc.flecha(ax, (x, 7.6 - h2 / 2), (8.5 + (x - 8.5) * 0.35, 4.4 + h3 / 2))
    gc.caja_lista(ax, 8.5, 4.4, 12, h3, 'Decisión', [
        'la redundancia se usa como descriptor del estilo (cuánto reitera una noticia), no como indicador de falsedad',
        'variables R continuas y sin umbrales: máxima, media, top-3, varianza, consecutivas media y mínima',
        'τ redundante (0,7232) solo marca en el auditor un par que «repite la misma afirmación»'], 'decision')
    y_f = 4.4 - h3 / 2 - 1.2
    gc.flecha(ax, (8.5, 4.4 - h3 / 2), (8.5, y_f + 0.4))
    gc.caja(ax, 8.5, y_f, 9, 0.8, 'Variables R (6) y dimensión D3 Reiteración', tipo='salida')
    return fig


def evaluacion_emociones():
    """Fuente: Modelos_Individuales/Emociones/README.md y variables_texto.py."""
    E = 0.62
    fig, ax = gc.lienzo_completo(17, 13.4, E)
    gc.titulo_diagrama(ax, 0.3, 13.2, 'Elección del instrumento de emociones', 'evaluar_modelos_emociones.py (sin ajuste fino; 7 clases, argmax)')
    h = alto(4, E)
    gc.caja_lista(ax, 4.3, 9.9, 7.6, h, 'EmoEvent / EmoEvalEs (prueba)', [
        '1.615 tuits en español anotados por personas', 'el archivo publicado tiene la etiqueta desplazada',
        '30 filas: se tomó del corpus completo cruzando', 'por texto (en train y dev coincide al 99,98–100 %)'], 'entrada')
    gc.caja_lista(ax, 12.7, 9.9, 7.6, h, 'Noticias del corpus (sin etiqueta)', [
        '301 noticias estratificadas por región y clase', 'no existe corpus de noticias en español',
        'anotado con emociones: se mide el acuerdo', 'entre modelos como transferencia indirecta'], 'entrada')
    h2 = alto(3, E)
    gc.flecha(ax, (4.3, 9.9 - h / 2), (7.2, 6.9 + h2 / 2))
    gc.flecha(ax, (12.7, 9.9 - h / 2), (9.8, 6.9 + h2 / 2))
    gc.caja_lista(ax, 8.5, 6.9, 10, h2, '3 modelos con publicación revisada por pares', [
        'RoBERTuito (pysentimiento): F1 macro 0,565', 'BETO (pysentimiento): 0,520 · XLM-R (GSI-UPM): 0,438',
        'salidas leídas por el nombre de cada etiqueta (id2label)'], 'modelo')
    gc.flecha(ax, (8.5, 6.9 - h2 / 2), (8.5, 4.35 + h2 / 2))
    gc.caja_lista(ax, 8.5, 4.35, 10, h2, 'RoBERTuito como instrumento', [
        'se aplica oración por oración (128 tokens, preprocesamiento de pysentimiento)',
        'asco no se usa como variable (F1 = 0 en los tres modelos)', 'ningún modelo alcanza F1 macro ≥ 0,65 (criterio de referencia)'], 'decision')
    gc.flecha(ax, (8.5, 4.35 - h2 / 2), (8.5, 1.55 + 0.55))
    gc.caja_lista(ax, 8.5, 1.55, 13, alto(1, E), 'Variables E (12) y dimensión D2 Carga emocional', [
        'intensidad = 1 − P(sin emoción) · ira, miedo, tristeza, alegría y sorpresa · media y máximo por noticia'], 'salida')
    return fig


def tabla_variables():
    """Fuentes: variables_texto.py, construir_tabla_variables.py y tabla_variables_4418_meta.json."""
    E = 0.6
    fig, ax = gc.lienzo_completo(20, 13.6, E)
    gc.titulo_diagrama(ax, 0.3, 13.4, 'De la noticia a las 43 variables', 'variables_texto.ExtractorVariables · construir_tabla_variables.py → tabla_variables_4418.csv')
    gc.caja_lista(ax, 10, 10.3, 9, alto(1, E), 'Noticia → oraciones (Punkt, español)', ['cada grupo se calcula con su propia herramienta'], 'entrada')
    grupos = [
        ('S · Sensacionalismo (4)', 'JJNeila', ['P del documento (1.000 caracteres)', 'P media y máxima por oración', 'desviación entre oraciones'], 'modelo'),
        ('R · Redundancia (6)', 'hiiamsid', ['similitud máxima, media, top-3', 'varianza de las similitudes', 'consecutivas: media y mínima'], 'modelo'),
        ('E · Emociones (12)', 'RoBERTuito', ['intensidad emocional', 'ira, miedo, tristeza, alegría,', 'sorpresa (media y máximo)'], 'modelo'),
        ('O · Ortotipográficos (6)', 'expresiones regulares', ['mayúsculas sostenidas y proporción', 'exclamaciones, interrogaciones,', 'puntos suspensivos, comillas'], 'proceso'),
        ('L · Lingüísticos (15)', 'spaCy y expresiones regulares', ['atenuadores, intensificadores, verbos', 'de atribución, legibilidad, riqueza', 'léxica, pronombres, cifras, longitud'], 'proceso'),
    ]
    w, h = 3.75, alto(4, E)
    xs = [2.05 + i * 3.98 for i in range(5)]
    for x, (t, herr, lineas, tipo) in zip(xs, grupos):
        gc.flecha(ax, (10 + (x - 10) * 0.35, 10.3 - alto(1, E) / 2), (x, 6.9 + h / 2))
        gc.caja_lista(ax, x, 6.9, w, h, t, [f'[{herr}]'] + lineas, tipo, tam=9.5, tam_detalle=8)
        gc.flecha(ax, (x, 6.9 - h / 2), (10 + (x - 10) * 0.35, 3.6 + alto(2, E) / 2))
    gc.caja_lista(ax, 10, 3.6, 12, alto(2, E), 'Tabla de 43 variables (4.418 noticias)', [
        'una fila por noticia; las mismas variables alimentan el IML (28 con dirección) y la regresión logística (43)',
        'para un texto nuevo el extractor es el mismo, verificado contra la tabla (diferencia máxima 1,3e-6)'], 'salida')
    ax.text(10, 3.6 - alto(2, E) / 2 - 0.35, 'Las emociones y el sensacionalismo por oración se agregan por noticia; la redundancia necesita al menos 2 oraciones.',
            ha='center', fontsize=8, color=gc.TINTA_SUAVE, style='italic')
    return fig


def evaluacion_sin_fuga():
    """Fuentes: evaluar_modelo_sobre_variables_sin_fuga.py y Modelos_Individuales/Clasificar_fake/Modelo_sobre_Variables/README.md."""
    E = 0.62
    fig, ax = gc.lienzo_completo(17, 15.2, E)
    gc.titulo_diagrama(ax, 0.3, 15.0, 'Evaluación sin fuga del Modelo sobre Variables', 'seleccionar_modelo_sobre_variables.py · evaluar_modelo_sobre_variables_sin_fuga.py')
    h = alto(2, E)
    h = alto(1, E)
    gc.caja_lista(ax, 8.5, 10.9, 10, h, 'Tabla de 43 variables → 4.100 textos únicos', [
        '1.836 falsas y 2.264 reales; ningún texto queda a la vez en entrenamiento y prueba'], 'entrada')
    gc.flecha(ax, (8.5, 10.9 - h / 2), (8.5, 9.75))
    gc.grupo(ax, 1.0, 4.35, 16.0, 9.75, 'Validación cruzada externa: StratifiedGroupKFold(5), grupo = texto', color='#2a78d6', tam=10)
    h2 = alto(3, E)
    gc.caja_lista(ax, 5.0, 8.4, 6.6, h2, 'Pliegues de entrenamiento (4/5)', [
        'imputación por mediana y estandarización', 'ajustadas solo aquí',
        'regresión logística (L2, C = 1, sin búsqueda)'], 'modelo')
    gc.caja_lista(ax, 12.2, 8.4, 6.0, h2, 'Umbral anidado', [
        'StratifiedGroupKFold(5) interna', 'sobre el entrenamiento', 'umbral que maximiza F1 macro'], 'decision')
    gc.flecha(ax, (5.0 + 3.3, 8.4), (12.2 - 3.0, 8.4))
    gc.caja_lista(ax, 8.5, 5.55, 9, alto(1, E), 'Pliegue de prueba (1/5)', ['se aplica el modelo y su umbral; nada se recalcula'], 'proceso')
    gc.flecha(ax, (5.0, 8.4 - h2 / 2), (5.0, 5.55 + alto(1, E) / 2))
    gc.flecha(ax, (12.2, 8.4 - h2 / 2), (12.2, 5.55 + alto(1, E) / 2))
    ax.text(5.15, 6.75, 'modelo ajustado', ha='left', va='center', fontsize=8.5, color=gc.TINTA_2, style='italic')
    ax.text(12.35, 6.75, 'umbral', ha='left', va='center', fontsize=8.5, color=gc.TINTA_2, style='italic')
    h3 = alto(4, E)
    gc.flecha(ax, (8.5, 4.35), (8.5, 2.6 + h3 / 2))
    gc.caja_lista(ax, 8.5, 2.6, 15, h3, 'Resultados (F1 macro, FP, FN, ROC-AUC; por región y excluyendo cada fuente)', [
        'selección de familia: regresión logística 0,625 · gradient boosting 0,603 · bosque aleatorio 0,579',
        'ablación por grupo: L 0,596 · S 0,590 · O 0,555 · E 0,550 · R 0,536',
        'Completo (43 variables) 0,625 · SaBERT 0,780 · Completo + SaBERT 0,798',
        'se elige la regresión logística: el mayor F1 de las tres y explicable variable por variable'], 'salida')
    return fig


def iml_y_estilos():
    """Fuentes: estilos_discursivos.py y metricas_iml_estilos_discursivos.json."""
    E = 0.6
    fig, ax = gc.lienzo_completo(19, 14.4, E)
    gc.titulo_diagrama(ax, 0.3, 14.2, 'IML y estilos discursivos', 'estilos_discursivos.py · referencia_iml_estilos.json')
    h = alto(3, E)
    gc.caja_lista(ax, 4.6, 11.0, 8.4, h, '28 de las 43 variables', [
        'solo las que tienen dirección fijada a priori por la literatura', '(↑ suma, ↓ resta; p. ej., más verbos de atribución = menos D5)',
        '15 se excluyen: legibilidad, longitud, tristeza, alegría…'], 'entrada')
    h2 = alto(3, E)
    gc.flecha(ax, (4.6, 11.0 - h / 2), (4.6, 8.3 + h2 / 2))
    gc.caja_lista(ax, 4.6, 8.3, 8.4, h2, 'Percentiles en el corpus', [
        'percentil de cada variable (invertido si su dirección es ↓)', 'media por dimensión y percentil de esa media',
        '→ D1–D5 en escala 0–100'], 'proceso')
    h3 = alto(5, E)
    gc.flecha(ax, (4.6, 8.3 - h2 / 2), (4.6, 5.0 + h3 / 2))
    gc.caja_lista(ax, 4.6, 5.0, 8.4, h3, '5 dimensiones', [
        'D1 Sensacionalismo (JJNeila)', 'D2 Carga emocional (RoBERTuito)', 'D3 Reiteración (hiiamsid)',
        'D4 Énfasis ortotipográfico', 'D5 Aserción sin respaldo'], 'proceso')
    gc.flecha(ax, (4.6, 5.0 - h3 / 2), (4.6, 1.5 + 0.55))
    gc.caja_lista(ax, 4.6, 1.5, 8.4, alto(1, E), 'IML = media de D1–D5', ['AUC 0,602 frente a la veracidad: describe el estilo, no decide'], 'salida')

    gc.flecha_ruta(ax, [(4.6 + 4.2, 5.0), (10.2, 5.0), (10.2, 11.0), (11.2, 11.0)])
    h4 = alto(3, E)
    gc.caja_lista(ax, 14.9, 11.0, 7.4, h4, 'k-medias sobre D1–D5', [
        'k de 3 a 6 (k = 2 solo separa «más» de «menos»)', 'se elige la mayor silueta entre los k estables',
        '(ARI medio ≥ 0,75 en 30 remuestreos) → k = 3'], 'decision')
    h5 = alto(2, E)
    gc.flecha(ax, (14.9, 11.0 - h4 / 2), (14.9, 8.3 + h5 / 2))
    gc.caja_lista(ax, 14.9, 8.3, 7.4, h5, 'Pertenencia a cada estilo', [
        'inverso de la distancia a cada centro,', 'normalizado: estilo dominante y secundario'], 'proceso')
    h6 = alto(6, E)
    gc.flecha(ax, (14.9, 8.3 - h5 / 2), (14.9, 4.0 + h6 / 2))
    gc.caja_lista(ax, 14.9, 4.0, 7.4, h6, '3 estilos discursivos', [
        f'1 · {ESTILOS[0]}: poco sensacionalismo y', 'emoción, reitera ideas clave, cita fuentes',
        f'2 · {ESTILOS[1]}: fluido y subjetivo,', 'sin énfasis visual',
        f'3 · {ESTILOS[2]}: dramático, emotivo y', 'con mayúsculas y signos'], 'salida')
    ax.text(14.9, 0.45, 'Las noticias forman un continuo (silueta ≈ 0,20): el estilo es una pertenencia gradual, no una etiqueta.',
            ha='center', fontsize=7.8, color=gc.TINTA_SUAVE, style='italic')
    return fig


def mexgen():
    """Fuentes: xai_metodos.mexgen y spans_oraciones; cifras de segmentación del reporte comparativo XAI.
    Los valores del ejemplo son ilustrativos."""
    E = 0.6
    fig, ax = gc.lienzo_completo(19, 15.0, E)
    gc.titulo_diagrama(ax, 0.3, 14.8, 'MexGen adaptado: leave-one-out por oraciones',
                       'xai_metodos.mexgen · adaptación de MExGen (Monteiro Paes et al., ACL 2025)')
    oraciones = ['O1  «El Gobierno anunció un nuevo impuesto.»', 'O2  «¡ES UN ROBO A TODOS LOS CIUDADANOS!»',
                 'O3  «La medida entrará en vigor en enero.»', 'O4  «Nadie se atreve a decirlo, pero es verdad.»']
    gc.caja_lista(ax, 4.9, 11.0, 9.2, alto(4, E), 'Noticia segmentada en oraciones (Punkt)', oraciones, 'entrada', alinear='left')
    gc.flecha(ax, (9.5, 11.0), (10.6, 11.0))
    gc.caja_lista(ax, 14.6, 11.0, 7.6, alto(4, E), 'k + 1 evaluaciones del modelo f', [
        'f(texto completo) = 0,82', 'f(sin O1) = 0,80 · f(sin O2) = 0,51', 'f(sin O3) = 0,83 · f(sin O4) = 0,70',
        '(f: P(falsa), P(sensacionalista)…)'], 'modelo')
    h = alto(5, E)
    gc.flecha(ax, (14.6, 11.0 - alto(4, E) / 2), (14.6, 8.4 + h / 2))
    gc.caja_lista(ax, 14.6, 8.4, 7.6, h, 'Δᵢ = f(texto) − f(texto sin Oᵢ)', [
        'O2: +0,31  ← la que más sostiene la salida', 'O4: +0,12', 'O1: +0,02', 'O3: −0,01  (quitarla la sube un poco)',
        '(valores ilustrativos)'], 'salida', alinear='left')
    gc.caja_lista(ax, 4.9, 8.4, 9.2, h, 'Qué se toma de MExGen y qué se simplifica', [
        'se toma: unidades lingüísticas y perturbación por eliminación', '(no por [MASK], que el modelo nunca vio)',
        'algoritmo leave-one-out: k + 1 evaluaciones, determinista', 'sin escalarizador: la salida ya es un número',
        'sin refinamiento a frases o palabras: se queda en la oración'], 'proceso', alinear='left')
    # Puntos frente a comas
    gc.grupo(ax, 0.3, 3.55, 18.7, 6.85, '¿Por qué cortar en puntos y no en comas?', color='#c98500', tam=10.5)
    gc.caja_lista(ax, 4.9, 4.9, 8.8, alto(4, E), 'Cortar también en comas (versión anterior)', [
        '«El ministro, | que llegó tarde, | anunció la medida.»', '→ «El ministro» y «que llegó tarde» no son ideas completas',
        '12,6 cláusulas por noticia; 35 % sin verbo', 'la coma suele unir una idea con su complemento'], 'nota', alinear='left')
    gc.caja_lista(ax, 14.1, 4.9, 8.8, alto(4, E), 'Cortar en puntos (versión actual)', [
        '«El ministro, que llegó tarde, anunció la medida.»', '→ una sola unidad: sujeto y lo que se dice de él',
        '4,6 oraciones por noticia; 4 % sin verbo', 'el punto marca con más certeza que la idea terminó'], 'salida', alinear='left')
    return fig


def comparacion_xai():
    """Fuentes: comparar_explicabilidad.py, xai_metodos.py y Explicabilidad_Propia/XAI_Literatura_Comparacion/README.md."""
    E = 0.6
    fig, ax = gc.lienzo_completo(19, 15.6, E)
    gc.titulo_diagrama(ax, 0.3, 15.4, 'Comparación de MexGen con seis métodos de la literatura', 'comparar_explicabilidad.py · xai_metodos.py · ejecutado en GPU (GTX 1650)')
    h = alto(3, E)
    gc.caja_lista(ax, 4.6, 12.0, 8.4, h, 'Muestra: 100 noticias por modelo', [
        'textos únicos, 25 por región × clase', 'como máximo 200 palabras (caben en 256 tokens)', 'al menos 3 oraciones'], 'entrada')
    gc.caja_lista(ax, 14.0, 12.0, 8.8, h, '4 modelos explicados', [
        'JJNeila: P(sensacionalista) · SaBERT: P(falsa)', 'hiiamsid: similitud media entre oraciones',
        'Modelo sobre Variables: P(falsa) (no diferenciable)'], 'modelo')
    h2 = alto(4, E)
    for x, t, l, tipo in [
        (3.4, 'Por palabra (nativos)', ['IG, IxG (gradientes)', 'Attention Rollout, LRP-Chefer', 'LIME, SHAP (perturbación)', 'solo transformers'], 'proceso'),
        (9.5, 'Por oración', ['MexGen (propio)', 'LIME-oración, SHAP-oración', '(variantes nativas)', 'aplican a los 4 modelos'], 'proceso'),
        (15.6, 'Controles', ['palabra → oración por suma', '(mide cuánto se pierde)', 'MexGen por cláusulas (comas)', 'borrado al azar por granularidad'], 'nota')]:
        gc.flecha(ax, (9.3, 12.0 - h / 2), (x, 8.6 + h2 / 2))
        gc.caja_lista(ax, x, 8.6, 5.6, h2, t, l, tipo)
    h3 = alto(6, E)
    for x in (3.4, 9.5, 15.6):
        gc.flecha(ax, (x, 8.6 - h2 / 2), (9.5 + (x - 9.5) * 0.4, 4.6 + h3 / 2))
    gc.caja_lista(ax, 9.5, 4.6, 16, h3, 'Métricas de fidelidad (eje común: fracción de palabras borradas, 10 % a 50 %)', [
        'AOPC: caída media de la salida al borrar lo más importante según el método',
        'ganancia sobre el azar = AOPC − AOPC de borrar al azar unidades de la MISMA granularidad',
        'suficiencia@20 % (conservar solo lo más importante) · cambio de clase@20 %',
        'Adebayo (15 noticias): reiniciar las 3 capas superiores y la cabeza; la explicación debería cambiar',
        'acuerdo con MexGen (Spearman) · MexGen − método: Wilcoxon pareado con corrección de Holm',
        'control de la puntuación: borrado por palabra conservando el punto final'], 'decision')
    y_r = 4.6 - h3 / 2 - 1.1
    gc.flecha(ax, (9.5, 4.6 - h3 / 2), (9.5, y_r + alto(1, E) / 2))
    gc.caja_lista(ax, 9.5, y_r, 16, alto(1, E), 'Resultado', [
        'por oración MexGen ≈ LIME y SHAP en fidelidad, con menor costo y sin azar; Rollout y LRP-Chefer fallan Adebayo'], 'salida')
    return fig


def auditoria_texto_nuevo():
    """Fuentes: Explicabilidad_Propia/auditar_texto_personalizado.py y codigos/auditor_estilo.py."""
    E = 0.6
    fig, ax = gc.lienzo_completo(23, 14.6, E)
    c = 11.5
    gc.titulo_diagrama(ax, 0.3, 14.4, 'Qué pasa al auditar un texto nuevo', 'auditar_texto_personalizado.py → auditor_estilo.AuditorEstilo.auditar')
    gc.caja_lista(ax, c, 11.5, 9, alto(2, E), 'Texto de la persona usuaria', [
        'línea de comandos (auditar_texto_personalizado.py)', 'o el dashboard (Fase 7: pendiente de conectar)'], 'entrada')
    h = alto(2, E)
    gc.flecha(ax, (c, 11.5 - alto(2, E) / 2), (c, 9.2 + h / 2))
    gc.caja_lista(ax, c, 9.2, 11, h, 'ExtractorVariables.extraer', [
        '43 variables + detalle por oración (sensacionalismo, emociones, matriz de similitud, marcadores)'], 'modelo')
    h2 = alto(5, E)
    xs = (2.95, 8.65, 14.35, 20.05)
    for x in xs:
        gc.flecha(ax, (c + (x - c) * 0.45, 9.2 - h / 2), (x, 5.6 + h2 / 2))
    gc.caja_lista(ax, xs[0], 5.6, 5.4, h2, '1 · Perfil de estilo', [
        'D1–D5: percentiles frente al', 'corpus (referencia_estilos)', 'IML', 'estilo dominante y secundario', 'con su pertenencia (%)'], 'proceso')
    gc.caja_lista(ax, xs[1], 5.6, 5.4, h2, '2 · Dónde está cada rasgo', [
        'oración más sensacionalista', 'oración más emotiva y su emoción', 'par de oraciones más parecido',
        '(«repite la misma afirmación»', 'si supera τ = 0,7232)'], 'proceso')
    gc.caja_lista(ax, xs[2], 5.6, 5.4, h2, '3 · Modelo sobre Variables', [
        'P(falsa) y predicción frente al', 'umbral (mediana de los anidados)', 'variables que más empujan:',
        'coeficiente × z en log-odds', '(explica el clasificador)'], 'modelo')
    gc.caja_lista(ax, xs[3], 5.6, 5.4, h2, '4 · MexGen por oraciones', [
        'quita una oración por vez (Punkt)', 'y vuelve a extraer las 43 variables', 'ΔIML y ΔP(falsa) por oración',
        'k + 1 extracciones', 'para k oraciones'], 'salida')
    for x in xs:
        gc.flecha(ax, (x, 5.6 - h2 / 2), (c + (x - c) * 0.45, 2.3 + 0.5))
    gc.caja_lista(ax, c, 2.3, 12, alto(1, E), 'Informe Markdown (informe_markdown) y JSON opcional', [
        'el auditor no juzga la veracidad: describe el estilo y lee el clasificador por separado'], 'salida')
    return fig


DIAGRAMAS = {
    'marco_propuesto': marco_propuesto, 'pipeline_general': pipeline_general, 'construccion_corpus': construccion_corpus,
    'evaluacion_sensacionalismo': evaluacion_sensacionalismo, 'evaluacion_redundancia': evaluacion_redundancia,
    'evaluacion_emociones': evaluacion_emociones, 'tabla_variables': tabla_variables, 'evaluacion_sin_fuga': evaluacion_sin_fuga,
    'iml_y_estilos': iml_y_estilos, 'mexgen': mexgen, 'comparacion_xai': comparacion_xai, 'auditoria_texto_nuevo': auditoria_texto_nuevo,
}


def recortar(fig, margen=0.35):
    """Ajusta el alto del lienzo al contenido (cajas, líneas y textos) para no dejar franjas vacías."""
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.text import Annotation
    ax = fig.axes[0]
    ancho = ax.get_xlim()[1]
    pt = 72 * fig.get_size_inches()[0] / ancho          # puntos por unidad del lienzo
    titulo, sub = ax.texts[0], (ax.texts[1] if isinstance(ax.texts[1], Annotation) else None)
    resto = [t for t in ax.texts if t is not titulo and t is not sub]
    ys = [y for p in ax.patches if isinstance(p, FancyBboxPatch) for y in (p.get_y(), p.get_y() + p.get_height())]
    ys += [y for l in ax.lines for y in l.get_ydata()]
    ys += [t.xy[1] if isinstance(t, Annotation) else t.get_position()[1] for t in resto]
    # El título (y su subtítulo) se coloca justo encima del contenido
    y_tit = max(ys) + (52 if sub else 30) / pt
    titulo.set_y(y_tit)
    if sub:
        sub.xy = (sub.xy[0], y_tit)
    y0, y1 = min(ys) - margen, y_tit + 0.15
    ax.set_ylim(y0, y1)
    w_in = fig.get_size_inches()[0]
    fig.set_size_inches(w_in, w_in * (y1 - y0) / ancho)
    return fig


def main():
    nombres = [a for a in sys.argv[1:]] or list(DIAGRAMAS)
    for n in nombres:
        ruta = gc.guardar(recortar(DIAGRAMAS[n]()), OUT / f'{n}.png')
        print(f"[OK] {ruta}")


if __name__ == '__main__':
    main()
