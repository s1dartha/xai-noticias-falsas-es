"""
Genera Dataset/Reporte_Corpus_Ampliado.md a partir del corpus ampliado (4.418 noticias). Todas las cifras se calculan
del CSV; la procedencia de cada fuente resume lo que hace construir_corpus_latam.py y lo que declaran las fichas
públicas de cada dataset.
"""

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config_proyecto.py").exists())))
import config_proyecto as cfg
import graficos_comunes as gc

PROCEDENCIA = {
    'Freiren/Unified-and-Balanced-Spanish-Fake-News-Corpus': (
        'Columna `class` del corpus base (True = real).',
        'Según su ficha, unifica cuatro corpus: Posadas-Durán et al. (572), Acosta et al. (598), Tretiakov et al. (2.000) y Blanco-Fernández et al. '
        '(57.231, el 93 %). Este último se construyó con extracción web y se amplió con modelos generativos (Blanco-Fernández et al., Applied Sciences, 2024): '
        'parte de los textos puede ser sintética. No se conoce de qué corpus de origen proviene cada noticia.'),
    'Edds/spanish-fake-news-fixed': ('Columna `class` del corpus base (True = real).', 'Versión corregida de un corpus de noticias en español publicada en Hugging Face.'),
    'IsaacRodgz/Fake-news-latam-omdena': (
        'Columna `Corrected_label` (desde el 2026-10-24; antes se usaba por error `Prediction`, la salida de un clasificador de Omdena).',
        'Noticias políticas latinoamericanas recopiladas por el proyecto Omdena; incluye `Prediction_prob` y la etiqueta corregida.'),
    'FakeNewsCorpusSpanish': ('Columna `Category` (True / Fake) de `train.xlsx` y `development.xlsx`.',
                              'Posadas-Durán et al. (2019), Journal of Intelligent & Fuzzy Systems 36(5). Las cifras están anonimizadas como `*NUMBER*`.'),
    'mariagrandury/fake_news_corpus_spanish': ('Columna `CATEGORY`.', 'Corpus de FakeDeS (IberLEF 2021; Gómez-Adorno et al., 2021).'),
}


def corto(f):
    for k in PROCEDENCIA:
        if k in f:
            return k
    return f


def pct(a, b):
    return f"{100 * a / b:.1f}".replace('.', ',') + " %"


IMG = Path(__file__).resolve().parent / 'Imagenes'
ABREV = {'Freiren/Unified-and-Balanced-Spanish-Fake-News-Corpus': 'Unified', 'Edds/spanish-fake-news-fixed': 'Edds',
         'IsaacRodgz/Fake-news-latam-omdena': 'Omdena', 'FakeNewsCorpusSpanish': 'MEX-A3T', 'mariagrandury/fake_news_corpus_spanish': 'FakeDeS'}


def figuras(d):
    """Composición, longitud y duplicados del corpus. Devuelve rutas relativas al reporte."""
    import numpy as np
    sys.path.insert(0, str(cfg.ROOT))
    import graficos_comunes as gc
    plt = gc.pyplot()
    out = {}
    d = d.assign(etq=d['fuente'].map(lambda f: ABREV.get(f, f)))

    # 1. Composición por fuente y región, apilada por clase
    g = d.groupby(['region', 'etq', 'y']).size().unstack(fill_value=0)
    g = g.loc[g.sum(1).sort_values().index]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    etiquetas = [f"{f} ({r})" for r, f in g.index]
    pos = np.arange(len(g))
    ax.barh(pos, g.get(0, 0), color=gc.CLASE['REAL'], height=0.62, label='Real (0)', edgecolor='white', linewidth=1)
    ax.barh(pos, g.get(1, 0), left=g.get(0, 0), color=gc.CLASE['FAKE'], height=0.62, label='Falsa (1)', edgecolor='white', linewidth=1)
    for i, (r0, r1) in enumerate(zip(g.get(0, 0), g.get(1, 0))):
        ax.text(r0 + r1 + 15, i, f"{r0 + r1}  ({pct(r1, r0 + r1)} falsas)", va='center', fontsize=8.5, color=gc.TINTA_2)
    ax.set_yticks(pos, etiquetas)
    ax.set_xlabel('Noticias')
    ax.set_xlim(0, g.sum(1).max() * 1.35)
    ax.grid(axis='y', visible=False)
    ax.set_title('Composición del corpus ampliado por fuente, región y clase')
    ax.legend(loc='lower right')
    out['composicion'] = gc.guardar(fig, IMG / 'composicion_fuente_region_clase.png')

    # 2. Longitud (palabras) por región y clase
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    bins = np.linspace(d['conteo_palabras_text'].min(), d['conteo_palabras_text'].max(), 31)
    for ax, (r, gr) in zip(axs, d.groupby('region')):
        for y, nombre in [(0, 'Real'), (1, 'Falsa')]:
            w = gr.loc[gr.y == y, 'conteo_palabras_text']
            ax.hist(w, bins=bins, color=gc.CLASE[y], histtype='step', linewidth=2, label=f"{nombre} (mediana {w.median():.0f})")
        ax.set_title(f"Longitud de los textos: {r}")
        ax.set_xlabel('Palabras por noticia')
        ax.legend(fontsize=8.5)
    axs[0].set_ylabel('Noticias')
    out['longitud'] = gc.guardar(fig, IMG / 'longitud_por_region_clase.png')

    # 3. Duplicados entre fuentes (número de textos compartidos por cada par de fuentes)
    dup = d[d.duplicated('norm', keep=False)]
    fuentes = sorted(d['etq'].unique())
    M = pd.DataFrame(0, index=fuentes, columns=fuentes)
    for _, gg in dup.groupby('norm'):
        fs = sorted(gg['etq'].unique())
        if len(fs) == 1:
            M.loc[fs[0], fs[0]] += 1
        for i in range(len(fs)):
            for j in range(i + 1, len(fs)):
                M.loc[fs[i], fs[j]] += 1
                M.loc[fs[j], fs[i]] += 1
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.imshow(M.values, cmap=gc.secuencial())
    ax.grid(False)
    ax.set_xticks(range(len(fuentes)), fuentes, rotation=30, ha='right')
    ax.set_yticks(range(len(fuentes)), fuentes)
    for i in range(len(fuentes)):
        for j in range(len(fuentes)):
            v = M.values[i, j]
            ax.text(j, i, str(v), ha='center', va='center', fontsize=9, color='white' if v > M.values.max() * 0.55 else gc.TINTA)
    ax.set_title('Textos duplicados compartidos entre fuentes\n(diagonal: duplicados dentro de la misma fuente)', fontsize=10)
    out['duplicados'] = gc.guardar(fig, IMG / 'duplicados_entre_fuentes.png')
    return {k: 'Imagenes/' + v.name for k, v in out.items()}


def main():
    d = pd.read_csv(cfg.CORPUS_AMPLIADO_CSV)
    d['y'] = d['clase_num'].map(cfg.etiqueta_binaria)
    d['fuente'] = d['Fuente'].map(corto)
    d['norm'] = d['Text'].map(lambda t: ' '.join(str(t).lower().split()))
    d['mascara'] = d['Text'].str.contains(r'\*(?:NUMBER|URL|PHONE)\*', flags=re.I, regex=True)
    n = len(d)
    figs = figuras(d)
    L = ["# Corpus ampliado España – América Latina", "",
         "**Generado por:** `generar_reporte_corpus.py` a partir de `Noticias_entre_70_y_370_palabras_AMPLIADO_LATAM.csv` "
         "(construido por `construir_corpus_latam.py`)  ", "**Etiqueta:** `clase_num` 1 = falsa, 0 = real (ver `ETIQUETAS.md`)", "",
         gc.diagrama_md('construccion_corpus.png', cfg.DATASET_FAKE_DIR, 'Construcción del corpus ampliado'),
         "## 1. Composición", "",
         "| Fuente | Región | Noticias | Reales | Falsas | % falsas | Con máscaras `*NUMBER*` |", "|:---|:---|---:|---:|---:|---:|---:|"]
    for (f, r), g in d.groupby(['fuente', 'region']):
        L.append(f"| `{f}` | {r} | {len(g)} | {(g.y == 0).sum()} | {(g.y == 1).sum()} | {pct((g.y == 1).sum(), len(g))} | {pct(g.mascara.sum(), len(g))} |")
    for r, g in d.groupby('region'):
        L.append(f"| **{r}** | | **{len(g)}** | {(g.y == 0).sum()} | {(g.y == 1).sum()} | {pct((g.y == 1).sum(), len(g))} | {pct(g.mascara.sum(), len(g))} |")
    L.append(f"| **Total** | | **{n}** | {(d.y == 0).sum()} | {(d.y == 1).sum()} | {pct((d.y == 1).sum(), n)} | {pct(d.mascara.sum(), n)} |")
    L += ["", f"![Composición por fuente, región y clase]({figs['composicion']})", "",
          "Cada barra es una fuente en su región, dividida en noticias reales (azul) y falsas (rojo); a la derecha, el total y la proporción de falsas.", ""]
    dup = d[d.duplicated('norm', keep=False)]
    entre = dup.groupby('norm')['fuente'].nunique()
    L += ["", "## 2. Duplicados", "",
          f"- Textos únicos: **{d['norm'].nunique()}** de {n} noticias.",
          f"- Textos que aparecen más de una vez: {dup['norm'].nunique()} ({len(dup)} filas); en {int((entre > 1).sum())} de ellos las copias están en fuentes distintas.",
          f"- Duplicados con etiquetas contradictorias: {int((dup.groupby('norm')['y'].nunique() > 1).sum())}.",
          "- Todas las evaluaciones del Modelo sobre Variables, del IML y de la explicabilidad trabajan con los textos únicos.", "",
          f"![Duplicados entre fuentes]({figs['duplicados']})", "",
          "Fuera de la diagonal, número de textos que aparecen a la vez en dos fuentes; en la diagonal, textos repetidos dentro de una misma fuente. "
          "Todos los duplicados están entre Unified y otra fuente (Edds o FakeDeS).", "",
          "## 3. Longitud", "",
          "| Región | Palabras (media) | Mediana | Mínimo | Máximo |", "|:---|---:|---:|---:|---:|"]
    for r, g in d.groupby('region'):
        w = g['conteo_palabras_text']
        L.append(f"| {r} | {w.mean():.0f} | {w.median():.0f} | {w.min()} | {w.max()} |")
    L += ["", f"![Longitud por región y clase]({figs['longitud']})", "",
          "Distribución del número de palabras de las noticias reales y falsas en cada región (el corpus se limitó a textos de 70 a 370 palabras).", "",
          "## 4. Procedencia de las etiquetas", "", "| Fuente | Columna de origen de la etiqueta | Notas |", "|:---|:---|:---|"]
    for k, (col, nota) in PROCEDENCIA.items():
        L.append(f"| `{k}` | {col} | {nota} |")
    L += ["", "## 5. Cautelas para la interpretación", "",
          "- **Atajos propios de cada fuente.** Las máscaras `*NUMBER*` aparecen casi solo en MEX-A3T y el formato varía entre corpus; un clasificador puede "
          "aprender la fuente en lugar de la veracidad. Por eso la evaluación incluye la exclusión de una fuente por vez.",
          "- **Corpus Unified (España).** Una parte puede ser texto generado o ampliado con modelos generativos; esto puede explicar por qué en España los "
          "modelos de transformers obtienen cifras mucho más altas que las variables de estilo.",
          "- **Omdena.** Sus etiquetas son las corregidas por el propio proyecto; los modelos evaluados quedan cerca del azar en esta fuente.", ""]
    salida = Path(__file__).resolve().parent / 'Reporte_Corpus_Ampliado.md'
    salida.write_text("\n".join(L) + "\n", encoding='utf-8')
    print(f"[OK] {salida}")


if __name__ == '__main__':
    main()
