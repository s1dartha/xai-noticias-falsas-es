"""
Figuras propias de los informes complementarios (Informes_complementarios/).

No recalcula ningún modelo: lee resultados ya guardados en el repositorio.
    1. bootstrap_diferencias_oraciones.png  ← Analisis_complementarios/revision_tutor/tarea3_bootstrap_pareado.csv
    2. estilos_por_region.png               ← Explicabilidad_Propia/reportes/iml_estilos_discursivos_4418.csv
    3. correlacion_dimensiones.png          ← Explicabilidad_Propia/reportes/metricas_iml_estilos_discursivos.json

Uso (desde la raíz del repositorio):
    python Informes_complementarios/figuras_informes.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import graficos_comunes as gc  # noqa: E402

plt = gc.pyplot()
SALIDA = Path(__file__).resolve().parent / 'figuras'


def bootstrap_diferencias():
    """Diferencia pareada (expl. por oraciones − LIME/SHAP por oración) con su IC 95 % por bootstrap."""
    df = pd.read_csv(RAIZ / 'Analisis_complementarios/revision_tutor/tarea3_bootstrap_pareado.csv')
    modelos = list(dict.fromkeys(df['modelo']))
    # Mismos colores que LIME-oración y SHAP-oración en las figuras de la Sección 4.3
    comparaciones = {'LIME-oración': gc.SERIE[4], 'SHAP-oración': gc.SERIE[6]}
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for j, (metodo, color) in enumerate(comparaciones.items()):
        sub = df[df['comparacion'].str.endswith(metodo)].set_index('modelo').loc[modelos]
        y = np.arange(len(modelos)) + (0.15 if j == 0 else -0.15)
        ax.hlines(y, sub['ic95_inf'], sub['ic95_sup'], color=color, linewidth=2)
        ax.plot(sub['diferencia_media'], y, 'o', color=color, markersize=8,
                markeredgecolor=gc.SUPERFICIE, markeredgewidth=2, label=f'frente a {metodo}')
    ax.axvline(0, color=gc.TINTA_2, linewidth=1, linestyle='--')
    ax.set_yticks(np.arange(len(modelos)), modelos)
    ax.invert_yaxis()
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('Diferencia de ganancia (AOPC sobre el azar de oraciones)')
    gc.coma(ax, 'x', 3)
    ax.set_title('Expl. por oraciones frente a LIME y SHAP por oración: IC 95 % (bootstrap pareado, 10.000 réplicas)')
    ax.legend(loc='lower right', fontsize=8.5)
    return gc.guardar(fig, SALIDA / 'bootstrap_diferencias_oraciones.png')


def estilos_por_region():
    """Composición de estilos por región y proporción de falsas en cada estilo y región (textos únicos)."""
    df = pd.read_csv(RAIZ / 'Explicabilidad_Propia/reportes/iml_estilos_discursivos_4418.csv').drop_duplicates('id_texto')
    estilos = sorted(df['estilo'].unique())
    regiones = ['España', 'América Latina']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.4))
    ancho = 0.26
    for i, e in enumerate(estilos):
        color = gc.ESTILOS[i + 1]
        comp = [100 * (df.loc[df['region'] == r, 'estilo'] == e).mean() for r in regiones]
        tasa = [100 * df.loc[(df['region'] == r) & (df['estilo'] == e), 'label_num'].mean() for r in regiones]
        x = np.arange(len(regiones)) + (i - 1) * (ancho + 0.02)
        etiqueta = e.split(': ', 1)[-1]
        a1.bar(x, comp, ancho, color=color, label=etiqueta)
        a2.bar(x, tasa, ancho, color=color, label=etiqueta)
        for ax, vals in ((a1, comp), (a2, tasa)):
            for xi, v in zip(x, vals):
                ax.text(xi, v + 1, f'{v:.0f}', ha='center', va='bottom', fontsize=8, color=gc.TINTA_2, zorder=4,
                        bbox=dict(boxstyle='square,pad=0.1', facecolor=gc.SUPERFICIE, edgecolor='none'))
    for r in regiones:
        global_r = 100 * df.loc[df['region'] == r, 'label_num'].mean()
        k = regiones.index(r)
        a2.hlines(global_r, k - 0.45, k + 0.45, color=gc.TINTA_2, linewidth=1, linestyle='--', zorder=3)
        print(f'  falsas en {r}: {global_r:.1f} %')
    for ax, t in ((a1, '% de las noticias de cada región en cada estilo'),
                  (a2, '% de noticias falsas en cada estilo (línea: % de la región)')):
        ax.set_xticks(range(len(regiones)), regiones)
        ax.grid(axis='x', visible=False)
        ax.set_ylim(0, 75)
        ax.set_title(t, fontsize=10)
    a1.legend(loc='upper left', fontsize=8)
    return gc.guardar(fig, SALIDA / 'estilos_por_region.png')


def correlacion_dimensiones():
    """Correlación de Spearman entre D1–D5 (textos únicos)."""
    m = json.loads((RAIZ / 'Explicabilidad_Propia/reportes/metricas_iml_estilos_discursivos.json').read_text(encoding='utf-8'))
    c = pd.DataFrame(m['correlacion_dimensiones_spearman'])
    dims = list(c.columns)
    nombres = [f"{d}\n{m['dimensiones'][d]['nombre']}" for d in dims]
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(c.loc[dims, dims].values, cmap=gc.divergente(), vmin=-1, vmax=1)
    for i in range(len(dims)):
        for j in range(len(dims)):
            v = c.loc[dims[i], dims[j]]
            ax.text(j, i, f'{v:.2f}'.replace('.', ','), ha='center', va='center', fontsize=8.5, color=gc.TINTA)
    ax.set_xticks(range(len(dims)), dims, fontsize=8)
    ax.set_yticks(range(len(dims)), nombres, fontsize=7.5)
    ax.grid(False)
    ax.set_title('Correlación de Spearman entre las cinco dimensiones del IML')
    fig.colorbar(im, ax=ax, shrink=0.8)
    return gc.guardar(fig, SALIDA / 'correlacion_dimensiones.png')


if __name__ == '__main__':
    for f in (bootstrap_diferencias, estilos_por_region, correlacion_dimensiones):
        print('[OK]', f().relative_to(RAIZ))
