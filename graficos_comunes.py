"""
Estilo común de figuras y diagramas del repositorio.

Uso en un generador de reportes:
    import graficos_comunes as gc
    plt = gc.pyplot()                       # matplotlib con backend Agg y el estilo aplicado
    fig, ax = plt.subplots(...)
    ax.bar(x, y, color=gc.SERIE[0])
    gc.guardar(fig, ruta_png)

Colores:
    SERIE        8 colores categóricos en orden fijo (se asignan por entidad, nunca se reciclan)
    CLASE        REAL / FAKE (mismo color en todas las figuras)
    ESTILOS      color de cada estilo discursivo (1, 2, 3)
    TINTA, TINTA_2, TINTA_SUAVE, REJILLA, EJE, SUPERFICIE
    secuencial(n), divergente()  mapas de color de una y dos tonalidades

Diagramas (cajas y flechas en coordenadas de datos de un eje sin marcos):
    fig, ax = gc.lienzo(ancho, alto)        # eje de 0..ancho × 0..alto en unidades arbitrarias
    gc.caja(ax, x, y, w, h, titulo, detalle, tipo='proceso')
    gc.flecha(ax, (x0, y0), (x1, y1), texto=None)
Tipos de caja: 'entrada', 'proceso', 'modelo', 'salida', 'decision', 'nota'.
"""

from pathlib import Path

# --- Paleta (instancia de referencia validada para daltonismo; ver skill dataviz) ---
SERIE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
CLASE = {'REAL': '#2a78d6', 'FAKE': '#e34948', 0: '#2a78d6', 1: '#e34948'}
ESTILOS = {1: '#2a78d6', 2: '#1baf7a', 3: '#eb6834'}
TINTA, TINTA_2, TINTA_SUAVE = '#0b0b0b', '#52514e', '#898781'
REJILLA, EJE, SUPERFICIE = '#e1e0d9', '#c3c2b7', '#ffffff'
AZUL_SEC = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']

# Relleno y borde de las cajas de diagrama por tipo
TIPOS_CAJA = {
    'entrada':  ('#f0efec', '#52514e'),
    'proceso':  ('#e6f0fc', '#2a78d6'),
    'modelo':   ('#fdeee6', '#eb6834'),
    'salida':   ('#e3f6ee', '#1baf7a'),
    'decision': ('#fdf3d9', '#c98500'),
    'nota':     ('#ffffff', '#c3c2b7'),
}


def aplicar_estilo():
    import matplotlib as mpl
    mpl.rcParams.update({
        'figure.facecolor': SUPERFICIE, 'axes.facecolor': SUPERFICIE, 'savefig.facecolor': SUPERFICIE,
        'font.family': 'sans-serif', 'font.sans-serif': ['Segoe UI', 'DejaVu Sans', 'Arial'], 'font.size': 10,
        'axes.edgecolor': EJE, 'axes.labelcolor': TINTA_2, 'axes.titlecolor': TINTA, 'axes.titlesize': 11,
        'axes.titleweight': 'regular', 'axes.titlelocation': 'left',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': REJILLA, 'grid.linewidth': 0.6,
        'xtick.color': TINTA_SUAVE, 'ytick.color': TINTA_SUAVE, 'xtick.labelcolor': TINTA_2, 'ytick.labelcolor': TINTA_2,
        'legend.frameon': False, 'lines.linewidth': 2, 'axes.prop_cycle': mpl.cycler(color=SERIE),
        'savefig.dpi': 200, 'savefig.bbox': 'tight',
        'axes.formatter.use_locale': False,
    })


def pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    aplicar_estilo()
    return plt


def guardar(fig, ruta):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return ruta


def coma(ax, eje='y', decimales=None):
    """Separador decimal con coma en los ejes."""
    from matplotlib.ticker import FuncFormatter
    fmt = FuncFormatter(lambda v, _: (f"{v:.{decimales}f}" if decimales is not None else f"{v:g}").replace('.', ','))
    (ax.yaxis if eje == 'y' else ax.xaxis).set_major_formatter(fmt)


def secuencial(n=256):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list('azul', ['#f4f8fe'] + AZUL_SEC, N=n)


def divergente(n=256):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list('div', ['#2a78d6', '#f0efec', '#e34948'], N=n)


# =============================================================================
# DIAGRAMAS
# =============================================================================
def lienzo(ancho, alto, escala=0.9):
    """Figura sin ejes con un sistema de coordenadas 0..ancho × 0..alto (y crece hacia arriba)."""
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(ancho * escala, alto * escala))
    ax.set_xlim(0, ancho)
    ax.set_ylim(0, alto)
    ax.axis('off')
    ax.set_aspect('equal')
    return fig, ax


def caja(ax, x, y, w, h, titulo, detalle=None, tipo='proceso', tam=10, tam_detalle=8.5):
    """Caja redondeada centrada en (x, y). `detalle` admite saltos de línea."""
    from matplotlib.patches import FancyBboxPatch
    relleno, borde = TIPOS_CAJA[tipo]
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.02,rounding_size=0.12',
                                fc=relleno, ec=borde, lw=1.4, zorder=2))
    if detalle:
        ax.text(x, y + h * 0.18, titulo, ha='center', va='center', fontsize=tam, color=TINTA, weight='bold', zorder=3, wrap=True)
        ax.text(x, y - h * 0.16, detalle, ha='center', va='center', fontsize=tam_detalle, color=TINTA_2, zorder=3, linespacing=1.25)
    else:
        ax.text(x, y, titulo, ha='center', va='center', fontsize=tam, color=TINTA, weight='bold', zorder=3, linespacing=1.25)


def flecha(ax, origen, destino, texto=None, color=TINTA_2, estilo='-|>', curva=0.0, tam=8.5, lado=0.12):
    """Flecha entre dos puntos; `texto` se escribe junto al punto medio."""
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(origen, destino, arrowstyle=estilo, mutation_scale=14, color=color, lw=1.3,
                                 connectionstyle=f'arc3,rad={curva}', zorder=1, shrinkA=2, shrinkB=2))
    if texto:
        mx, my = (origen[0] + destino[0]) / 2, (origen[1] + destino[1]) / 2
        ax.text(mx + lado, my, texto, ha='left', va='center', fontsize=tam, color=TINTA_2, style='italic', zorder=3)


def leyenda_tipos(ax, x, y, tipos=('entrada', 'proceso', 'modelo', 'salida'), paso=2.2):
    nombres = {'entrada': 'Datos / entrada', 'proceso': 'Procesamiento', 'modelo': 'Modelo preentrenado o ajustado',
               'salida': 'Resultado', 'decision': 'Decisión / criterio', 'nota': 'Nota'}
    from matplotlib.patches import FancyBboxPatch
    for i, t in enumerate(tipos):
        relleno, borde = TIPOS_CAJA[t]
        ax.add_patch(FancyBboxPatch((x + i * paso * 1.6, y - 0.15), 0.4, 0.3, boxstyle='round,pad=0.02', fc=relleno, ec=borde, lw=1.2))
        ax.text(x + i * paso * 1.6 + 0.55, y, nombres[t], va='center', fontsize=8, color=TINTA_2)


# --- Primitivas adicionales para diagramas con texto en lista (Diagramas/generar_diagramas.py) ---
def caja_lista(ax, x, y, w, h, titulo, lineas=(), tipo='proceso', tam=10, tam_detalle=8.5, discontinua=False, alinear='center'):
    """Caja centrada en (x, y) con el título arriba y las líneas debajo (posiciones en puntos desde el borde superior,
    de modo que el texto no depende de la escala del lienzo). `alinear='left'` alinea las líneas a la izquierda."""
    from matplotlib.patches import FancyBboxPatch
    relleno, borde = TIPOS_CAJA[tipo]
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.02,rounding_size=0.12',
                                fc=relleno, ec=borde, lw=1.4, ls='--' if discontinua else '-', zorder=2))
    if not lineas:
        ax.text(x, y, titulo, ha='center', va='center', fontsize=tam, color=TINTA, weight='bold', zorder=3, linespacing=1.25)
        return
    ax.annotate(titulo, (x, y + h / 2), xytext=(0, -7), textcoords='offset points', ha='center', va='top',
                fontsize=tam, color=TINTA, weight='bold', zorder=3, linespacing=1.2)
    n_tit = titulo.count('\n') + 1
    xl = x if alinear == 'center' else x - w / 2 + 0.15
    ax.annotate('\n'.join(lineas), (xl, y + h / 2), xytext=(0, -7 - n_tit * tam * 1.3 - 4), textcoords='offset points',
                ha=alinear, va='top', fontsize=tam_detalle, color=TINTA_2, zorder=3, linespacing=1.35)


def grupo(ax, x0, y0, x1, y1, etiqueta, subtitulo=None, color=EJE, tam=11, lado='izq'):
    """Rectángulo discontinuo que agrupa varias cajas, con su etiqueta en la esquina superior izquierda
    (o derecha con lado='der')."""
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle='round,pad=0.02,rounding_size=0.2',
                                fc='none', ec=color, lw=1.6, ls=(0, (5, 3)), zorder=0))
    xa, dx, ha = (x0, 8, 'left') if lado == 'izq' else (x1, -8, 'right')
    ax.annotate(etiqueta, (xa, y1), xytext=(dx, -6), textcoords='offset points', ha=ha, va='top',
                fontsize=tam, color=color, weight='bold')
    if subtitulo:
        ax.annotate(subtitulo, (xa, y1), xytext=(dx, -6 - tam * 1.45), textcoords='offset points', ha=ha, va='top',
                    fontsize=tam - 2, color=TINTA_2, style='italic')


def lienzo_completo(ancho, alto, escala=0.9):
    """Como lienzo(), pero el eje ocupa toda la figura: 1 unidad = `escala` pulgadas exactas, de modo que el alto de
    las cajas puede calcularse a partir del tamaño de letra."""
    fig, ax = lienzo(ancho, alto, escala)
    fig.subplots_adjust(0, 0, 1, 1)
    return fig, ax


def flecha_ruta(ax, puntos, texto=None, color=TINTA_2, tam=8.5, pos_texto=None):
    """Flecha poligonal (codos rectos) que pasa por `puntos`; la punta está en el último punto."""
    from matplotlib.patches import FancyArrowPatch
    xs, ys = zip(*puntos)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=1.3, zorder=1, solid_capstyle='round')
    ax.add_patch(FancyArrowPatch(puntos[-2], puntos[-1], arrowstyle='-|>', mutation_scale=14, color=color, lw=1.3,
                                 zorder=1, shrinkA=0, shrinkB=2))
    if texto:
        px, py = pos_texto or ((puntos[0][0] + puntos[1][0]) / 2, (puntos[0][1] + puntos[1][1]) / 2)
        ax.text(px, py, texto, ha='center', va='bottom', fontsize=tam, color=TINTA_2, style='italic', zorder=3,
                bbox=dict(fc=SUPERFICIE, ec='none', pad=1))


def titulo_diagrama(ax, x, y, titulo, subtitulo=None, tam=14):
    ax.text(x, y, titulo, ha='left', va='top', fontsize=tam, color=TINTA, weight='bold')
    if subtitulo:
        ax.annotate(subtitulo, (x, y), xytext=(0, -tam * 1.6), textcoords='offset points', ha='left', va='top',
                    fontsize=tam - 4, color=TINTA_2)


def diagrama_md(nombre, carpeta_reporte, pie):
    """Markdown que embebe Diagramas/<nombre> desde un reporte ubicado en `carpeta_reporte`."""
    import os
    import config_proyecto as cfg
    ruta = os.path.relpath(cfg.DIAGRAMAS_DIR / nombre, carpeta_reporte).replace('\\', '/')
    return f"![{pie}]({ruta})\n\n*{pie}. Fuente: `Diagramas/generar_diagramas.py`.*\n"
