"""
Corrección P1-27: regenera las figuras del OE3 cambiando solo la etiqueta de los métodos propios.
  «MexGen»            -> «Expl. por oraciones»
  «MexGen-cláusulas»  -> «Expl. por segmentos»   (y la leyenda «Cláusulas (MexGen anterior)»)

No carga modelos ni recalcula explicaciones: usa las funciones figuras() y figuras_extra() de
Explicabilidad_Propia/XAI_Literatura_Comparacion/codigos/generar_reporte_comparacion_xai.py sobre
reportes/datos/metricas_comparacion_xai.json y la caché codigos/cache/jjneila/*.json (solo lectura).
El código fuente original no se modifica: se lee, se sustituyen las cadenas de texto visibles en memoria y se ejecuta
con la carpeta de salida redirigida a Analisis_complementarios/figuras_oe3/. No se escribe el reporte .md.
"""
import sys
import json
import types
import importlib.util
from pathlib import Path

RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "config_proyecto.py").exists())
sys.path.insert(0, str(RAIZ))
import config_proyecto as cfg  # noqa: E402

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

ORIGEN = cfg.XAI_COMPARACION_DIR / 'codigos' / 'generar_reporte_comparacion_xai.py'
SALIDA = Path(__file__).resolve().parent / 'figuras_oe3'
N_OR, N_SEG = 'Expl. por oraciones', 'Expl. por segmentos'
RENOMBRE = {'MexGen': N_OR, 'MexGen-cláusulas': N_SEG}
# Sustituciones en el código (en este orden: primero las más largas)
SUSTITUCIONES = [("'MexGen-cláusulas'", f"'{N_SEG}'"), ("'Cláusulas (MexGen anterior)'", f"'{N_SEG}'"),
                 ("'MexGen'", f"'{N_OR}'"), ("Acuerdo con MexGen en el orden", f"Acuerdo con {N_OR} en el orden")]


def ren(d):
    return {RENOMBRE.get(k, k): v for k, v in d.items()}


def main():
    # Módulo original (sin tocar) para elegir_ejemplo, que lee la caché con las claves originales
    spec = importlib.util.spec_from_file_location('rep_original', ORIGEN)
    orig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orig)

    src = ORIGEN.read_text(encoding='utf-8')
    for a, b in SUSTITUCIONES:
        assert a in src, a
        src = src.replace(a, b)
    mod = types.ModuleType('rep_renombrado')
    mod.__file__ = str(ORIGEN)
    exec(compile(src, str(ORIGEN), 'exec'), mod.__dict__)
    mod.IMG = SALIDA

    def elegir_ejemplo(objetivo='jjneila'):
        d = orig.elegir_ejemplo(objetivo)
        if d is not None:
            d['_atribuciones'] = ren(d['_atribuciones'])
        return d
    mod.elegir_ejemplo = elegir_ejemplo

    j = json.load(open(cfg.XAI_COMPARACION_DATOS_DIR / 'metricas_comparacion_xai.json', encoding='utf-8'))
    for o, docs in j['documentos'].items():
        for d in docs:
            d['metodos'] = ren(d['metodos'])
    j['adebayo'] = {o: [ren(a) for a in v] for o, v in j['adebayo'].items()}
    R = {o: mod.resumir(j['documentos'][o], j['adebayo'].get(o, [])) for o in mod.OBJ if j['documentos'].get(o)}
    SALIDA.mkdir(parents=True, exist_ok=True)
    figs = mod.figuras(R, j['metadata'])
    extra, _ = mod.figuras_extra(R, j)
    figs.update(extra)
    for k, v in figs.items():
        print(k, SALIDA / Path(v).name)


if __name__ == '__main__':
    main()
