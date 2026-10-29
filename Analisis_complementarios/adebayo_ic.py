"""
Corrección P1-29: IC 95 % por bootstrap de la correlación media de Spearman de la prueba de Adebayo, por método y modelo.
Lee las correlaciones por documento ya guardadas en metricas_comparacion_xai.json (clave 'adebayo'); no carga modelos
ni recalcula explicaciones. Bootstrap percentil sobre documentos, 2.000 réplicas, semilla 42.
"""
import sys
import json
from pathlib import Path

import numpy as np

RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "config_proyecto.py").exists())
sys.path.insert(0, str(RAIZ))
import config_proyecto as cfg  # noqa: E402

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

N_BOOT = 2000


def main():
    fuente = cfg.XAI_COMPARACION_DATOS_DIR / 'metricas_comparacion_xai.json'
    j = json.loads(fuente.read_text(encoding='utf-8'))
    out = {'metadata': {'fuente': str(fuente.relative_to(RAIZ)), 'n_boot': N_BOOT, 'semilla': cfg.SEED,
                        'metodo': 'media de Spearman por documento; IC percentil 2,5–97,5 remuestreando documentos; NaN excluidos'},
           'resultados': {}}
    for modelo, docs in j['adebayo'].items():
        if not docs:
            out['resultados'][modelo] = 'sin prueba de Adebayo (no se ejecutó para este objetivo)'
            continue
        out['resultados'][modelo] = {}
        for m in docs[0]:
            v = np.array([d.get(m, np.nan) for d in docs], dtype=float)
            v = v[~np.isnan(v)]
            rng = np.random.default_rng(cfg.SEED)
            boots = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(N_BOOT)]
            out['resultados'][modelo][m] = {'n_documentos': int(len(v)), 'media': float(v.mean()),
                                            'ic95': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
            r = out['resultados'][modelo][m]
            print(f"{modelo:<12} {m:<18} n={r['n_documentos']:>2} media={r['media']:+.3f} IC=[{r['ic95'][0]:+.3f}, {r['ic95'][1]:+.3f}]")
    (Path(__file__).resolve().parent / 'adebayo_ic.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
