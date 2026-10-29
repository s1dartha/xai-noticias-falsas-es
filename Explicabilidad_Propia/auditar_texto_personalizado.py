"""
Auditoría de estilo discursivo de una noticia cualquiera (línea de comandos).

Entrega el IML, el estilo discursivo con sus grados de pertenencia, las oraciones donde se concentra cada rasgo y la
lectura del Modelo sobre Variables (codigos/auditor_estilo.py). No es un veredicto de veracidad.

Uso:
  python auditar_texto_personalizado.py --texto "Texto de la noticia..."
  python auditar_texto_personalizado.py --archivo noticia.txt --salida_json resultado.json --salida_md informe.md
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'codigos'))
from auditor_estilo import AuditorEstilo, informe_markdown


def main():
    ap = argparse.ArgumentParser(description="Auditoría de estilo discursivo de una noticia")
    ap.add_argument("--texto", help="Texto de la noticia")
    ap.add_argument("--archivo", help="Archivo .txt con la noticia")
    ap.add_argument("--titulo", default="Noticia personalizada")
    ap.add_argument("--salida_json")
    ap.add_argument("--salida_md")
    a = ap.parse_args()
    texto = Path(a.archivo).read_text(encoding='utf-8').strip() if a.archivo else (a.texto or '').strip()
    if not texto:
        sys.exit("No se proporcionó texto (use --texto o --archivo).")
    r = AuditorEstilo().auditar(texto, a.titulo)
    md = "\n".join([f"# {a.titulo}", ""] + informe_markdown(r))
    print(md)
    if a.salida_json:
        Path(a.salida_json).write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding='utf-8')
    if a.salida_md:
        Path(a.salida_md).write_text(md + "\n", encoding='utf-8')


if __name__ == '__main__':
    main()
