"""
Tres cálculos complementarios solicitados en la revisión final del trabajo (ninguno reentrena modelos).

1. JJNeila sobre los 70 textos sintéticos de control (Sección 3.4.1) con el umbral calibrado (τ = 0,85).
2. Longitud del corpus ampliado (4.418 noticias): % de noticias con más de 256 subpalabras según el tokenizador
   de SaBERT y % con más de 1.000 caracteres.
3. Bootstrap pareado de la diferencia de ganancia sobre el azar entre la explicación por oraciones y
   LIME-oración / SHAP-oración (Tabla 4.18, 100 noticias por modelo).

Entradas (rutas del repositorio, vía config_proyecto):
  Modelos_Individuales/Sensacionalismo/Dataset/dataset_IA_sintetico_70.csv
  Modelos_Individuales/Sensacionalismo/Evaluacion_Modelos/cache/jjneila__ia_sintetico.npy
  Modelos_Individuales/Clasificar_fake/Dataset/Noticias_entre_70_y_370_palabras_AMPLIADO_LATAM.csv
  Explicabilidad_Propia/XAI_Literatura_Comparacion/reportes/datos/metricas_comparacion_xai.json
Salidas: Analisis_complementarios/revision_tutor/ (resultados_revision_tutor.json y tarea{1,2,3}_*.csv).

Uso (desde la raíz del repositorio):
  python Analisis_complementarios/calculos_revision_tutor.py          # las tres tareas
  python Analisis_complementarios/calculos_revision_tutor.py 3        # solo la tarea 3 (sin modelos, segundos)
Las tareas 1 y 2 descargan de Hugging Face, en las revisiones fijadas en config_proyecto, el modelo JJNeila y el
tokenizador de SaBERT (~0,5 GB la primera vez); corren en CPU en pocos minutos. Al ejecutar solo algunas tareas, se
actualizan únicamente sus claves en el JSON de resultados.
"""
import sys
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config_proyecto.py").exists())))
import config_proyecto as cfg  # noqa: E402

SALIDA = Path(__file__).resolve().parent / "revision_tutor"
RESULTADOS = SALIDA / "resultados_revision_tutor.json"
SEED = cfg.SEED

SINTETICOS = cfg.SENSACIONALISMO_DATASET_DIR / "dataset_IA_sintetico_70.csv"
CACHE_JJ = cfg.SENSACIONALISMO_DIR / "Evaluacion_Modelos" / "cache" / "jjneila__ia_sintetico.npy"
CORPUS = cfg.CORPUS_AMPLIADO_CSV
XAI = cfg.XAI_COMPARACION_DATOS_DIR / "metricas_comparacion_xai.json"

_JJ = cfg.MODELOS_HF["Sensacionalismo_JJNeila"]
JJ_ID, JJ_REV, JJ_IDX, JJ_MAXLEN, TAU = _JJ["id"], _JJ["revision"], _JJ["idx_positivo"], 128, _JJ["umbral_calibrado"]
_SB = cfg.MODELOS_HF["M2_SaBERT"]
SABERT_ID, SABERT_REV = _SB["id"], _SB["revision"]


def ic_bootstrap_f1(y, pred, n_boot=1000):
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            boots.append(f1_score(y[i], pred[i], average="macro"))
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def tarea1():
    """JJNeila (128 subpalabras, τ calibrado) sobre los 70 textos sintéticos; compara con la caché de la Fase 3."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ia = pd.read_csv(SINTETICOS)
    textos = ia["Texto"].astype(str).tolist()
    y = (ia["Clase"] == "Sensacionalista").astype(int).values
    tok = AutoTokenizer.from_pretrained(JJ_ID, revision=JJ_REV)
    model = AutoModelForSequenceClassification.from_pretrained(JJ_ID, revision=JJ_REV).eval()
    with torch.inference_mode():
        enc = tok(textos, padding=True, truncation=True, max_length=JJ_MAXLEN, return_tensors="pt")
        p = torch.softmax(model(**enc).logits.float(), -1).numpy()[:, JJ_IDX]
    pred = (p >= TAU).astype(int)
    cache = np.load(CACHE_JJ)[:, JJ_IDX]
    pd.DataFrame({"Texto": textos, "Clase": ia["Clase"], "p_sensacionalista": p,
                  "prediccion": np.where(pred == 1, "Sensacionalista", "No sensacionalista"),
                  "acierto": pred == y}).to_csv(SALIDA / "tarea1_predicciones_sinteticos.csv", index=False, encoding="utf-8-sig")
    return {
        "modelo": f"{JJ_ID}@{JJ_REV[:7]}", "max_length": JJ_MAXLEN, "umbral_calibrado": TAU,
        "n": int(len(y)), "positivos": int(y.sum()), "aciertos": int((pred == y).sum()),
        "f1_macro": float(f1_score(y, pred, average="macro")), "f1_macro_ic95_bootstrap": ic_bootstrap_f1(y, pred),
        "matriz_confusion_[[VN,FP],[FN,VP]]": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "p_min_sensacionalistas": float(p[y == 1].min()), "p_max_no_sensacionalistas": float(p[y == 0].max()),
        "diferencia_max_con_cache_original": float(np.abs(p - cache).max()),
    }


def tarea2():
    """Longitud del corpus ampliado en subpalabras (tokenizador de SaBERT) y en caracteres."""
    from transformers import AutoTokenizer

    # NFC: mismo texto que ven los modelos tras corregir_nfc.py (las tildes NFD de Omdena cuentan caracteres de más)
    textos = [unicodedata.normalize("NFC", t) for t in pd.read_csv(CORPUS)["Text"].astype(str)]
    tok = AutoTokenizer.from_pretrained(SABERT_ID, revision=SABERT_REV)
    n_sub = np.array([len(ids) for ids in tok(textos, add_special_tokens=False)["input_ids"]])
    n_car = np.array([len(t) for t in textos])
    n = len(textos)

    def resumen(v):
        return {"mediana": float(np.median(v)), "p25": float(np.percentile(v, 25)), "p75": float(np.percentile(v, 75)), "max": int(v.max())}

    pd.DataFrame({"subpalabras_sin_especiales": n_sub, "caracteres": n_car}).to_csv(SALIDA / "tarea2_longitudes_corpus.csv", index=False)
    return {
        "tokenizador": f"{SABERT_ID}@{SABERT_REV[:7]}", "n_noticias": n,
        "mas_de_256_subpalabras_con_CLS_SEP": {"n": int((n_sub + 2 > 256).sum()), "pct": float(100 * (n_sub + 2 > 256).mean())},
        "mas_de_256_subpalabras_sin_especiales": {"n": int((n_sub > 256).sum()), "pct": float(100 * (n_sub > 256).mean())},
        "mas_de_1000_caracteres": {"n": int((n_car > 1000).sum()), "pct": float(100 * (n_car > 1000).mean())},
        "subpalabras": resumen(n_sub), "caracteres": resumen(n_car),
    }


def tarea3(n_boot=10000):
    """Bootstrap pareado por noticia de la diferencia de ganancia (explicación por oraciones − método)."""
    docs = json.loads(XAI.read_text(encoding="utf-8"))["documentos"]
    nombres = {"jjneila": "Sensacionalismo", "sabert": "Veracidad (SaBERT)", "redundancia": "Redundancia", "modelo_variables": "Clasificador interpretable"}
    rng = np.random.default_rng(SEED)
    out, filas = {}, []
    for clave, nombre in nombres.items():
        def ganancia(d, m):
            v = d["metodos"].get(m, {}).get("aopc")
            return np.nan if v is None else v - d["aleatorio"]["oracion"]["aopc"]

        # «MexGen» es la clave interna de la explicación por oraciones en metricas_comparacion_xai.json
        g = {m: np.array([ganancia(d, m) for d in docs[clave]], dtype=float) for m in ("MexGen", "LIME-oración", "SHAP-oración")}
        out[nombre] = {"media_ganancia": {m: float(np.nanmean(v)) for m, v in g.items()}}
        for m in ("LIME-oración", "SHAP-oración"):
            dif = g["MexGen"] - g[m]
            dif = dif[~np.isnan(dif)]
            idx = rng.integers(0, len(dif), (n_boot, len(dif)))
            medias = dif[idx].mean(axis=1)
            ic = [float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))]
            out[nombre][f"oraciones_menos_{m}"] = {"n_pares": int(len(dif)), "diferencia_media": float(dif.mean()), "ic95": ic}
            filas.append({"modelo": nombre, "comparacion": f"Expl. por oraciones − {m}", "n_pares": len(dif),
                          "diferencia_media": round(float(dif.mean()), 4), "ic95_inf": round(ic[0], 4), "ic95_sup": round(ic[1], 4),
                          "ic_incluye_0": ic[0] <= 0 <= ic[1]})
    pd.DataFrame(filas).to_csv(SALIDA / "tarea3_bootstrap_pareado.csv", index=False, encoding="utf-8-sig")
    return {"n_boot": n_boot, "semilla": SEED, "metodo": "bootstrap pareado por noticia, IC percentil 95 %",
            "ganancia": "AOPC del método menos AOPC del borrado aleatorio por oraciones de la misma noticia", "resultados": out}


TAREAS = {"1": ("tarea1_sinteticos_jjneila", tarea1), "2": ("tarea2_longitud_corpus", tarea2), "3": ("tarea3_bootstrap_tabla_4_18", tarea3)}


def main(seleccion):
    SALIDA.mkdir(parents=True, exist_ok=True)
    res = json.loads(RESULTADOS.read_text(encoding="utf-8")) if RESULTADOS.exists() else {}
    for t in seleccion:
        clave, funcion = TAREAS[t]
        res[clave] = funcion()
    res = {clave: res[clave] for clave, _ in TAREAS.values() if clave in res}
    RESULTADOS.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items()}, ensure_ascii=True, indent=1))


if __name__ == "__main__":
    main([a for a in sys.argv[1:] if a in TAREAS] or list(TAREAS))
