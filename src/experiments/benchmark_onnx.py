"""Benchmark de latência: pipeline sklearn original vs. exportação ONNX.

Compara os dois backends sobre o conjunto de teste com predição unitária
(um texto por chamada, como na API): latência média/P95, tamanho do
artefato, concordância das predições e análise das divergências (deltas
de probabilidade nos casos em que os backends discordam).

Este módulo não altera o pipeline operacional; o resultado é gravado em
`docs/results/onnx_benchmark.json` e alimenta a tabela do README.

Uso:
    uv run python -m src.export_onnx          # gera models/logreg_tfidf.onnx
    uv run python -m src.experiments.benchmark_onnx
"""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

from src.config import MODEL_PATH, RESULTS_DIR, SPLITS_DIR
from src.export_onnx import ONNX_INPUT_NAME, ONNX_MODEL_PATH

BENCHMARK_RESULTS_PATH = RESULTS_DIR / "onnx_benchmark.json"
LATENCY_SAMPLES = 200


def _latency_stats(timings_ms: list[float]) -> dict:
    return {
        "mean_ms": round(float(np.mean(timings_ms)), 3),
        "p95_ms": round(float(np.percentile(timings_ms, 95)), 3),
        "max_ms": round(float(np.max(timings_ms)), 3),
        "n_samples": len(timings_ms),
    }


def _bench(predict_one, texts: list[str]) -> dict:
    timings = []
    for text in texts:
        start = time.perf_counter()
        predict_one(text)
        timings.append((time.perf_counter() - start) * 1000)
    return _latency_stats(timings)


def main() -> None:
    if not ONNX_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{ONNX_MODEL_PATH} não encontrado — rode `python -m src.export_onnx`."
        )

    model = joblib.load(MODEL_PATH)
    session = ort.InferenceSession(
        str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"]
    )

    test = pd.read_csv(SPLITS_DIR / "test.csv")
    texts = test["medical_abstract"].astype(str).tolist()

    # Concordância de predições no test set completo.
    sk_pred = model.predict(texts)
    sk_proba = model.predict_proba(texts)
    onnx_input = np.array(texts, dtype=object).reshape(-1, 1)
    onnx_labels, onnx_proba = session.run(None, {ONNX_INPUT_NAME: onnx_input})

    disagree = sk_pred != onnx_labels
    n_disagree = int(disagree.sum())
    agreement = round(1 - n_disagree / len(texts), 6)

    # Nas divergências, mede o quão limítrofe era a decisão do sklearn:
    # margem entre as duas classes mais prováveis e o maior delta de
    # probabilidade entre os backends.
    divergence_analysis = {}
    if n_disagree:
        margins = np.sort(sk_proba[disagree], axis=1)
        sklearn_margins = margins[:, -1] - margins[:, -2]
        max_proba_delta = np.abs(sk_proba[disagree] - onnx_proba[disagree]).max()
        divergence_analysis = {
            "n_disagreements": n_disagree,
            "sklearn_margin_mean": round(float(sklearn_margins.mean()), 4),
            "sklearn_margin_max": round(float(sklearn_margins.max()), 4),
            "max_probability_delta": round(float(max_proba_delta), 4),
        }

    # Latência unitária, mesma amostra para os dois backends.
    sample = texts[:LATENCY_SAMPLES]
    sklearn_latency = _bench(lambda t: model.predict([t]), sample)
    onnx_latency = _bench(
        lambda t: session.run(None, {ONNX_INPUT_NAME: np.array([[t]], dtype=object)}),
        sample,
    )

    results = {
        "test_set_size": len(texts),
        "prediction_agreement": agreement,
        "divergence_analysis": divergence_analysis,
        "sklearn": {
            "artifact": MODEL_PATH.name,
            "size_kb": round(MODEL_PATH.stat().st_size / 1024, 1),
            "latency": sklearn_latency,
        },
        "onnx": {
            "artifact": ONNX_MODEL_PATH.name,
            "size_kb": round(ONNX_MODEL_PATH.stat().st_size / 1024, 1),
            "latency": onnx_latency,
        },
        "speedup_mean": round(sklearn_latency["mean_ms"] / onnx_latency["mean_ms"], 2),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Concordância sklearn vs ONNX: {agreement:.4%} ({n_disagree} divergências)")
    if divergence_analysis:
        print(f"Divergências: {divergence_analysis}")
    print(
        f"sklearn: média {sklearn_latency['mean_ms']} ms | "
        f"p95 {sklearn_latency['p95_ms']} ms"
    )
    print(
        f"ONNX:    média {onnx_latency['mean_ms']} ms | p95 {onnx_latency['p95_ms']} ms"
    )
    print(f"Speedup (média): {results['speedup_mean']}x")
    print(f"Resultados salvos em: {BENCHMARK_RESULTS_PATH}")


if __name__ == "__main__":
    main()
