"""Benchmark opcional entre MLP e Logistic Regression na validação.

Este módulo não faz parte do pipeline operacional nem utiliza o conjunto de teste.

Uso:
    uv run --no-sync python -m src.experiments.benchmark_models [--version v2]
"""

from __future__ import annotations

import argparse
import json
import time

import joblib

from src.config import MODELS_DIR, RESULTS_DIR, SPLITS_DIR
from src.evaluation.evaluate import evaluate_model, load_evaluation_data
from src.parameters import load_params
from src.training.pipeline import build_logreg_pipeline, build_mlp_pipeline
from src.training.train import load_training_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="v2", help="sufixo dos artefatos")
    args = parser.parse_args()

    params = load_params()
    train = load_training_data(SPLITS_DIR / "train.csv")
    validation = load_evaluation_data(SPLITS_DIR / "validation.csv")
    candidates = {
        "mlp_tfidf": build_mlp_pipeline(params.train),
        "logreg_tfidf": build_logreg_pipeline(params.train),
    }

    benchmark_dir = MODELS_DIR / "benchmarks"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    for name, model in candidates.items():
        print(f"\n=== {name} ===")
        start = time.perf_counter()
        model.fit(train["medical_abstract"], train["triage_level"])
        train_seconds = time.perf_counter() - start

        model_path = benchmark_dir / f"{name}_{args.version}.joblib"
        joblib.dump(model, model_path)
        metrics = evaluate_model(
            model,
            validation,
            latency_samples=params.evaluate.latency_samples,
        )
        metrics.update(
            {
                "model": name,
                "version": args.version,
                "n_train": len(train),
                "n_validation": len(validation),
                "train_seconds": round(train_seconds, 2),
                "model_size_mb": round(model_path.stat().st_size / 1024**2, 2),
            }
        )
        all_metrics[name] = metrics

        urgent_recall = metrics["per_class"]["urgente"]["recall"]
        print(
            f"validation: accuracy={metrics['accuracy']} "
            f"macro_f1={metrics['macro_f1']} "
            f"recall_urgente={round(urgent_recall, 4)}"
        )
        print(f"Modelo de benchmark salvo em: {model_path}")

    metrics_path = RESULTS_DIR / f"benchmark_metrics_{args.version}.json"
    metrics_path.write_text(
        json.dumps(all_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nMétricas do benchmark salvas em: {metrics_path}")


if __name__ == "__main__":
    main()
