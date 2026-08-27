"""Treino dos modelos de triagem sobre os splits pseudo-rotulados.

Treina dois candidatos com os mesmos hiperparâmetros de TF-IDF do baseline v1
(notebook 06), para comparação justa:
- MLPClassifier (mesma configuração do baseline v1);
- LogisticRegression com class_weight='balanced' (alavanca para o recall
  de `urgente`, que ficou em 0.000 no baseline v1 — MLP não suporta
  class_weight).

Artefatos são salvos com sufixo de versão para não sobrescrever o baseline v1.

Uso:
    uv run --no-sync python -m src.training.train_mlp [--version v2]
"""

from __future__ import annotations

import argparse
import json
import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from src.config import (
    MODELS_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    SPLITS_DIR,
    TRIAGE_LEVELS,
)
from src.evaluation.metrics import evaluate_predictions, measure_latency


def build_tfidf() -> TfidfVectorizer:
    """Mesma configuração do baseline v1 (notebook 06)."""
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=5000,
        stop_words="english",
        sublinear_tf=True,
    )


def build_candidates() -> dict[str, Pipeline]:
    return {
        "mlp_tfidf": Pipeline(
            [
                ("tfidf", build_tfidf()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(64,),
                        activation="relu",
                        solver="adam",
                        alpha=1e-4,
                        learning_rate_init=1e-3,
                        max_iter=200,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=12,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "logreg_tfidf": Pipeline(
            [
                ("tfidf", build_tfidf()),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def load_split(name: str) -> pd.DataFrame:
    path = SPLITS_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não existe. Rode antes: python -m src.data.split"
        )
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="v2", help="sufixo dos artefatos gerados")
    args = parser.parse_args()

    train = load_split("train")
    validation = load_split("validation")
    test = load_split("test")
    print(f"train={len(train):,} validation={len(validation):,} test={len(test):,}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    for name, pipeline in build_candidates().items():
        print(f"\n=== {name} ===")
        start = time.perf_counter()
        pipeline.fit(train["medical_abstract"], train["triage_level"])
        train_seconds = time.perf_counter() - start

        model_path = MODELS_DIR / f"{name}_{args.version}.joblib"
        joblib.dump(pipeline, model_path)

        metrics = {
            "model": name,
            "version": args.version,
            "n_train": len(train),
            "n_validation": len(validation),
            "n_test": len(test),
            "train_seconds": round(train_seconds, 2),
            "model_size_mb": round(model_path.stat().st_size / 1024**2, 2),
            "random_state": RANDOM_STATE,
        }
        for split_name, split_df in [("validation", validation), ("test", test)]:
            y_pred = pipeline.predict(split_df["medical_abstract"])
            metrics[split_name] = evaluate_predictions(
                split_df["triage_level"], y_pred, TRIAGE_LEVELS
            )
        metrics["latency"] = measure_latency(pipeline, test["medical_abstract"])

        all_metrics[name] = metrics

        test_metrics = metrics["test"]
        urgente = test_metrics["per_class"]["urgente"]
        print(
            f"test: accuracy={test_metrics['accuracy']} "
            f"macro_f1={test_metrics['macro_f1']} "
            f"recall_urgente={round(urgente['recall'], 4)}"
        )
        print(f"salvo: {model_path}")

    metrics_path = RESULTS_DIR / f"triage_metrics_{args.version}.json"
    metrics_path.write_text(
        json.dumps(all_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nMétricas salvas em: {metrics_path}")


if __name__ == "__main__":
    main()
