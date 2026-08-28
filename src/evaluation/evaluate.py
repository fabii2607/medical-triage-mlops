"""Avaliação independente de um modelo de triagem já treinado.

Uso:
    uv run --no-sync python -m src.evaluation.evaluate --split validation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.config import (
    MODEL_PATH,
    SPLITS_DIR,
    TEST_METRICS_PATH,
    VALIDATION_METRICS_PATH,
)
from src.constants import TRIAGE_LEVELS
from src.evaluation.metrics import evaluate_predictions, measure_latency
from src.parameters import load_params

REQUIRED_COLUMNS = ("medical_abstract", "triage_level")
METRICS_PATHS = {
    "validation": VALIDATION_METRICS_PATH,
    "test": TEST_METRICS_PATH,
}


def load_evaluation_data(path: Path) -> pd.DataFrame:
    """Carrega e valida o contrato mínimo de um split de avaliação."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não existe. Rode antes: python -m src.data.split"
        )

    data = pd.read_csv(path)
    missing_columns = set(REQUIRED_COLUMNS) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(missing_columns)}")
    if data.empty:
        raise ValueError("O split de avaliação está vazio")
    if data[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("O split de avaliação contém valores nulos")

    unexpected_levels = set(data["triage_level"]) - set(TRIAGE_LEVELS)
    if unexpected_levels:
        raise ValueError(f"Classes inesperadas: {sorted(unexpected_levels)}")

    return data


def evaluate_model(model, data: pd.DataFrame, latency_samples: int) -> dict:
    """Calcula métricas preditivas e latência sem reajustar o modelo."""
    predictions = model.predict(data["medical_abstract"])
    result = evaluate_predictions(
        data["triage_level"],
        predictions,
        TRIAGE_LEVELS,
    )
    result["latency"] = measure_latency(
        model,
        data["medical_abstract"],
        n_samples=latency_samples,
    )
    return result


def save_metrics(metrics: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=tuple(METRICS_PATHS), required=True)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    input_path = args.input or SPLITS_DIR / f"{args.split}.csv"
    output_path = args.output or METRICS_PATHS[args.split]

    if not args.model.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {args.model}. "
            "Rode antes: python -m src.training.train"
        )

    data = load_evaluation_data(input_path)
    model = joblib.load(args.model)
    metrics = {
        "model": "logreg_tfidf",
        "split": args.split,
        "n_samples": len(data),
        "model_size_mb": round(args.model.stat().st_size / 1024**2, 2),
        **evaluate_model(model, data, load_params().evaluate.latency_samples),
    }
    save_metrics(metrics, output_path)

    urgent_recall = metrics["per_class"]["urgente"]["recall"]
    print(
        f"{args.split}: accuracy={metrics['accuracy']} "
        f"macro_f1={metrics['macro_f1']} "
        f"recall_urgente={round(urgent_recall, 4)}"
    )
    print(f"Métricas salvas em: {output_path}")


if __name__ == "__main__":
    main()
