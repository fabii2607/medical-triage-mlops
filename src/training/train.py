"""Treinamento operacional do modelo TF-IDF + Logistic Regression.

Uso:
    uv run --no-sync python -m src.training.train
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import MODEL_PATH, SPLITS_DIR
from src.constants import TRIAGE_LEVELS
from src.parameters import TrainParams, load_params
from src.training.pipeline import build_logreg_pipeline

TRAIN_PATH = SPLITS_DIR / "train.csv"
REQUIRED_COLUMNS = ("medical_abstract", "triage_level")


def load_training_data(path: Path) -> pd.DataFrame:
    """Carrega e valida o contrato mínimo do split de treino."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não existe. Rode antes: python -m src.data.split"
        )

    data = pd.read_csv(path)
    missing_columns = set(REQUIRED_COLUMNS) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(missing_columns)}")
    if data.empty:
        raise ValueError("O split de treino está vazio")
    if data[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("O split de treino contém valores nulos")

    observed_levels = set(data["triage_level"])
    expected_levels = set(TRIAGE_LEVELS)
    if observed_levels != expected_levels:
        raise ValueError(
            "Classes do treino não correspondem ao contrato: "
            f"esperado={sorted(expected_levels)}, observado={sorted(observed_levels)}"
        )

    return data


def train_model(data: pd.DataFrame, params: TrainParams) -> Pipeline:
    """Ajusta somente o modelo operacional sobre o split de treino."""
    model = build_logreg_pipeline(params)
    model.fit(data["medical_abstract"], data["triage_level"])
    return model


def save_model(model: Pipeline, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=TRAIN_PATH)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    data = load_training_data(args.input)
    start = time.perf_counter()
    model = train_model(data, load_params().train)
    train_seconds = time.perf_counter() - start
    save_model(model, args.output)

    print(f"Treinamento concluído em {train_seconds:.2f}s")
    print(f"Registros de treino: {len(data):,}")
    print(f"Modelo salvo em: {args.output}")


if __name__ == "__main__":
    main()
