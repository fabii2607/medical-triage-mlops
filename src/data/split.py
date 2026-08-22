"""Divisão estratificada 70/15/15 do dataset pseudo-rotulado (notebook 05).

Uso:
    uv run --no-sync python -m src.data.split [--input caminho.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CHECKPOINT_PATH,
    PSEUDOLABELED_PATH,
    RANDOM_STATE,
    SPLITS_DIR,
    TRIAGE_LEVELS,
)


def load_pseudolabeled(input_path: Path | None = None) -> pd.DataFrame:
    if input_path is not None:
        path = input_path
    else:
        path = PSEUDOLABELED_PATH if PSEUDOLABELED_PATH.exists() else CHECKPOINT_PATH
    if not path.exists():
        raise FileNotFoundError(
            "Nenhum dataset pseudo-rotulado encontrado. "
            "Rode antes: python -m src.labeling.pseudolabel"
        )
    print(f"Carregando: {path}")
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["medical_abstract"] = df["medical_abstract"].astype(str).str.strip()
    df = df[df["medical_abstract"] != ""]
    df = df.dropna(subset=["medical_abstract", "triage_level"])
    df = df[df["triage_level"].isin(TRIAGE_LEVELS)]
    df = df.drop_duplicates(subset="medical_abstract")
    return df.reset_index(drop=True)


def make_splits(
    df: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """70% treino / 15% validação / 15% teste, estratificado por triage_level."""
    train, rest = train_test_split(
        df,
        test_size=0.30,
        stratify=df["triage_level"],
        random_state=random_state,
    )
    validation, test = train_test_split(
        rest,
        test_size=0.50,
        stratify=rest["triage_level"],
        random_state=random_state,
    )

    for name, part in [("train", train), ("validation", validation), ("test", test)]:
        others = pd.concat([p for n, p in [("train", train), ("validation", validation), ("test", test)] if n != name])
        overlap = set(part["medical_abstract"]) & set(others["medical_abstract"])
        assert len(overlap) == 0, f"Vazamento de texto entre {name} e os demais splits"

    return (
        train.sample(frac=1, random_state=random_state).reset_index(drop=True),
        validation.sample(frac=1, random_state=random_state).reset_index(drop=True),
        test.sample(frac=1, random_state=random_state).reset_index(drop=True),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="CSV pseudo-rotulado alternativo (ex.: saída de --limit da pseudo-rotulagem)",
    )
    args = parser.parse_args()

    df = clean(load_pseudolabeled(args.input))
    print(f"Registros após limpeza: {len(df):,}")

    train, validation, test = make_splits(df)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(SPLITS_DIR / "train.csv", index=False)
    validation.to_csv(SPLITS_DIR / "validation.csv", index=False)
    test.to_csv(SPLITS_DIR / "test.csv", index=False)

    for name, part in [("train", train), ("validation", validation), ("test", test)]:
        dist = part["triage_level"].value_counts().to_dict()
        print(f"{name}: {len(part):,} | {dist}")
    print(f"\nSalvo em: {SPLITS_DIR}")


if __name__ == "__main__":
    main()
