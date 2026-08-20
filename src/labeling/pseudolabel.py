"""Pseudo-rotulagem do corpus completo com o BioBERT pré-treinado.

Refatoração do notebook 04 com as correções identificadas na revisão:
- processa TODOS os abstracts únicos (sem limite de tempo por padrão);
- embaralha antes da inferência (remove o viés alfabético do groupby);
- max_length=512 (elimina o truncamento que afetava 60% dos textos com 256);
- retomada a partir do checkpoint (--resume);
- parse incremental das predições (sem custo quadrático nos checkpoints).

Uso:
    uv run --no-sync python -m src.labeling.pseudolabel [--resume] [--limit N]
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from src.config import (
    BATCH_SIZE,
    CHECKPOINT_EVERY,
    CHECKPOINT_PATH,
    CONFIDENCE_THRESHOLD,
    LABELS_PATH,
    MAX_LENGTH,
    PROCESSED_DIR,
    PSEUDOLABEL_MODEL,
    PSEUDOLABELED_PATH,
    RANDOM_STATE,
    TEST_PATH,
    TRAIN_PATH,
)
from src.labeling.biobert import load_classifier, load_tokenizer, predict_batch
from src.labeling.triage_rules import map_to_three_levels

RESULT_COLUMNS = [
    "medical_abstract",
    "triage_level",
    "urgent_score",
    "nonurgent_score",
    "confidence",
    "binary_prediction",
    "n_tokens",
    "was_truncated",
    "original_condition_labels",
    "original_condition_names",
    "original_splits",
    "pseudolabel_model",
    "pseudolabel_threshold",
    "max_length",
]


def build_unique_abstracts() -> pd.DataFrame:
    """Consolida treino+teste em abstracts únicos, preservando rastreabilidade."""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    labels = pd.read_csv(LABELS_PATH)

    label_map = dict(zip(labels["condition_label"], labels["condition_name"]))

    combined = pd.concat(
        [train.assign(original_split="train"), test.assign(original_split="test")],
        ignore_index=True,
    )
    combined["medical_abstract"] = combined["medical_abstract"].astype(str).str.strip()

    unique_abstracts = (
        combined.groupby("medical_abstract")
        .agg(
            original_condition_labels=(
                "condition_label",
                lambda values: "|".join(map(str, sorted(set(values)))),
            ),
            original_condition_names=(
                "condition_label",
                lambda values: "|".join(
                    label_map[value] for value in sorted(set(values))
                ),
            ),
            original_splits=(
                "original_split",
                lambda values: "|".join(sorted(set(values))),
            ),
        )
        .reset_index()
    )

    # O groupby ordena alfabeticamente; o shuffle garante que qualquer
    # interrupção deixe um subconjunto representativo, não enviesado.
    return unique_abstracts.sample(frac=1, random_state=RANDOM_STATE).reset_index(
        drop=True
    )


def add_token_stats(df: pd.DataFrame, tokenizer, max_length: int) -> pd.DataFrame:
    encodings = tokenizer(
        df["medical_abstract"].tolist(),
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]
    df = df.copy()
    df["n_tokens"] = [len(ids) for ids in encodings]
    df["was_truncated"] = df["n_tokens"] > max_length
    truncated = df["was_truncated"].mean()
    print(f"Textos acima de {max_length} tokens: {df['was_truncated'].sum():,} ({truncated:.1%})")
    return df


def finalize(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def run(resume: bool = False, limit: int | None = None) -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer()
    unique_abstracts = build_unique_abstracts()
    print(f"Abstracts únicos: {len(unique_abstracts):,}")

    done_rows: list[dict] = []
    if resume and CHECKPOINT_PATH.exists():
        checkpoint = pd.read_csv(CHECKPOINT_PATH)
        # Só aproveita registros gerados com a mesma configuração.
        compatible = checkpoint[
            (checkpoint["max_length"] == MAX_LENGTH)
            & (checkpoint["pseudolabel_threshold"] == CONFIDENCE_THRESHOLD)
            & (checkpoint["pseudolabel_model"] == PSEUDOLABEL_MODEL)
        ]
        done_rows = compatible[RESULT_COLUMNS].to_dict("records")
        done_texts = set(compatible["medical_abstract"])
        unique_abstracts = unique_abstracts[
            ~unique_abstracts["medical_abstract"].isin(done_texts)
        ].reset_index(drop=True)
        print(f"Retomando do checkpoint: {len(done_rows):,} já processados, "
              f"{len(unique_abstracts):,} restantes")

    # Com --limit (smoke test), grava num arquivo separado para não
    # sobrescrever o dataset completo nem o checkpoint.
    if limit is not None:
        unique_abstracts = unique_abstracts.head(limit)
        checkpoint_path = PROCESSED_DIR / "pseudolabel_smoke_test.csv"
    else:
        checkpoint_path = CHECKPOINT_PATH

    unique_abstracts = add_token_stats(unique_abstracts, tokenizer, MAX_LENGTH)

    classifier = load_classifier()

    rows = list(done_rows)
    total = len(done_rows) + len(unique_abstracts)
    start_time = time.perf_counter()
    since_checkpoint = 0

    for start in range(0, len(unique_abstracts), BATCH_SIZE):
        batch = unique_abstracts.iloc[start : start + BATCH_SIZE]
        parsed = predict_batch(classifier, batch["medical_abstract"].tolist())

        for (_, record), (binary_prediction, urgent, nonurgent, confidence) in zip(
            batch.iterrows(), parsed
        ):
            rows.append(
                {
                    "medical_abstract": record["medical_abstract"],
                    "triage_level": map_to_three_levels(urgent, nonurgent),
                    "urgent_score": urgent,
                    "nonurgent_score": nonurgent,
                    "confidence": confidence,
                    "binary_prediction": binary_prediction,
                    "n_tokens": record["n_tokens"],
                    "was_truncated": record["was_truncated"],
                    "original_condition_labels": record["original_condition_labels"],
                    "original_condition_names": record["original_condition_names"],
                    "original_splits": record["original_splits"],
                    "pseudolabel_model": PSEUDOLABEL_MODEL,
                    "pseudolabel_threshold": CONFIDENCE_THRESHOLD,
                    "max_length": MAX_LENGTH,
                }
            )

        since_checkpoint += len(batch)
        if since_checkpoint >= CHECKPOINT_EVERY:
            finalize(rows).to_csv(checkpoint_path, index=False)
            since_checkpoint = 0
            elapsed = time.perf_counter() - start_time
            rate = (len(rows) - len(done_rows)) / elapsed
            remaining = (total - len(rows)) / rate if rate > 0 else float("nan")
            print(
                f"{len(rows):,}/{total:,} processados "
                f"({rate:.1f} textos/s, ~{remaining / 60:.1f} min restantes)",
                flush=True,
            )

    result = finalize(rows)
    result.to_csv(checkpoint_path, index=False)
    if limit is None:
        result.to_csv(PSEUDOLABELED_PATH, index=False)
        saved_path = PSEUDOLABELED_PATH
    else:
        saved_path = checkpoint_path

    elapsed = time.perf_counter() - start_time
    print(f"\nConcluído: {len(result):,} abstracts em {elapsed / 60:.1f} min")
    print("\nDistribuição de triage_level:")
    print(result["triage_level"].value_counts())
    print(f"\nSalvo em: {saved_path}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="retoma do checkpoint, pulando abstracts já processados",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="processa apenas N abstracts (para teste rápido)",
    )
    args = parser.parse_args()
    run(resume=args.resume, limit=args.limit)


if __name__ == "__main__":
    main()
