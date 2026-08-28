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
    CHECKPOINT_PATH,
    LABELS_PATH,
    PROCESSED_DIR,
    PSEUDOLABELED_PATH,
    TEST_PATH,
    TRAIN_PATH,
)
from src.labeling.biobert import load_classifier, load_tokenizer, predict_batch
from src.labeling.triage_rules import map_to_three_levels
from src.parameters import LabelingParams, load_params

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


def build_unique_abstracts(random_state: int) -> pd.DataFrame:
    """Consolida treino+teste em abstracts únicos, preservando rastreabilidade."""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    labels = pd.read_csv(LABELS_PATH)

    label_map = dict(
        zip(
            labels["condition_label"],
            labels["condition_name"],
            strict=True,
        )
    )

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
    return unique_abstracts.sample(frac=1, random_state=random_state).reset_index(
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
    print(
        f"Textos acima de {max_length} tokens: {df['was_truncated'].sum():,} ({truncated:.1%})"
    )
    return df


def finalize(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def run(
    params: LabelingParams,
    resume: bool = False,
    limit: int | None = None,
    checkpoint_every: int = 500,
) -> pd.DataFrame:
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every deve ser maior que zero")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(params.model_name)
    unique_abstracts = build_unique_abstracts(params.random_state)
    print(f"Abstracts únicos: {len(unique_abstracts):,}")

    done_rows: list[dict] = []
    if resume and CHECKPOINT_PATH.exists():
        checkpoint = pd.read_csv(CHECKPOINT_PATH)
        # Só aproveita registros gerados com a mesma configuração.
        compatible = checkpoint[
            (checkpoint["max_length"] == params.max_length)
            & (checkpoint["pseudolabel_threshold"] == params.confidence_threshold)
            & (checkpoint["pseudolabel_model"] == params.model_name)
        ]
        done_rows = compatible[RESULT_COLUMNS].to_dict("records")
        done_texts = set(compatible["medical_abstract"])
        unique_abstracts = unique_abstracts[
            ~unique_abstracts["medical_abstract"].isin(done_texts)
        ].reset_index(drop=True)
        print(
            f"Retomando do checkpoint: {len(done_rows):,} já processados, "
            f"{len(unique_abstracts):,} restantes"
        )

    # Com --limit (smoke test), grava num arquivo separado para não
    # sobrescrever o dataset completo nem o checkpoint.
    if limit is not None:
        unique_abstracts = unique_abstracts.head(limit)
        checkpoint_path = PROCESSED_DIR / "pseudolabel_smoke_test.csv"
    else:
        checkpoint_path = CHECKPOINT_PATH

    unique_abstracts = add_token_stats(unique_abstracts, tokenizer, params.max_length)

    classifier = load_classifier(params.model_name)

    rows = list(done_rows)
    total = len(done_rows) + len(unique_abstracts)
    start_time = time.perf_counter()
    since_checkpoint = 0

    for start in range(0, len(unique_abstracts), params.batch_size):
        batch = unique_abstracts.iloc[start : start + params.batch_size]
        parsed = predict_batch(
            classifier,
            batch["medical_abstract"].tolist(),
            batch_size=params.batch_size,
            max_length=params.max_length,
        )

        for (_, record), (binary_prediction, urgent, nonurgent, confidence) in zip(
            batch.iterrows(), parsed, strict=True
        ):
            rows.append(
                {
                    "medical_abstract": record["medical_abstract"],
                    "triage_level": map_to_three_levels(
                        urgent,
                        nonurgent,
                        threshold=params.confidence_threshold,
                    ),
                    "urgent_score": urgent,
                    "nonurgent_score": nonurgent,
                    "confidence": confidence,
                    "binary_prediction": binary_prediction,
                    "n_tokens": record["n_tokens"],
                    "was_truncated": record["was_truncated"],
                    "original_condition_labels": record["original_condition_labels"],
                    "original_condition_names": record["original_condition_names"],
                    "original_splits": record["original_splits"],
                    "pseudolabel_model": params.model_name,
                    "pseudolabel_threshold": params.confidence_threshold,
                    "max_length": params.max_length,
                }
            )

        since_checkpoint += len(batch)
        if since_checkpoint >= checkpoint_every:
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
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
        help="salva o progresso a cada N abstracts processados",
    )
    args = parser.parse_args()
    run(
        load_params().labeling,
        resume=args.resume,
        limit=args.limit,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
