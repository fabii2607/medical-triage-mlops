"""Carregamento e inferência do BioBERT pré-treinado (urgent / non-urgent)."""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoTokenizer, pipeline

from src.config import BATCH_SIZE, MAX_LENGTH, PSEUDOLABEL_MODEL


def load_tokenizer():
    return AutoTokenizer.from_pretrained(PSEUDOLABEL_MODEL)


def load_classifier():
    """Cria o pipeline de classificação, usando GPU quando disponível."""
    device = 0 if torch.cuda.is_available() else -1
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Dispositivo de inferência: {device_name}")

    return pipeline(
        task="text-classification",
        model=PSEUDOLABEL_MODEL,
        tokenizer=PSEUDOLABEL_MODEL,
        device=device,
    )


def normalize_binary_label(label: str) -> str:
    """Normaliza os nomes de label do checkpoint (LABEL_0/LABEL_1 etc.).

    O checkpoint publica apenas LABEL_0/LABEL_1; o mapeamento
    LABEL_1 -> urgent vem do model card (validado no notebook 02).
    """
    normalized = str(label).strip().lower()

    if normalized in {"label_1", "1", "urgent"}:
        return "urgent"

    if normalized in {"label_0", "0", "non-urgent", "non_urgent", "nonurgent"}:
        return "non-urgent"

    return normalized


def parse_prediction(prediction_items) -> tuple[str, float, float, float]:
    """Extrai (binary_prediction, urgent_score, nonurgent_score, confidence)."""
    scores = {
        normalize_binary_label(item["label"]): float(item["score"])
        for item in prediction_items
    }

    urgent_score = scores.get("urgent", np.nan)
    nonurgent_score = scores.get("non-urgent", np.nan)

    if np.isnan(urgent_score) or np.isnan(nonurgent_score):
        raise ValueError(
            f"Não foi possível identificar urgent/non-urgent em: {prediction_items}"
        )

    binary_prediction = "urgent" if urgent_score >= nonurgent_score else "non-urgent"
    confidence = max(urgent_score, nonurgent_score)

    return binary_prediction, urgent_score, nonurgent_score, confidence


def predict_batch(
    classifier,
    texts: list[str],
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
) -> list[tuple[str, float, float, float]]:
    """Roda o pipeline em um lote de textos e devolve as tuplas já parseadas."""
    raw = classifier(
        texts,
        batch_size=batch_size,
        truncation=True,
        max_length=max_length,
        top_k=None,
    )
    return [parse_prediction(items) for items in raw]
