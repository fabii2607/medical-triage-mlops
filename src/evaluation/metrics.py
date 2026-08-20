"""Métricas de avaliação e latência para os modelos de triagem."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_predictions(y_true, y_pred, labels: list[str]) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "macro_precision": round(
            precision_score(y_true, y_pred, average="macro", zero_division=0), 4
        ),
        "macro_recall": round(
            recall_score(y_true, y_pred, average="macro", zero_division=0), 4
        ),
        "macro_f1": round(
            f1_score(y_true, y_pred, average="macro", zero_division=0), 4
        ),
        "per_class": classification_report(
            y_true, y_pred, labels=labels, zero_division=0, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "confusion_matrix_labels": labels,
    }


def measure_latency(model, texts: pd.Series, n_samples: int = 200) -> dict:
    """Latência de predição unitária (um texto por chamada), em milissegundos."""
    sample = texts.sample(
        n=min(n_samples, len(texts)), random_state=0, replace=False
    ).tolist()

    timings = []
    for text in sample:
        start = time.perf_counter()
        model.predict([text])
        timings.append((time.perf_counter() - start) * 1000)

    return {
        "latency_mean_ms": round(float(np.mean(timings)), 3),
        "latency_p95_ms": round(float(np.percentile(timings, 95)), 3),
        "n_samples": len(timings),
    }
