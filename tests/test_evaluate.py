import json

import pandas as pd
import pytest

from src.evaluation.evaluate import (
    evaluate_model,
    load_evaluation_data,
    save_metrics,
)


class DeterministicModel:
    def predict(self, texts):
        predictions = []
        for text in texts:
            if "emergency" in text:
                predictions.append("urgente")
            elif "attention" in text:
                predictions.append("atenção")
            else:
                predictions.append("normal")
        return predictions


@pytest.fixture
def evaluation_dataset():
    return pd.DataFrame(
        {
            "medical_abstract": [
                "routine clinical followup",
                "moderate symptoms require attention",
                "acute cardiac emergency",
            ],
            "triage_level": ["normal", "atenção", "urgente"],
        }
    )


def test_evaluate_model_without_training(evaluation_dataset):
    metrics = evaluate_model(
        DeterministicModel(),
        evaluation_dataset,
        latency_samples=3,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["per_class"]["urgente"]["recall"] == 1.0
    assert metrics["latency"]["n_samples"] == 3


def test_save_metrics_as_json(tmp_path):
    output_path = tmp_path / "metrics.json"
    metrics = {"macro_f1": 0.75, "per_class": {"urgente": {"recall": 0.80}}}

    save_metrics(metrics, output_path)

    assert json.loads(output_path.read_text(encoding="utf-8")) == metrics


def test_load_evaluation_data_rejects_unexpected_class(tmp_path):
    input_path = tmp_path / "validation.csv"
    pd.DataFrame(
        {
            "medical_abstract": ["example"],
            "triage_level": ["desconhecido"],
        }
    ).to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="Classes inesperadas"):
        load_evaluation_data(input_path)
