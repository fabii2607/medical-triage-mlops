import pytest

from src.evaluation.quality_gate import build_gate_report
from src.parameters import QualityGateParams


@pytest.fixture
def gate_params():
    return QualityGateParams(
        enabled=True,
        min_validation_macro_f1=0.70,
        min_validation_urgent_recall=0.70,
    )


def validation_metrics(macro_f1=0.75, urgent_recall=0.80):
    return {
        "split": "validation",
        "macro_f1": macro_f1,
        "per_class": {
            "normal": {"recall": 0.80},
            "atenção": {"recall": 0.75},
            "urgente": {"recall": urgent_recall},
        },
    }


def test_gate_approves_metrics_above_thresholds(gate_params):
    report = build_gate_report(validation_metrics(), gate_params)

    assert report["passed"] is True
    assert report["status"] == "approved"
    assert report["failures"] == []


def test_gate_rejects_low_macro_f1(gate_params):
    report = build_gate_report(validation_metrics(macro_f1=0.69), gate_params)

    assert report["passed"] is False
    assert report["failures"] == ["macro_f1"]


def test_gate_rejects_low_urgent_recall(gate_params):
    report = build_gate_report(validation_metrics(urgent_recall=0.69), gate_params)

    assert report["passed"] is False
    assert report["failures"] == ["urgent_recall"]


def test_gate_rejects_missing_required_class(gate_params):
    metrics = validation_metrics()
    del metrics["per_class"]["atenção"]

    report = build_gate_report(metrics, gate_params)

    assert report["passed"] is False
    assert report["failures"] == ["required_classes"]


def test_gate_rejects_test_metrics(gate_params):
    metrics = validation_metrics()
    metrics["split"] = "test"

    with pytest.raises(ValueError, match="métricas da validação"):
        build_gate_report(metrics, gate_params)


def test_disabled_gate_does_not_require_thresholds():
    params = QualityGateParams(
        enabled=False,
        min_validation_macro_f1=None,
        min_validation_urgent_recall=None,
    )

    report = build_gate_report({}, params)

    assert report["passed"] is True
    assert report["status"] == "disabled"
