"""Quality gate técnico aplicado às métricas do split de validação.

Uso:
    uv run --no-sync python -m src.evaluation.quality_gate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import QUALITY_GATE_PATH, VALIDATION_METRICS_PATH
from src.constants import TRIAGE_LEVELS
from src.parameters import QualityGateParams, load_params


class QualityGateError(RuntimeError):
    """Indica que o modelo candidato não cumpriu o gate."""


def load_metrics(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de métricas não encontrado: {path}")

    metrics = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metrics, dict):
        raise ValueError("O arquivo de métricas deve conter um objeto JSON")
    return metrics


def build_gate_report(metrics: dict, params: QualityGateParams) -> dict:
    """Compara métricas da validação com os limites configurados."""
    if not params.enabled:
        return {
            "status": "disabled",
            "passed": True,
            "split": metrics.get("split"),
            "checks": {},
            "failures": [],
        }

    if metrics.get("split") != "validation":
        raise ValueError("O quality gate deve receber métricas da validação")

    try:
        macro_f1 = float(metrics["macro_f1"])
        urgent_recall = float(metrics["per_class"]["urgente"]["recall"])
        observed_classes = set(metrics["per_class"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Estrutura inválida no arquivo de métricas") from error

    expected_classes = set(TRIAGE_LEVELS)
    classes_present = expected_classes.issubset(observed_classes)
    macro_f1_passed = macro_f1 >= params.min_validation_macro_f1
    urgent_recall_passed = urgent_recall >= params.min_validation_urgent_recall

    checks = {
        "required_classes": {
            "expected": list(TRIAGE_LEVELS),
            "observed": sorted(observed_classes & expected_classes),
            "passed": classes_present,
        },
        "macro_f1": {
            "value": macro_f1,
            "minimum": params.min_validation_macro_f1,
            "passed": macro_f1_passed,
        },
        "urgent_recall": {
            "value": urgent_recall,
            "minimum": params.min_validation_urgent_recall,
            "passed": urgent_recall_passed,
        },
    }
    failures = [name for name, check in checks.items() if not check["passed"]]

    return {
        "status": "approved" if not failures else "rejected",
        "passed": not failures,
        "split": "validation",
        "checks": checks,
        "failures": failures,
    }


def save_gate_report(report: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_quality_gate(
    metrics_path: Path,
    output_path: Path,
    params: QualityGateParams,
) -> dict:
    report = build_gate_report(load_metrics(metrics_path), params)
    save_gate_report(report, output_path)
    if not report["passed"]:
        failures = ", ".join(report["failures"])
        raise QualityGateError(f"Quality gate reprovado: {failures}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=VALIDATION_METRICS_PATH)
    parser.add_argument("--output", type=Path, default=QUALITY_GATE_PATH)
    args = parser.parse_args()

    try:
        report = run_quality_gate(
            args.metrics,
            args.output,
            load_params().quality_gate,
        )
    except QualityGateError as error:
        raise SystemExit(str(error)) from error

    print(f"Quality gate: {report['status']}")
    print(f"Relatório salvo em: {args.output}")


if __name__ == "__main__":
    main()
