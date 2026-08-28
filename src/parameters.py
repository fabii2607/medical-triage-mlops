"""Carregamento e validação dos parâmetros do pipeline."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.config import PARAMS_PATH


class ParamsModel(BaseModel):
    """Modelo base que rejeita parâmetros desconhecidos."""

    model_config = ConfigDict(extra="forbid")


class LabelingParams(ParamsModel):
    model_name: str = Field(min_length=1)
    confidence_threshold: float = Field(gt=0.5, le=1.0)
    max_length: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    random_state: int


class SplitParams(ParamsModel):
    random_state: int
    train_size: float = Field(gt=0.0, lt=1.0)
    validation_size: float = Field(gt=0.0, lt=1.0)
    test_size: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_proportions(self) -> SplitParams:
        total = self.train_size + self.validation_size + self.test_size
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"As proporções do split devem somar 1.0; recebido {total}"
            )
        return self


class TfidfParams(ParamsModel):
    ngram_range: tuple[int, int]
    min_df: int = Field(gt=0)
    max_df: float = Field(gt=0.0, le=1.0)
    max_features: int = Field(gt=0)
    stop_words: str | None
    sublinear_tf: bool


class MlpParams(ParamsModel):
    hidden_layer_sizes: tuple[int, ...]
    activation: str
    solver: str
    alpha: float = Field(ge=0.0)
    learning_rate_init: float = Field(gt=0.0)
    max_iter: int = Field(gt=0)
    early_stopping: bool
    validation_fraction: float = Field(gt=0.0, lt=1.0)
    n_iter_no_change: int = Field(gt=0)
    random_state: int


class LogisticRegressionParams(ParamsModel):
    C: float = Field(gt=0.0)
    class_weight: str | None
    max_iter: int = Field(gt=0)
    solver: str
    random_state: int


class TrainParams(ParamsModel):
    tfidf: TfidfParams
    mlp: MlpParams
    logistic_regression: LogisticRegressionParams


class EvaluateParams(ParamsModel):
    latency_samples: int = Field(gt=0)


class QualityGateParams(ParamsModel):
    enabled: bool
    min_validation_macro_f1: float | None = Field(default=None, ge=0.0, le=1.0)
    min_validation_urgent_recall: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_enabled_thresholds(self) -> QualityGateParams:
        if self.enabled and (
            self.min_validation_macro_f1 is None
            or self.min_validation_urgent_recall is None
        ):
            raise ValueError("Os limites do quality gate são obrigatórios quando ativo")
        return self


class PipelineParams(ParamsModel):
    labeling: LabelingParams
    split: SplitParams
    train: TrainParams
    evaluate: EvaluateParams
    quality_gate: QualityGateParams


def load_params(path: Path = PARAMS_PATH) -> PipelineParams:
    """Carrega o YAML e devolve parâmetros tipados e validados."""
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de parâmetros não encontrado: {path}")

    with path.open(encoding="utf-8") as params_file:
        raw_params = yaml.safe_load(params_file)

    if not isinstance(raw_params, dict):
        raise ValueError(f"O arquivo de parâmetros deve conter um mapa YAML: {path}")

    return PipelineParams.model_validate(raw_params)
