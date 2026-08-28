from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.config import PARAMS_PATH
from src.parameters import SplitParams, load_params


def test_load_project_params():
    params = load_params()

    assert params.labeling.confidence_threshold == 0.70
    assert params.labeling.max_length == 512
    assert params.split.train_size == 0.70
    assert params.train.tfidf.ngram_range == (1, 2)
    assert params.train.logistic_regression.class_weight == "balanced"


def test_params_path_points_to_project_yaml():
    assert Path(__file__).resolve().parents[1] / "params.yaml" == PARAMS_PATH


def test_missing_params_file_is_rejected(tmp_path):
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="Arquivo de parâmetros"):
        load_params(missing_path)


def test_invalid_split_proportions_are_rejected():
    with pytest.raises(ValidationError, match=r"devem somar 1\.0"):
        SplitParams(
            random_state=42,
            train_size=0.70,
            validation_size=0.20,
            test_size=0.20,
        )


def test_non_mapping_yaml_is_rejected(tmp_path):
    params_path = tmp_path / "params.yaml"
    params_path.write_text(yaml.safe_dump(["not", "a", "mapping"]), encoding="utf-8")

    with pytest.raises(ValueError, match="deve conter um mapa YAML"):
        load_params(params_path)
