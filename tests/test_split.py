import numpy as np
import pandas as pd
import pytest

from src.data.split import clean, make_splits
from src.parameters import SplitParams


@pytest.fixture
def dataset():
    rng = np.random.default_rng(0)
    n = 400
    return pd.DataFrame(
        {
            "medical_abstract": [f"abstract único número {i}" for i in range(n)],
            "triage_level": rng.choice(
                ["normal", "atenção", "urgente"], size=n, p=[0.45, 0.40, 0.15]
            ),
        }
    )


@pytest.fixture
def split_params():
    return SplitParams(
        random_state=42,
        train_size=0.70,
        validation_size=0.15,
        test_size=0.15,
    )


def test_proporcoes_70_15_15(dataset, split_params):
    train, validation, test = make_splits(dataset, split_params)
    total = len(dataset)
    assert len(train) == pytest.approx(0.70 * total, abs=2)
    assert len(validation) == pytest.approx(0.15 * total, abs=2)
    assert len(test) == pytest.approx(0.15 * total, abs=2)
    assert len(train) + len(validation) + len(test) == total


def test_sem_vazamento_entre_splits(dataset, split_params):
    train, validation, test = make_splits(dataset, split_params)
    assert not set(train["medical_abstract"]) & set(validation["medical_abstract"])
    assert not set(train["medical_abstract"]) & set(test["medical_abstract"])
    assert not set(validation["medical_abstract"]) & set(test["medical_abstract"])


def test_estratificacao_preserva_distribuicao(dataset, split_params):
    train, validation, test = make_splits(dataset, split_params)
    original = dataset["triage_level"].value_counts(normalize=True)
    for part in (train, validation, test):
        dist = part["triage_level"].value_counts(normalize=True)
        for level in original.index:
            assert dist[level] == pytest.approx(original[level], abs=0.03)


def test_clean_remove_duplicatas_vazios_e_labels_invalidos():
    df = pd.DataFrame(
        {
            "medical_abstract": ["texto a", "texto a", "  ", "texto b", "texto c"],
            "triage_level": ["normal", "normal", "urgente", "inválido", "urgente"],
        }
    )
    result = clean(df)
    assert list(result["medical_abstract"]) == ["texto a", "texto c"]


def test_proporcoes_customizadas(dataset):
    params = SplitParams(
        random_state=42,
        train_size=0.60,
        validation_size=0.20,
        test_size=0.20,
    )
    train, validation, test = make_splits(dataset, params)
    assert len(train) == pytest.approx(0.60 * len(dataset), abs=2)
    assert len(validation) == pytest.approx(0.20 * len(dataset), abs=2)
    assert len(test) == pytest.approx(0.20 * len(dataset), abs=2)
