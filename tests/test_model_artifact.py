"""Smoke test do artefato real do modelo servido pela API.

Roda apenas quando `models/logreg_tfidf.joblib` existe localmente
(`dvc pull` ou `dvc repro`); no CI, sem credenciais GCP, é pulado.
Teria capturado a divergência de nome entre o artefato treinado e o
caminho esperado pela API.
"""

import pytest

from api.main import MODEL_PATH
from api.model import TriageModel

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason=f"artefato {MODEL_PATH.name} ausente — rode `dvc pull` ou `dvc repro`",
)

URGENT_TEXT = (
    "Patient presenting acute myocardial infarction with severe chest pain "
    "and hypotension requiring immediate intervention."
)


def test_artefato_real_carrega():
    model = TriageModel(MODEL_PATH)
    model.load()
    assert model.is_loaded


def test_artefato_real_prediz_com_contrato_esperado():
    model = TriageModel(MODEL_PATH)
    model.load()

    level, probabilities = model.predict(URGENT_TEXT)

    assert set(probabilities) == {"normal", "atenção", "urgente"}
    assert level in probabilities
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    assert level == "urgente"
    assert probabilities["urgente"] > 0.5
