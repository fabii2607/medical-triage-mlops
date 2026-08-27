import pytest
from fastapi.testclient import TestClient

from api.main import app


class FakeTriageModel:
    """Modelo determinístico usado para testar apenas o contrato da API."""

    def __init__(self, model_path):
        self.model_path = model_path
        self._is_loaded = False

    def load(self):
        self._is_loaded = True

    @property
    def is_loaded(self):
        return self._is_loaded

    def predict(self, text):
        return "urgente", {
            "atenção": 0.10,
            "normal": 0.05,
            "urgente": 0.85,
        }


class UnavailableTriageModel(FakeTriageModel):
    """Simula uma falha ao carregar o artefato do modelo."""

    def load(self):
        raise FileNotFoundError("Model artifact not found")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("api.main.TriageModel", FakeTriageModel)

    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_health_when_model_is_not_loaded(monkeypatch):
    monkeypatch.setattr("api.main.TriageModel", UnavailableTriageModel)

    with TestClient(app) as test_client:
        response = test_client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"detail": "Model is not loaded"}


def test_predict_urgent_text(client):
    response = client.post(
        "/predict",
        json={
            "text": (
                "Patient presenting acute myocardial infarction with severe chest pain."
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["triage_level"] == "urgente"

    assert data["model_version"] == "logreg_tfidf_v2"

    assert set(data["probabilities"].keys()) == {
        "atenção",
        "normal",
        "urgente",
    }

    assert sum(data["probabilities"].values()) == pytest.approx(1.0)


def test_predict_empty_text(client):
    response = client.post(
        "/predict",
        json={"text": ""},
    )

    assert response.status_code == 422


def test_predict_whitespace_text(client):
    response = client.post(
        "/predict",
        json={"text": "     "},
    )

    assert response.status_code == 422


def test_predict_missing_text(client):
    response = client.post(
        "/predict",
        json={},
    )

    assert response.status_code == 422
