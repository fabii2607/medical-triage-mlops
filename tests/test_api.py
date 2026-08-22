import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_urgent_text(client):
    response = client.post(
        "/predict",
        json={
            "text": (
                "Patient presenting acute myocardial "
                "infarction with severe chest pain."
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["triage_level"] in {
        "urgente",
        "atenção",
        "normal",
    }

    assert data["model_version"] == "logreg_tfidf_v2"

    assert set(data["probabilities"].keys()) == {
        "atenção",
        "normal",
        "urgente",
    }

    assert sum(
        data["probabilities"].values()
    ) == pytest.approx(1.0)


def test_predict_empty_text(client):
    response = client.post(
        "/predict",
        json={
            "text": ""
        },
    )

    assert response.status_code == 422


def test_predict_whitespace_text(client):
    response = client.post(
        "/predict",
        json={
            "text": "     "
        },
    )

    assert response.status_code == 422


def test_predict_missing_text(client):
    response = client.post(
        "/predict",
        json={},
    )

    assert response.status_code == 422