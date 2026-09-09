"""Testes estruturais da DAG de treinamento do Airflow."""

from __future__ import annotations

import os
from importlib.util import find_spec
from itertools import pairwise
from pathlib import Path

import pytest

AIRFLOW_INSTALLED = find_spec("airflow") is not None

pytestmark = pytest.mark.skipif(
    not AIRFLOW_INSTALLED,
    reason="Airflow é validado em um job isolado do CI",
)

DAGS_DIR = Path(__file__).resolve().parents[1] / "dags"
DAG_ID = "medical_triage_training"
EXPECTED_TASKS = (
    "split",
    "train",
    "evaluate_validation",
    "quality_gate",
    "evaluate_test",
)


@pytest.fixture(scope="module")
def dag_bag(tmp_path_factory):
    """Carrega as DAGs sem usar o diretório global do Airflow."""
    os.environ.setdefault(
        "AIRFLOW_HOME",
        str(tmp_path_factory.mktemp("airflow")),
    )

    from airflow.models.dagbag import DagBag

    return DagBag(dag_folder=str(DAGS_DIR), include_examples=False)


@pytest.fixture(scope="module")
def training_dag(dag_bag):
    """Retorna a DAG de treinamento carregada pelo DagBag."""
    assert not dag_bag.import_errors
    assert DAG_ID in dag_bag.dags
    return dag_bag.dags[DAG_ID]


def test_dag_imports_without_errors(dag_bag):
    assert dag_bag.import_errors == {}


def test_dag_configuration(training_dag):
    assert training_dag.catchup is False
    assert training_dag.schedule is None


def test_dag_has_expected_tasks(training_dag):
    assert set(training_dag.task_ids) == set(EXPECTED_TASKS)


def test_dag_task_order(training_dag):
    observed_edges = {
        (task.task_id, downstream.task_id)
        for task in training_dag.tasks
        for downstream in task.downstream_list
    }
    assert observed_edges == set(pairwise(EXPECTED_TASKS))
