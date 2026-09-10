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
    "fetch_versioned_data",
    "split",
    "train",
    "evaluate_validation",
    "quality_gate",
    "evaluate_test",
    "publish_artifacts",
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
    assert training_dag.max_active_runs == 1
    assert training_dag.params["force_retrain"] is False


def test_dag_has_expected_tasks(training_dag):
    assert set(training_dag.task_ids) == set(EXPECTED_TASKS)


def test_dag_task_order(training_dag):
    observed_edges = {
        (task.task_id, downstream.task_id)
        for task in training_dag.tasks
        for downstream in task.downstream_list
    }
    assert observed_edges == set(pairwise(EXPECTED_TASKS))


def test_remote_tasks_have_one_retry(training_dag):
    for task_id in ("fetch_versioned_data", "publish_artifacts"):
        task = training_dag.get_task(task_id)
        assert task.retries == 1
        assert task.retry_delay.total_seconds() == 120


def test_local_pipeline_tasks_do_not_retry(training_dag):
    for task_id in EXPECTED_TASKS[1:-1]:
        assert training_dag.get_task(task_id).retries == 0


def test_dvc_commands(training_dag):
    fetch_command = training_dag.get_task("fetch_versioned_data").bash_command
    publish_command = training_dag.get_task("publish_artifacts").bash_command

    assert "dvc pull" in fetch_command
    assert "medical_abstracts_triage_pseudolabeled.csv.dvc" in fetch_command
    assert publish_command.endswith("dvc push")

    for task_id in EXPECTED_TASKS[1:-1]:
        command = training_dag.get_task(task_id).bash_command
        assert (
            f"--single-item {{% if params.force_retrain %}}--force {{% endif %}}{task_id}"
            in command
        )
