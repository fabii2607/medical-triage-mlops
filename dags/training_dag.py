"""DAG de treino/retreino do modelo de triagem médica.

Orquestra os stages do pipeline DVC — a fonte única da lógica é o `dvc.yaml`
na raiz do projeto; cada task executa um stage isolado (`--single-item`),
na mesma ordem do pipeline:

    split -> train -> evaluate_validation -> quality_gate -> evaluate_test

O `quality_gate` interrompe a DAG (exit code != 0) quando as métricas de
validação ficam abaixo dos thresholds do `params.yaml`, impedindo a
avaliação final de um modelo reprovado — mesmo comportamento do `dvc repro`.

Requisitos: o repositório montado em /opt/airflow/project com os dados de
`data/processed/` presentes (via `dvc pull` no host ou execução anterior).
Ver docker-compose.airflow.yml.
"""

from datetime import datetime
from itertools import pairwise

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator

PROJECT_DIR = "/opt/airflow/project"

STAGES = [
    "split",
    "train",
    "evaluate_validation",
    "quality_gate",
    "evaluate_test",
]

with DAG(
    dag_id="medical_triage_training",
    description="Pipeline de treino da triagem: split -> train -> evaluate -> gate",
    schedule=None,  # disparo manual; em produção, agendar (ex.: semanal)
    start_date=datetime(2026, 9, 1),
    catchup=False,
    tags=["mlops", "training", "dvc"],
) as dag:
    tasks = [
        BashOperator(
            task_id=stage,
            bash_command=f"cd {PROJECT_DIR} && dvc repro --single-item {stage}",
        )
        for stage in STAGES
    ]

    for upstream, downstream in pairwise(tasks):
        upstream >> downstream
