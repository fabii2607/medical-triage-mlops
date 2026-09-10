"""DAG de treino/retreino do modelo de triagem médica.

Orquestra os stages do pipeline DVC — a fonte única da lógica é o `dvc.yaml`
na raiz do projeto; cada task executa um stage isolado (`--single-item`),
na mesma ordem do pipeline:

    fetch_versioned_data
        -> split -> train -> evaluate_validation -> quality_gate -> evaluate_test
        -> publish_artifacts

O `quality_gate` interrompe a DAG (exit code != 0) quando as métricas de
validação ficam abaixo dos thresholds do `params.yaml`, impedindo a
avaliação final de um modelo reprovado — mesmo comportamento do `dvc repro`.

O dataset de entrada é obtido do GCS pela primeira task. Depois da aprovação,
os outputs versionados são enviados ao remote pela última task. Credenciais
não ficam na DAG: são injetadas no container pelo Docker Compose.
"""

from datetime import datetime, timedelta
from itertools import pairwise

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, Param

PROJECT_DIR = "/opt/airflow/project"

STAGES = [
    "split",
    "train",
    "evaluate_validation",
    "quality_gate",
    "evaluate_test",
]

INPUT_POINTER = "data/processed/medical_abstracts_triage_pseudolabeled.csv.dvc"
FORCE_FLAG = "{% if params.force_retrain %}--force {% endif %}"
NETWORK_RETRY_DELAY = timedelta(minutes=2)

with DAG(
    dag_id="medical_triage_training",
    description="Pipeline de treino da triagem: split -> train -> evaluate -> gate",
    schedule=None,  # disparo manual; em produção, agendar (ex.: semanal)
    start_date=datetime(2026, 9, 1),
    catchup=False,
    max_active_runs=1,
    params={
        "force_retrain": Param(
            False,
            type="boolean",
            description="Força a reprodução dos stages mesmo sem mudanças.",
        )
    },
    tags=["mlops", "training", "dvc"],
) as dag:
    fetch_versioned_data = BashOperator(
        task_id="fetch_versioned_data",
        bash_command=f"cd {PROJECT_DIR} && dvc pull {INPUT_POINTER}",
        retries=1,
        retry_delay=NETWORK_RETRY_DELAY,
    )

    stage_tasks = [
        BashOperator(
            task_id=stage,
            bash_command=(
                f"cd {PROJECT_DIR} && dvc repro --single-item {FORCE_FLAG}{stage}"
            ),
            retries=0,
        )
        for stage in STAGES
    ]

    publish_artifacts = BashOperator(
        task_id="publish_artifacts",
        bash_command=f"cd {PROJECT_DIR} && dvc push",
        retries=1,
        retry_delay=NETWORK_RETRY_DELAY,
    )

    tasks = [fetch_versioned_data, *stage_tasks, publish_artifacts]
    for upstream, downstream in pairwise(tasks):
        upstream >> downstream
