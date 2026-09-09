# Handoff — CI e validação do Airflow

## Estado atual

O pipeline operacional de treinamento está declarado no `dvc.yaml`:

```text
split → train → evaluate_validation → quality_gate → evaluate_test
```

A DAG `medical_triage_training`, em `dags/training_dag.py`, orquestra esses
stages individualmente com `dvc repro --single-item`. O Airflow não contém a
lógica de ML: ele controla ordem, falhas e logs; o DVC permanece como fonte
única da execução de cada etapa.

## Integração contínua

O workflow `.github/workflows/ci.yml` executa em pull requests para `main`,
pushes na `main` e por disparo manual. Ele possui três verificações:

1. ambiente principal: DVC DAG, Ruff, formatação e pytest;
2. ambiente Airflow: versão instalada, testes estruturais da DAG, validação do
   Docker Compose e build de `Dockerfile.airflow`;
3. build da imagem de serving da API.

O job principal não instala Airflow. Os testes de `tests/test_dag.py` são
ignorados nesse ambiente e executados obrigatoriamente no job Airflow, que usa:

```bash
UV_PROJECT_ENVIRONMENT=.venv-airflow \
uv sync --locked --no-default-groups --group airflow
```

O CI de pull request não recebe credenciais GCP, não executa `dvc pull` e não
retreina o modelo.

## Dependências e imagem do Airflow

O `pyproject.toml` e o `uv.lock` são as únicas fontes de dependências:

- `test`: pytest compartilhado;
- `mlops`: DVC com suporte ao GCS;
- `dev`: inclui `test` e `mlops`;
- `airflow`: inclui `test`, `mlops` e Airflow 3.1.7.

Não existe `requirements-airflow.txt`. A imagem oficial executa o Airflow com
seu Python nativo e cria `/home/airflow/project-venv` a partir do lockfile para
o núcleo do projeto e o grupo `mlops`. Isso evita misturar a versão do FastAPI
da API com a versão exigida internamente pelo Airflow.

## Execução local

Crie o ambiente isolado:

```bash
UV_PROJECT_ENVIRONMENT=.venv-airflow \
uv sync --locked --no-default-groups --group airflow
```

Valide os testes:

```bash
UV_PROJECT_ENVIRONMENT=.venv-airflow \
uv run --no-sync pytest tests/test_dag.py
```

Suba a orquestração:

```bash
docker compose -f docker-compose.airflow.yml up --build
```

A interface fica disponível em `http://localhost:8080`. No estado atual, o
dataset pseudo-rotulado deve estar presente em `data/processed/`, obtido antes
com `dvc pull`.

## Limites atuais e próxima etapa

A DAG executa sob demanda (`schedule=None`) e ainda não contém tasks para:

- obter os dados automaticamente com `dvc pull`;
- publicar os artefatos aprovados com `dvc push`;
- impedir execuções simultâneas com `max_active_runs=1`.

Esses pontos pertencem à próxima evolução do Continuous Training. O deploy
contínuo no Cloud Run também é uma etapa separada: deverá obter o modelo pelo
DVC, publicar uma imagem imutável no Artifact Registry e testar `/health` após
o deploy.

## Critérios desta entrega

- [x] DAG importa sem erros no Airflow 3.1.7.
- [x] ID, configuração, tasks e dependências possuem testes automatizados.
- [x] Ambiente da API permanece sem Airflow.
- [x] Compose do Airflow é validado no CI.
- [x] Imagem do Airflow é construída no CI.
- [ ] Confirmar o novo job verde em pull request para `main`.
- [ ] Registrar uma execução ponta a ponta da DAG.
