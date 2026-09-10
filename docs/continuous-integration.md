# Continuous Integration com GitHub Actions

## Objetivo

O workflow `.github/workflows/ci.yml` verifica cada mudança antes que ela seja
considerada apta para implantação. O CI não treina o modelo, não acessa o GCS
e não possui credenciais de nuvem.

```text
pull request para main ou push na main
                    ↓
        ┌───────────┴────────────┐
        ↓                        ↓
qualidade e testes        validação do Airflow
        ↓
build da imagem da API
```

## Gatilhos

O CI executa em:

- pull request cujo destino seja `main`;
- push direto ou commit de merge na `main`;
- disparo manual com `workflow_dispatch`.

Pushes em branches de feature não disparam o CI até que exista um pull
request para `main`. A concorrência é separada por workflow e referência; uma
execução antiga da mesma referência é cancelada quando uma nova começa.

O workflow possui apenas a permissão `contents: read` e todas as actions são
referenciadas por SHA imutável.

## Jobs

### Qualidade, pipeline e testes

O job `quality` utiliza Python 3.11 e uv 0.12.6. A instalação abaixo também
confirma que o `uv.lock` está sincronizado com o `pyproject.toml`:

```bash
uv sync --locked --extra api
```

Depois executa:

```bash
uv run --no-sync dvc dag
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pytest
```

`dvc dag` valida a declaração e as dependências do pipeline, mas não acessa o
remote nem reproduz os stages. Os testes do Airflow são ignorados no ambiente
principal e cobertos pelo job isolado descrito abaixo.

### Airflow

O job `airflow` cria `.venv-airflow` somente no runner, usando o grupo
`airflow`, porque Airflow e a API exigem versões incompatíveis do FastAPI.
Ele confirma a instalação, executa `tests/test_dag.py`, valida o Compose e
constrói `Dockerfile.airflow`.

Durante `docker compose config`, o job informa apenas um caminho fictício para
o ADC. Nenhuma credencial é aberta ou montada, e o CI não executa `dvc pull`,
`dvc repro`, retreinamento ou `dvc push`.

### Build da API

Depois do job de qualidade, o job `docker` constrói a imagem de serving. Em
pull requests, essa etapa valida Dockerfile, dependências e código, mas não
inclui o modelo real, pois o CI não possui acesso ao DVC.

O modelo é obtido e seu contrato é testado somente no CD, depois que o commit
já passou pelo CI na `main`.

## Executar as verificações localmente

Ambiente principal:

```bash
uv sync --locked --extra api
uv run --no-sync dvc dag
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pytest
docker build -t medical-triage-api:ci .
```

Ambiente do Airflow:

```bash
UV_PROJECT_ENVIRONMENT=.venv-airflow \
uv sync --locked --no-default-groups --group airflow

UV_PROJECT_ENVIRONMENT=.venv-airflow \
uv run --no-sync pytest tests/test_dag.py

GOOGLE_APPLICATION_CREDENTIALS_HOST=/tmp/adc-placeholder.json \
docker compose -f docker-compose.airflow.yml config --quiet

docker build -f Dockerfile.airflow -t medical-triage-airflow:ci .
```

## O que acontece quando o CI falha

Git e GitHub Actions são mecanismos diferentes. O GitHub recebe o commit
primeiro e só depois executa o workflow. Portanto, uma falha do CI:

- não desfaz nem rejeita automaticamente um push já aceito;
- deixa a execução e o commit com status vermelho;
- impede o CD deste projeto, pois o workflow de deploy exige CI concluído com
  sucesso;
- pode bloquear o merge de um pull request se a branch `main` possuir regra
  de proteção exigindo os checks.

Sem proteção de branch, um push direto na `main` continua na `main` mesmo que
o CI falhe. A correção normal é criar outro commit corrigindo o problema; não
é necessário apagar o commit anterior.

## Relação entre CI, CT e CD

| Fluxo | Onde executa | Responsabilidade |
|---|---|---|
| CI | GitHub Actions | Validar código, testes, DAG e builds |
| CT | Airflow local | Obter dados, treinar, avaliar e publicar artefatos no DVC |
| CD | GitHub Actions + GCP | Empacotar o modelo aprovado e publicar no Cloud Run |

O CI é pré-condição do CD. O CT é independente e só altera o modelo publicado
depois que `dvc.lock`, métricas e código são revisados e integrados na `main`.

## Evidência atual

O CI do merge `1f8c1fd` na `main` foi aprovado e liberou a primeira execução
do workflow de CD. Na validação local mais recente:

```text
ambiente principal: 37 testes aprovados e 7 testes Airflow ignorados
ambiente Airflow:   7 testes aprovados
Ruff lint/format:   aprovado
builds Docker:      aprovados
```
