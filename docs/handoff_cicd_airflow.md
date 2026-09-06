# Handoff — Etapa 2: CI/CD (GitHub Actions) e DAG Airflow

> **[2026-09-06] Etapa entregue.** Documento mantido como registro. Diferenças
> em relação ao combinado: o pipeline de treino foi implementado como stages
> **DVC** (`dvc.yaml`: split → train → evaluate_validation → quality_gate →
> evaluate_test) e a DAG (`dags/training_dag.py`, via
> `docker-compose.airflow.yml`) orquestra esses stages — não os CLIs soltos
> citados abaixo. `train_mlp.py` foi substituído por `src/training/train.py`.

> Guia autossuficiente para quem vai implementar esta etapa. Pode ser colado
> como contexto no seu copilot — contém o estado do repo, os contratos e os
> critérios de aceite.

## Contexto em 1 minuto

O projeto é um sistema de triagem de laudos médicos (3 classes: `urgente`,
`atenção`, `normal`). O pipeline de ML já existe como CLIs reproduzíveis em
`src/` e os testes já passam. Esta etapa entrega duas coisas **independentes
entre si**:

1. **Workflow GitHub Actions** rodando lint + testes a cada push/PR
2. **DAG Airflow** orquestrando o pipeline de treino (ingestão → treino → salvamento)

Nenhuma das duas depende da API (Etapa 1, em desenvolvimento paralelo).

## Setup

```bash
git clone https://github.com/fabii2607/medical-triage-mlops.git
cd medical-triage-mlops
uv sync
uv run pytest              # 12 testes devem passar
uv run ruff check src tests  # lint limpo
```

## Estado atual que importa para esta etapa

- **Python 3.11**, dependências via `uv` (`pyproject.toml` + `uv.lock`);
  `ruff`, `pytest` e `httpx` já estão no grupo dev.
- **Os testes não dependem de dados nem de modelos** — usam dados sintéticos
  e funções puras. O CI não precisa baixar/gerar nada.
- CLIs do pipeline (todas determinísticas, seed 42):

```bash
uv run python -m src.labeling.pseudolabel [--limit N] [--resume]
uv run python -m src.data.split [--input caminho.csv]
uv run python -m src.training.train_mlp --version v2
```

- `data/raw/` está **no git** (insumo). `data/processed/` e `models/*.joblib`
  estão **fora do git** (regenerados pelo pipeline).
- A pseudo-rotulagem completa (11.227 textos, BioBERT) leva ~1 min em GPU e
  de ~30 min a ~8 h em CPU, dependendo da máquina. Com `--limit N` ela
  processa N textos e salva em `data/processed/pseudolabel_smoke_test.csv`
  (arquivo separado, não sobrescreve o dataset completo).
- O time desenvolve em **Windows** → Airflow obrigatoriamente via Docker.

## Parte A — Workflow GitHub Actions

Arquivo: `.github/workflows/ci.yml`. Requisito mínimo do desafio: **2
automações** (lint + test já cumprem; o `docker build` entra depois que o
Dockerfile da Etapa 1 existir — deixe o job comentado ou condicional).

Estrutura sugerida:

- Trigger: `push` e `pull_request` na `main` (e branches `feat/**` se quiser)
- Job único ou dois jobs: `astral-sh/setup-uv` → `uv sync --frozen` →
  `uv run ruff check src tests` → `uv run pytest`
- Python 3.11; cache do uv habilitado pelo action

Atenção: use `uv sync --frozen` (falha se o lock estiver dessincronizado —
é o comportamento desejado no CI).

## Parte B — DAG Airflow

Arquivo: `dags/training_dag.py`. Requisito do desafio: "DAG funcional
realizando as etapas de ingestão e treino" — simples é suficiente.

**Tasks sugeridas** (espelham as CLIs existentes — a DAG é um wrapper):

```
pseudolabel (ingestão/rotulagem) → split → train
```

**Problema prático e solução já decidida:** o worker do Airflow normalmente
não tem GPU, e a pseudo-rotulagem completa em CPU é lenta. A DAG deve rodar
com **limite parametrizado** (Airflow Variable, ex.: `PSEUDOLABEL_LIMIT=500`,
~2–10 min em CPU) e encadear via `--input`:

```bash
python -m src.labeling.pseudolabel --limit 500
python -m src.data.split --input data/processed/pseudolabel_smoke_test.csv
python -m src.training.train_mlp --version airflow
```

O sufixo `--version airflow` evita sobrescrever os artefatos oficiais (`_v2`).

**Infra sugerida:** Airflow standalone em Docker Compose
(`apache/airflow:2.x`), com:
- imagem estendida (`Dockerfile.airflow`): instala as dependências do projeto
  (gere com `uv export --no-dev > requirements-airflow.txt`; o container usa
  pip — não use uv dentro da imagem do Airflow)
- volume montando o repositório (para `src/`, `data/raw/` e saída em
  `data/processed/`/`models/`)
- `BashOperator` chamando `python -m src...` (o projeto precisa estar no
  `PYTHONPATH` do worker — monte na pasta e exporte `PYTHONPATH=/opt/projeto`)

## Decisões já tomadas que esta etapa deve respeitar

| Decisão | Implicação |
|---|---|
| Commits semânticos (`feat:`, `fix:`, `ci:`, `test:`, `docs:`, `chore:`) | Critério explícito do desafio — manter o padrão |
| Artefatos fora do git | O CI nunca deve commitar CSV/joblib; a DAG grava só no volume local |
| `torch`/`transformers` são pesados (~2 GB no container) | Aceitável na imagem do Airflow (é quem roda a pseudo-rotulagem); jamais na imagem da API |
| Lockfile é a fonte de verdade | Mudou dependência → `uv lock` e commit do `pyproject.toml` + `uv.lock` juntos |

## Checklist de aceite

- [ ] Push em branch dispara o workflow; lint e testes passam no Actions
- [ ] PR para `main` mostra os checks
- [ ] `docker compose up` sobe o Airflow local; a DAG aparece sem erros de import
- [ ] Trigger manual da DAG completa as 3 tasks (com `PSEUDOLABEL_LIMIT` baixo)
- [ ] Artefatos da DAG aparecem em `models/` (sufixo `airflow`), sem tocar nos `_v2`
- [ ] Nada de CSV/joblib novo entra no git

## Referências

| Documento | Conteúdo |
|---|---|
| [README.md](../README.md) | Quickstart, arquitetura, resultados |
| [handoff_api.md](handoff_api.md) | Etapa 1 (paralela) — o Dockerfile dela habilita o job de build no CI |
