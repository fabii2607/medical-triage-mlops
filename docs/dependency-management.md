# Gerenciamento de dependências e ambientes

Este documento registra as decisões adotadas para separar as dependências da API, do pseudo-labeling, do desenvolvimento e do Airflow sem criar vários projetos Python ou vários lockfiles.

## Objetivos

- Manter `pyproject.toml` como fonte das dependências.
- Manter um único `uv.lock`, reproduzível entre sistemas operacionais e arquiteturas suportadas.
- Não instalar ferramentas de desenvolvimento ou modelos pesados na imagem da API.
- Isolar dependências incompatíveis da API e do Airflow.
- Usar os mesmos conjuntos de dependências localmente, no CI e no Docker.

## Versões padronizadas

O projeto utiliza Python 3.11:

```toml
requires-python = ">=3.11,<3.12"
```

O uv aceito pelo projeto deve pertencer à série 0.12:

```toml
[tool.uv]
required-version = ">=0.12.6,<0.13"
```

O CI e o Docker utilizam explicitamente uv 0.12.6. A faixa permite que os integrantes utilizem correções posteriores da série 0.12 sem aceitar uma versão antiga ou uma nova série potencialmente incompatível.

O uv não se atualiza automaticamente quando a versão instalada não atende à faixa. Nesse caso, o comando falha e o desenvolvedor deve atualizar a ferramenta antes de continuar.

## Classificação das dependências

### Núcleo compartilhado

As dependências em `[project].dependencies` são necessárias pelo pipeline leve de treinamento, avaliação ou inferência:

- joblib;
- NumPy;
- pandas;
- Pydantic;
- PyYAML;
- scikit-learn.

Elas são instaladas em todos os ambientes do projeto.

### Extras

Extras representam funcionalidades opcionais que podem fazer parte da execução da aplicação.

O extra `api` contém o serving HTTP:

```bash
uv sync --locked --extra api
```

O extra `labeling` contém Torch e Transformers, usados no pseudo-labeling com BioBERT:

```bash
uv sync --locked --extra labeling
```

O modelo leve servido pela API não precisa do extra `labeling`.

### Grupos

Grupos representam ferramentas e ambientes internos do repositório:

- `test`: pytest compartilhado pelos ambientes que executam testes;
- `mlops`: DVC e integração com GCS;
- `dev`: lint, notebooks e experimentos, incluindo `test` e `mlops`;
- `airflow`: Airflow local, incluindo `test` e `mlops`.

O grupo `dev` é instalado por padrão. Os grupos pequenos `test` e `mlops`
evitam duplicar pytest e DVC nos ambientes que os consomem. Pytest não fica em
`mlops` porque testes e versionamento de artefatos têm responsabilidades
diferentes.

## Conflito entre API e Airflow

A API utiliza FastAPI 0.141.1 ou superior. O Airflow 3.1.7 exige FastAPI abaixo de 0.118. Uma única virtualenv não pode conter as duas versões simultaneamente.

O conflito é declarado no `pyproject.toml`:

```toml
conflicts = [
    [
        { extra = "api" },
        { group = "airflow" },
    ],
]
```

O `uv.lock` registra as duas resoluções, mas cada ambiente instala apenas uma delas.

## Ambientes locais

### Ambiente principal

A `.venv` é o ambiente padrão para API, testes, notebooks e desenvolvimento:

```bash
uv sync --locked --extra api --extra labeling
```

Quando o BioBERT não for necessário, o extra pode ser omitido:

```bash
uv sync --locked --extra api
```

### Ambiente Airflow

O Airflow utiliza uma virtualenv própria porque sua versão do FastAPI é incompatível com a API:

```bash
UV_PROJECT_ENVIRONMENT=.venv-airflow \
uv sync --locked --no-default-groups --group airflow
```

Para executar um comando nesse ambiente:

```bash
UV_PROJECT_ENVIRONMENT=.venv-airflow \
uv run --no-sync airflow version
```

A variável `UV_PROJECT_ENVIRONMENT` altera a pasta utilizada pelo uv somente naquele comando. Sem ela, o uv utiliza `.venv`.

As pastas `.venv` e `.venv-airflow` são locais e não são versionadas.

## Kernels dos notebooks no VS Code

Para simplificar o trabalho do time, todos os notebooks podem inicialmente utilizar o kernel da `.venv` principal, criada com o extra `labeling`.

No VS Code:

1. Abra o notebook.
2. Clique em **Select Kernel**.
3. Selecione **Python Environments**.
4. Selecione `.venv/bin/python` no macOS/Linux ou `.venv\Scripts\python.exe` no Windows.

O notebook pode versionar o nome lógico do kernel, mas não compartilha a virtualenv nem os pacotes instalados. Cada integrante executa `uv sync` e seleciona o interpretador local.

Caso o custo de armazenamento do BioBERT se torne um problema, poderão ser criados posteriormente kernels distintos para notebooks comuns e notebooks de labeling.

## Docker da API

O Dockerfile instala as dependências a partir do `pyproject.toml` e do `uv.lock`:

```bash
uv sync \
    --locked \
    --no-default-groups \
    --extra api \
    --no-install-project \
    --no-cache
```

- `--locked` impede a alteração do lockfile durante o build.
- `--no-default-groups` não instala o grupo `dev`.
- `--extra api` instala somente a funcionalidade de serving além do núcleo.
- `--no-install-project` instala as dependências sem empacotar o código local.
- `--no-cache` evita armazenar o cache de downloads na imagem.

O código e o modelo são copiados depois da instalação das dependências para aproveitar o cache de camadas do Docker.

A API executa como o usuário não privilegiado `app`, UID e GID 10001. Código, modelo e virtualenv permanecem pertencentes ao root e são apenas lidos pelo processo.

O `.dockerignore` adota uma lista de permissão: todo o contexto é ignorado e somente Dockerfile, `pyproject.toml`, `uv.lock`, `api/` e `models/` são enviados ao builder.

## Docker do Airflow

O `Dockerfile.airflow` mantém o Python da imagem oficial responsável pelo
Airflow. Separadamente, cria `/home/airflow/project-venv` com o núcleo do
projeto e o grupo `mlops`:

```bash
uv sync \
    --locked \
    --no-default-groups \
    --group mlops \
    --no-install-project \
    --no-cache
```

Essa separação impede que o FastAPI da aplicação sobrescreva dependências
internas do Airflow. O `pyproject.toml` e o `uv.lock` são a única fonte das
dependências; não existe um `requirements-airflow.txt` paralelo para manter.

## Integração contínua

O CI instala o ambiente necessário aos testes atuais:

```bash
uv sync --locked --extra api
```

O `--locked` valida que o lockfile está atualizado e instala as dependências. Por isso não existe um passo separado com `uv lock --check`.

O extra `labeling` não é instalado no job atual porque os testes não importam Torch ou Transformers. Quando existirem testes específicos do BioBERT, eles devem ficar em um job separado para não aumentar o tempo de todos os testes.

O Airflow possui um job separado no CI. Ele cria `.venv-airflow` apenas no
runner, confirma a instalação do Airflow e executa `tests/test_dag.py`. No
ambiente principal, esses testes são ignorados porque Airflow não faz parte da
API nem do desenvolvimento comum.

O mesmo job valida a configuração de `docker-compose.airflow.yml` e constrói
`Dockerfile.airflow`. Ele não acessa o GCS, não executa retreinamento e não
recebe credenciais.

## Como adicionar uma dependência

Antes de adicionar uma biblioteca, identifique onde ela é executada:

| Uso | Destino |
|---|---|
| Treinamento leve, avaliação e inferência | `[project].dependencies` |
| API em produção | extra `api` |
| BioBERT e pseudo-labeling | extra `labeling` |
| Pytest compartilhado | grupo `test` |
| Lint, notebooks e experimentos | grupo `dev` |
| Versionamento de dados e modelos | grupo `mlops` |
| Orquestração local | grupo `airflow` |

Exemplos:

```bash
uv add --optional api pacote
uv add --optional labeling pacote
uv add --group dev pacote
uv add --group airflow pacote
```

Após qualquer alteração:

```bash
uv lock
uv sync --locked --extra api --extra labeling
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pytest
```

O `pyproject.toml` deve ser resolvido manualmente em conflitos de Git. O `uv.lock` não deve ser combinado linha por linha: depois de resolver o `pyproject.toml`, ele deve ser regenerado com `uv lock`.
