# Continuous Training local com Airflow e DVC

Este documento descreve o fluxo de treino e retreino executado localmente.
A validação de código está documentada em
[continuous-integration.md](continuous-integration.md), e a publicação da API
em [continuous-deployment.md](continuous-deployment.md).

## Estado atual

O pipeline operacional de treinamento está declarado no `dvc.yaml`:

```text
split → train → evaluate_validation → quality_gate → evaluate_test
```

A DAG `medical_triage_training`, em `dags/training_dag.py`, envolve esse fluxo
com obtenção e publicação dos artefatos versionados:

```text
fetch_versioned_data
  → split
  → train
  → evaluate_validation
  → quality_gate
  → evaluate_test
  → publish_artifacts
```

O Airflow controla ordem, concorrência, tentativas e logs. A lógica de ML
continua em `src/` e no `dvc.yaml`; ela não foi duplicada na DAG.

## Autenticação local no GCS

Gere o Application Default Credentials (ADC) fora do repositório:

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project medical-triage-mlops
```

Copie o template e ajuste o caminho absoluto para o seu computador:

```bash
cp .env.example .env
```

Exemplo no macOS:

```dotenv
GOOGLE_APPLICATION_CREDENTIALS_HOST=/Users/usuario/.config/gcloud/application_default_credentials.json
```

O `.env` e o JSON real são ignorados pelo Git. O Compose monta o ADC como
somente leitura em um caminho fixo no container e define
`GOOGLE_APPLICATION_CREDENTIALS` para o DVC. Cada integrante utiliza sua
própria identidade; a credencial não entra na imagem nem em volume Docker.

## Infraestrutura local

Inicie o Airflow com:

```bash
docker compose -f docker-compose.airflow.yml up --build -d
```

O Compose cria:

- `airflow-init`: tarefa curta, executada como root apenas para atribuir os
  volumes ao UID 50000;
- `airflow`: serviço principal executado como usuário não privilegiado;
- `airflow_state`: banco, configuração e logs locais do Airflow;
- `airflow_dvc_cache`: cache local reutilizado pelo DVC.

Os volumes sobrevivem a `docker compose down`, mas continuam somente no
Docker local. O armazenamento durável e compartilhado dos dados/modelos é o
bucket GCS. `docker compose down -v` também remove os volumes locais.

Enquanto o `gcsfs 2026.8.0` estiver instalado, o Compose define:

```text
GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT=false
```

Isso evita o discovery experimental de buckets que causou timeout de DNS no
ambiente local; não desabilita o GCS.

## Disparar o CT

Acesse `http://localhost:8080`, abra `medical_triage_training` e use
**Trigger DAG**. O campo `force_retrain` possui dois comportamentos:

- `false` (padrão): o DVC executa apenas stages invalidados;
- `true`: adiciona `--force` e reproduz todos os stages.

Também é possível usar a CLI:

```bash
docker compose -f docker-compose.airflow.yml exec airflow \
  airflow dags trigger --conf '{"force_retrain": true}' \
  medical_triage_training
```

`schedule=None` mantém o disparo manual e `max_active_runs=1` impede duas
execuções concorrentes de escreverem nos mesmos outputs. Uma segunda execução
fica na fila até a primeira terminar.

As tarefas que acessam o GCS (`fetch_versioned_data` e
`publish_artifacts`) possuem uma nova tentativa após dois minutos. As tarefas
locais e o quality gate não repetem automaticamente, pois falhas determinísticas
de dados, código ou métrica precisam ser corrigidas.

## Versionamento e promoção

`fetch_versioned_data` executa `dvc pull` no ponteiro do dataset presente no
commit atual. Cada stage usa `dvc repro --single-item`; a ordem dos upstreams
é garantida pela DAG. Se o gate reprovar, as tarefas seguintes ficam bloqueadas
e nenhum artefato é publicado.

Depois de uma execução aprovada, `publish_artifacts` executa `dvc push`. Esse
comando envia o cache, mas não cria commit Git. O responsável deve revisar
`dvc.lock`, métricas e demais alterações, criar uma branch/commit e promover o
candidato por pull request. Assim o modelo anterior permanece recuperável pelo
commit anterior.

## Evidência local de 09/09/2026

Execução forçada:

```text
run_id: manual__2026-09-10T00:39:00.783757+00:00
estado: success
duração: aproximadamente 32 s
tasks: 7/7 success
```

Resultados do gate:

```text
validation Macro F1: 0.7203
validation recall urgente: 0.7389
quality gate: approved
dvc push: Everything is up to date.
modelo SHA-256: 2586582f672f0535f624892decbc1212a5153e493f4350c96e81611df154a73e
```

Uma segunda execução, sem `force_retrain`, terminou com sucesso em cerca de
nove segundos. Os logs registraram `Data and pipelines are up to date`,
confirmando a idempotência do DVC.

## Integração com o CD

Este fluxo conclui o CT local acadêmico. Depois da revisão de `dvc.lock`, das
métricas e do modelo por pull request, o CD obtém o artefato aprovado na
`main`, constrói uma imagem imutável, publica no Artifact Registry e promove
uma revisão testada no Cloud Run.

O CD usa Workload Identity Federation e uma service account de privilégio
mínimo, nunca o ADC pessoal utilizado no Airflow local. A promoção e o rollback
estão detalhados em
[continuous-deployment.md](continuous-deployment.md).
