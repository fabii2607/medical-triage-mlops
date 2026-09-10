# Continuous Deployment no Google Cloud

## Objetivo

O workflow `.github/workflows/cd.yml` publica automaticamente na API do
Cloud Run apenas um commit da `main` aprovado pelo CI. A release contém o
código daquele commit e o modelo apontado pelo DVC no mesmo commit.

```text
merge/push em main
        ↓
CI aprovado
        ↓
dvc pull do modelo aprovado
        ↓
teste do contrato do artefato
        ↓
build + smoke test local
        ↓
Artifact Registry (imagem imutável)
        ↓
revisão candidata do Cloud Run, sem tráfego
        ↓
smoke test da candidata
        ↓
100% do tráfego para a candidata
        ↓
smoke test da URL estável ou rollback
```

## Disparo e vínculo com o CI

O CD possui dois gatilhos:

- conclusão com sucesso do workflow `Continuous Integration` na `main`;
- disparo manual (`workflow_dispatch`) feito a partir da `main`.

Assim, um push em branch de feature não publica uma imagem nem altera o
serviço. O CD faz checkout do SHA exato aprovado pelo CI, e não de uma branch
que possa ter avançado enquanto o job aguardava.

O grupo de concorrência `production-deployment` mantém somente uma implantação
ativa por vez. Uma segunda execução aguarda a anterior, evitando duas
promoções simultâneas.

## Autenticação sem chave

O GitHub Actions usa OpenID Connect e Workload Identity Federation (WIF). Não
existe JSON de service account nos secrets, no Git ou na imagem.

O provider aceita tokens somente quando as claims indicam:

```text
repository == fabii2607/medical-triage-mlops
ref == refs/heads/main
```

A service account de deploy é:

```text
github-cd@medical-triage-mlops.iam.gserviceaccount.com
```

Ela recebeu somente as permissões usadas pelo workflow:

| Recurso | Papel | Uso |
|---|---|---|
| Bucket do DVC | `roles/storage.objectViewer` | Baixar o modelo aprovado |
| Repositório Artifact Registry | `roles/artifactregistry.writer` | Enviar a imagem |
| Serviço Cloud Run | `roles/run.developer` | Criar revisão e alterar tráfego |
| Service account de runtime | `roles/iam.serviceAccountUser` | Vincular a identidade ao container |

O container executa com uma identidade separada e sem permissões do deploy:

```text
cloud-run-runtime@medical-triage-mlops.iam.gserviceaccount.com
```

A API não acessa GCS ou outros serviços Google durante a inferência, portanto
essa conta não precisa de papéis adicionais.

## Recursos de produção

| Configuração | Valor |
|---|---|
| Projeto | `medical-triage-mlops` |
| Região | `southamerica-east1` |
| Repositório Docker | `medical-triage` |
| Imagem | `medical-triage-api` |
| Serviço Cloud Run | `medical-triage-api` |
| Porta | `8000` |
| CPU e memória | `1 vCPU`, `512 MiB` |
| Escala | `0–20` instâncias |
| Concorrência | `80` requisições por instância |
| Timeout | `300` segundos |

Os identificadores acima não são segredos e permanecem declarados no
workflow. O limite de confiança real está no provider WIF e nas políticas IAM.
Na primeira execução do CD, a nova revisão substituirá a conta padrão usada
pelas revisões antigas pela service account de runtime dedicada.

## Artefato e imagem imutáveis

O job instala dependências a partir do `uv.lock` e baixa somente:

```bash
dvc pull models/logreg_tfidf.joblib
```

Em seguida executa o teste do contrato real do modelo. O SHA-256 do `.joblib`
é registrado no resumo da execução.

A imagem recebe como tag o SHA completo do commit:

```text
southamerica-east1-docker.pkg.dev/medical-triage-mlops/medical-triage/medical-triage-api:<git-sha>
```

Essa tag liga uma release a um estado exato do Git. O workflow não usa
`latest` para implantar, evitando que o conteúdo de uma release seja trocado
sem mudar sua referência.

## Promoção e rollback

Antes do deploy, o workflow registra a revisão que recebe 100% do tráfego.
Depois cria uma revisão candidata com `--no-traffic` e uma tag temporária.
Essa URL isolada passa por:

- `GET /health`, exigindo o modelo carregado;
- `POST /predict`, exigindo uma das três classes do contrato.

Somente depois desses testes a candidata recebe 100% do tráfego. A URL estável
passa novamente pelos dois testes. Se essa validação final falhar, o workflow
restaura 100% do tráfego para a revisão registrada no início.

O rollback cobre falhas detectadas pelos smoke tests. Erros semânticos ou
clínicos que não fazem parte desses testes ainda exigem observabilidade e uma
decisão humana.

## Operação

Depois do merge na `main`, acompanhe no GitHub:

1. **Actions → Continuous Integration**;
2. após o CI verde, **Actions → Continuous Deployment**;
3. confira no resumo do job o commit, SHA-256 do modelo, URI da imagem,
   revisão e URL publicada.

Para repetir uma release sem um novo commit, abra `Continuous Deployment`,
selecione **Run workflow** na `main` e execute. A autenticação continua limitada
à `main` pelo WIF.

Se o ambiente `production` possuir regra de aprovação no GitHub, o job pausa
antes da autenticação e aguarda o revisor. Essa proteção é opcional para a
entrega acadêmica, mas recomendada para uma produção real.

## Validação local de 10/09/2026

Antes da publicação da branch foram validados:

- sintaxe do workflow com `actionlint 1.7.12`;
- build da imagem a partir de `Dockerfile` e `uv.lock`;
- execução com o usuário não privilegiado `app`;
- resposta de `/health` com `model_loaded=true`;
- resposta de `/predict` com as três probabilidades e classe válida;
- 37 testes do ambiente principal e 7 testes do ambiente Airflow;
- Ruff lint e format check.

A autenticação WIF, o push da imagem e a promoção automatizada só podem ser
validados dentro do GitHub Actions após o workflow existir na `main`.

## Limites desta entrega

- O Cloud Run permanece público porque a API e o Prometheus já dependem desse
  acesso. Em produção clínica, autenticação e autorização seriam obrigatórias.
- O modelo é um classificador acadêmico treinado com pseudo-rótulos; a
  implantação não transforma suas métricas em validação clínica.
- O CT local produz e publica artefatos no DVC, mas uma pessoa ainda revisa e
  versiona `dvc.lock` e métricas por PR. O CD entrega somente o modelo que já
  está aprovado e referenciado na `main`.

## Referências oficiais

- [Workload Identity Federation para pipelines de deploy](https://docs.cloud.google.com/iam/docs/workload-identity-federation-with-deployment-pipelines)
- [Papéis IAM do Cloud Run](https://docs.cloud.google.com/run/docs/reference/iam/roles)
- [Autenticação Docker no Artifact Registry](https://docs.cloud.google.com/artifact-registry/docs/docker/authentication)
- [Rollout, rollback e migração de tráfego no Cloud Run](https://cloud.google.com/run/docs/rollouts-rollbacks-traffic-migration)
