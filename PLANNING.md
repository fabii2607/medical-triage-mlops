# Planejamento — Tech Challenge Fase 3 (MLET)

> **Tema:** Deploy de Modelo em Produção com Pipeline CI/CD, Monitoramento e Otimização de Latência.
> Este documento é o contexto-mestre do projeto. Cada etapa será executada com base nele.
> **Revisão (2026-08-19):** planejamento adaptado à linha adotada na `main` — modelo
> pré-treinado (BioBERT) gera pseudo-rótulos de urgência, e um modelo leve próprio é
> treinado sobre eles para servir em produção.
>
> **Revisão (2026-09-06) — sprint de fechamento:** as etapas 1–3 foram entregues pelo
> time (API + Docker + Cloud Run, CI, DVC com remote GCS, Prometheus/Grafana) e o
> sprint final completou: correção do nome canônico do artefato
> (`models/logreg_tfidf.joblib`), DAG Airflow sobre os stages DVC
> (`dags/training_dag.py` + `docker-compose.airflow.yml`), exportação ONNX com
> benchmark (7,4x; `src/export_onnx.py`), job local no Prometheus, provisionamento do
> Grafana e job de build no CI. Divergências deste documento em relação ao
> implementado: a nuvem escolhida foi **GCP** (Cloud Run + GCS + Artifact Registry),
> não AWS; o versionamento de dados adotou **DVC**; as métricas usam o prefixo
> `medical_triage_*`. Pendente apenas o vídeo STAR. Estado detalhado no README.

## 1. Visão geral do problema

Um hospital precisa de triagem automática de laudos médicos em texto. O sistema
classifica o texto de um laudo/abstract em **3 níveis de urgência**, priorizando a
fila de atendimento:

| Nível     | Significado                                      |
|-----------|--------------------------------------------------|
| `urgente` | Alta probabilidade de urgência — prioridade máxima |
| `atenção` | Zona de incerteza — revisão humana prioritária    |
| `normal`  | Baixa urgência — fluxo padrão                     |

**Decisão de target:** o dataset original (Medical Abstracts TC Corpus) tem categorias
de doenças, não urgência. Usamos o modelo pré-treinado
`Yuvrajxms09/biobert-triage-classifier` (binário: urgent / non-urgent) para gerar
**pseudo-rótulos**, e uma regra de threshold converte a saída binária em 3 níveis.
Um modelo leve próprio (TF-IDF + classificador sklearn) é treinado sobre os
pseudo-rótulos — é ele que vai para produção (leve, rápido, exportável para ONNX).

**Ressalva metodológica registrada:** o notebook 02 alerta que *confiança ≠ severidade* —
a classe `atenção` não foi aprendida pelo BioBERT; é uma **regra operacional** sobre a
zona de incerteza do classificador binário. A equipe decidiu seguir com essa
simplificação, documentando-a como limitação conhecida (justificativa: entrega um
mecanismo de priorização de revisão humana, mesmo sem semântica clínica de severidade).

## 2. Dados e pipeline de pseudo-rotulagem

- `data/raw/medical_tc_train.csv` (11.550) + `medical_tc_test.csv` (2.888) →
  **11.227 abstracts únicos** após consolidação (o dataset original tem textos
  repetidos entre train/test e multi-label achatado).
- Pipeline: `abstract → BioBERT (GPU) → urgent_score/nonurgent_score → regra 0.70 →
  normal/atenção/urgente` → split estratificado 70/15/15 → treino do modelo leve.
- Rastreabilidade: o CSV pseudo-rotulado guarda scores, modelo, threshold,
  `max_length`, truncamento e as categorias médicas originais. Os scores **não**
  entram como features (X = `medical_abstract`, y = `triage_level`).

### Correções aplicadas sobre a execução original (Fabi, notebooks 01–06)

| Problema identificado | Correção (em `src/`) |
|---|---|
| Só 720/11.227 abstracts rotulados (limite de 30 min em CPU) | Rodada completa em GPU (RTX 5080, torch cu128) |
| Os 720 eram os **primeiros em ordem alfabética** (groupby ordena) — viés de seleção | Shuffle com seed antes da inferência + corpus completo |
| `max_length=256` truncava 60,3% dos textos | `max_length=512` (limite do BERT) → ~3% truncados |
| Checkpoint sem retomada e com reparse quadrático | `--resume` + parse incremental |
| Teste final com só 16 exemplos de `urgente` | Resolvido pelo volume: corpus completo |
| Recall de `urgente` = 0.000 no MLP baseline | LogisticRegression `class_weight='balanced'` como candidato (MLP não suporta class_weight) |

## 3. Decisões técnicas (registro de decisão)

| Decisão | Escolha | Por quê |
|---|---|---|
| Pseudo-rotulagem | `Yuvrajxms09/biobert-triage-classifier` + threshold 0.70 | Único modelo público de triagem em texto médico avaliado no piloto (nb 02/03); threshold define a zona `atenção` |
| Modelo de produção | TF-IDF + MLP (baseline v1) vs TF-IDF + LogReg balanced (v2) — decidir pelas métricas | BioBERT é pesado demais para servir com baixa latência; modelo leve destila os pseudo-rótulos |
| API | FastAPI + Uvicorn | Requisito do PDF; async, tipagem via Pydantic |
| Guard-rail da API | Monitorar/alertar se a classe `urgente` nunca for emitida | Uma triagem que estruturalmente não emite a classe crítica é falha de negócio, não só de métrica |
| Serving | Docker (imagem `python:3.11-slim`) | Requisito; reprodutibilidade |
| Otimização de latência | Conversão do pipeline sklearn para **ONNX Runtime** | Técnica citada no PDF; comparação original vs otimizado mensurável |
| CI/CD | GitHub Actions: `ruff` (lint) → `pytest` (test) → `docker build` | Cobre as "2 automações" mínimas com folga; `ruff`/`pytest`/`httpx` já estão no grupo dev |
| Dependências | **uv** com `pyproject.toml` + `uv.lock` versionado | Reprodutibilidade; instalação rápida no CI/Docker |
| GPU local (Erick) | `uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu128` + `uv run --no-sync` | O lock trava torch CPU (portável p/ o time); a build cu128 é só local para a pseudo-rotulagem. `uv sync`/`uv run` sem `--no-sync` reverte para CPU |
| Orquestração | Airflow **standalone** em Docker Compose | DAG: pseudo-rotulagem → split → treino → avaliação → exportação ONNX |
| Monitoramento | prometheus-client + Prometheus + Grafana (Docker Compose) | Requisito; dashboard com ≥3 painéis (incluir distribuição de predições por classe — cobre o guard-rail) |
| Nuvem (análise textual) | **AWS** — ECR + ECS Fargate (real-time), CloudWatch; retreino batch via Airflow (MWAA) + S3; pseudo-rotulagem batch em instância GPU (g5) efêmera | Inferência é real-time; pseudo-rotulagem/retreino são batch |

## 4. Arquitetura

```
  OFFLINE (batch, GPU)                          ONLINE (real-time, CPU)
  ─────────────────────                         ───────────────────────
  medical abstracts (raw)
        ↓
  src/labeling/pseudolabel.py                   ┌──────────────────────────────┐
  (BioBERT + regra 0.70)          laudo ───────▶│  FastAPI  /predict           │
        ↓                                       │  ├─ pipeline sklearn         │
  data/processed/…pseudolabeled.csv             │  └─ ONNX Runtime (padrão)    │
        ↓                                       │  /metrics (prometheus)      │
  src/data/split.py (70/15/15)                  └──────────┬───────────────────┘
        ↓                                                  │ scrape
  src/training/train_mlp.py                        ┌───────▼────────┐   ┌─────────┐
        ↓                                          │   Prometheus   │──▶│ Grafana │
  models/*.joblib + docs/results/*.json            └────────────────┘   └─────────┘

  Airflow (standalone) ── DAG: pseudolabel → split → train → evaluate → export_onnx
```

## 5. Estrutura de pastas

```
medical-triage-mlops/
├── data/raw/                    # CSVs originais
├── data/processed/              # pseudo-rótulos + splits (versionados por ora)
├── notebooks/                   # estudo (Fabi): 01_eda … 06_mlp_training
├── models/                      # artefatos (joblib, onnx) — versionados com sufixo v1/v2
├── docs/results/                # metrics.json por versão de modelo
├── src/
│   ├── config.py                # caminhos e constantes do pipeline
│   ├── labeling/
│   │   ├── biobert.py           # carga + inferência do BioBERT
│   │   ├── triage_rules.py      # regra binário → 3 níveis
│   │   └── pseudolabel.py       # CLI: pseudo-rotulagem completa c/ checkpoint
│   ├── data/split.py            # CLI: split estratificado 70/15/15
│   ├── training/train_mlp.py    # CLI: treino MLP + LogReg, métricas versionadas
│   └── evaluation/metrics.py    # métricas + latência
├── api/                         # (Etapa 1) FastAPI: /predict, /health, /metrics
├── dags/                        # (Etapa 2) DAG Airflow
├── tests/                       # (Etapa 2) pytest
├── monitoring/                  # (Etapa 3) prometheus.yml + grafana/
├── .github/workflows/ci.yml     # (Etapa 2)
├── Dockerfile / docker-compose.yml
├── pyproject.toml / uv.lock
├── PLANNING.md
└── README.md
```

## 6. Etapas de execução

### Etapa 0 — Estudo em notebooks ✅ (Fabi)
Notebooks 01–06: EDA, piloto BioBERT, regra de triagem, pseudo-rotulagem parcial,
split e MLP baseline v1 (accuracy 0.565, macro F1 0.406, recall urgente 0.000 —
sobre dados enviesados; ver correções na seção 2).
Resumo em `docs/resumo_notebooks_refatoracao.md`.

### Etapa 0.5 — Refatoração para src/ + rodada completa ✅
1. ✅ Lógica dos notebooks 03–06 migrada para `src/` (labeling, data, training, evaluation).
2. ✅ Pseudo-rotulagem dos 11.227 abstracts em GPU (1 min; `max_length=512` → só 3% truncados;
   distribuição: normal 5.000 / atenção 4.873 / urgente 1.354).
3. ✅ Split 70/15/15 (7.858/1.684/1.685) e retreino — resultados no test set:

   | Modelo | Accuracy | Bal. Acc | Macro F1 | Recall urgente | Tamanho | Latência média |
   |---|---|---|---|---|---|---|
   | MLP v1 (Fabi, 720 amostras) | 0.565 | 0.442 | 0.406 | **0.000** | 7.53 MB | 4.32 ms |
   | MLP v2 (corpus completo) | 0.776 | 0.731 | 0.748 | 0.591 | 7.53 MB | 0.59 ms |
   | **LogReg balanced v2** | 0.767 | **0.777** | 0.748 | **0.808** | **0.30 MB** | **0.46 ms** |

4. ✅ **Modelo escolhido para a API: `logreg_tfidf_v2`** — macro F1 empatado com o MLP,
   mas recall de `urgente` 0.81 (critério de corte), 25x menor e mais rápido.
- **Entregável:** `data/processed/medical_abstracts_triage_pseudolabeled.csv` (11.227) +
  `docs/results/triage_metrics_v2.json` + `models/logreg_tfidf_v2.joblib`.

### Etapa 1 — Decisão arquitetural e API inicial
1. API FastAPI: `POST /predict` (texto → nível + scores), `GET /health`.
2. Dockerfile + medição de latência baseline (`benchmark.py`).
3. README: análise textual AWS (real-time vs batch) + instruções de execução.
- **Entregável:** API em Docker funcionando + decisão arquitetural no README.

### Etapa 2 — CI/CD e pipeline automatizado
1. Workflow GitHub Actions em push/PR: ruff → pytest → docker build.
2. Testes: unidade (regra de triagem, split, treino em amostra) e API (TestClient/httpx).
3. DAG Airflow: `pseudolabel → split → train → evaluate → export_onnx`
   (pseudolabel com `--limit` no ambiente local sem GPU).
- **Entregável:** `ci.yml` + `training_dag.py` rodando no Airflow local.

### Etapa 3 — Monitoramento e observabilidade
1. Instrumentar API: contador de requisições, histograma de latência, contador de erros,
   **contador de predições por classe** (guard-rail do `urgente`).
2. `docker-compose.yml` com api + prometheus + grafana provisionado.
3. Dashboard Grafana ≥3 painéis: req/s, latência p50/p95/p99, taxa de erro, distribuição por classe.
4. Script gerador de carga.
- **Entregável:** stack completa via compose + dashboard JSON versionado.

### Etapa 4 — Otimização de latência e entrega
1. Exportar pipeline para ONNX (`skl2onnx`) e servir com ONNX Runtime.
2. Benchmark: latência média/p95 sklearn vs ONNX (tabela no README).
3. Roteiro do vídeo STAR (≤5 min).
- **Entregável:** modelo otimizado + tabela comparativa + roteiro do vídeo.

## 7. Mapeamento etapas × critérios de avaliação

| Critério (peso) | Onde é atendido |
|---|---|
| Modelagem e Otimização (20%) | Etapas 0.5 e 4 — pipeline de pseudo-rotulagem + modelo próprio + ONNX |
| CI/CD (15%) | Etapa 2 — workflow lint/test/build |
| Orquestração Airflow (15%) | Etapa 2 — DAG pseudolabel→treino→export |
| Monitoramento (20%) | Etapa 3 — compose + dashboard ≥3 painéis |
| Documentação (15%) | Etapa 1 (arquitetura AWS) + README + registro de decisões neste arquivo |
| Vídeo STAR (15%) | Etapa 4 — roteiro e demonstração |

## 8. Convenções

- **Commits semânticos:** `feat:`, `fix:`, `test:`, `ci:`, `docs:`, `chore:`.
- **Python 3.11**, dependências com `uv`; `uv.lock` sempre commitado com o `pyproject.toml`.
- Artefatos versionados com sufixo (`_v1`, `_v2`) — nunca sobrescrever resultados de
  baseline anteriores (`models/mlp_tfidf.joblib` e `docs/results/mlp_metrics.json`
  são o baseline v1 da Fabi; manter intactos).
- `data/processed/` e `models/` estão versionados no git por simplicidade de entrega;
  reavaliar (DVC/S3) se o repo pesar no CI.

## 9. Riscos e mitigações

- **Qualidade dos pseudo-rótulos:** o BioBERT publica só `LABEL_0/LABEL_1`; o mapeamento
  `LABEL_1 → urgent` vem do model card (validado com sanity check no nb 02). Limitação
  documentada; a regra `atenção` é operacional, não clínica.
- **Airflow no Windows:** sempre via Docker.
- **GPU só na máquina do Erick:** pseudo-rotulagem é batch e roda uma vez; o restante
  do pipeline (split, treino, API) roda em CPU em qualquer máquina. `--limit` e
  `--resume` permitem rodadas parciais em CPU.
- **Desbalanceamento** (`urgente` é minoria): `class_weight='balanced'` no LogReg,
  macro F1 como métrica principal, recall de `urgente` como critério de corte.
- **Tamanho da imagem de serving:** `python:3.11-slim` + ONNX Runtime, sem torch.
