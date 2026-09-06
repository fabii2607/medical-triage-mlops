# Handoff — Etapa 1: API de Triagem (FastAPI)

> **[2026-09-06] Etapa entregue.** Documento mantido como registro. Diferenças
> em relação ao combinado: o artefato canônico passou a ser
> `models/logreg_tfidf.joblib` (gerado pela stage `train` do DVC;
> `logreg_tfidf_v2` sobrevive como `model_version` na resposta), e as métricas
> Prometheus adotaram o prefixo `medical_triage_*` (ver handoff_monitoramento).

> Guia para quem vai implementar a API. Este documento é autossuficiente —
> tudo que a etapa precisa está aqui, sem exigir leitura dos notebooks.

## Contexto em 1 minuto

O sistema classifica o texto de um laudo médico (inglês) em 3 níveis de
urgência: `urgente`, `atenção`, `normal`. O modelo que a API vai servir é um
**Pipeline sklearn (TF-IDF + LogisticRegression)** já treinado e validado:
0,3 MB, latência média de **0,46 ms** por predição, recall de 81% na classe
`urgente`. Ele foi destilado de pseudo-rótulos gerados por um BioBERT — mas
isso é história do pipeline offline: **a API não usa BioBERT, torch nem GPU**.
Runtime necessário: `scikit-learn` + `joblib`, nada mais.

## Setup

```bash
git clone https://github.com/fabii2607/medical-triage-mlops.git
cd medical-triage-mlops
uv sync            # instala tudo do lock (CPU-only, suficiente para a API)
uv run pytest      # 12 testes devem passar
```

## Obtendo o modelo

`models/*.joblib` está fora do git (artefato regenerável). Duas opções:

1. **Pedir o arquivo ao time** — `logreg_tfidf_v2.joblib`, 0,3 MB (caminho
   esperado: `models/logreg_tfidf_v2.joblib`).
2. **Regenerar via pipeline** — determinístico (seed 42), mas a etapa de
   pseudo-rotulagem leva ~2 min em GPU ou ~8 h em CPU:

```bash
uv run python -m src.labeling.pseudolabel
uv run python -m src.data.split
uv run python -m src.training.train_mlp --version v2
```

Validação rápida do artefato (deve prever `urgente` com ~0,92):

```bash
uv run python -c "import joblib; m = joblib.load('models/logreg_tfidf_v2.joblib'); \
print(m.predict(['Patient presenting acute myocardial infarction with severe chest pain.']))"
```

## Contrato do modelo

```python
import joblib

model = joblib.load("models/logreg_tfidf_v2.joblib")  # Pipeline completo (TF-IDF embutido)

model.classes_             # array(['atenção', 'normal', 'urgente'])  ← nesta ordem
model.predict([texto])     # array(['urgente'])
model.predict_proba([texto])  # array([[0.070, 0.006, 0.924]]) — soma 1, ordem de classes_
```

- Entrada: texto livre em inglês (abstract/laudo). Sem pré-processamento
  externo — o TF-IDF está dentro do pipeline.
- Carregue o modelo **uma vez no startup** (lifespan do FastAPI), nunca por
  request.
- Nunca hardcode a ordem das classes — use `model.classes_` para montar a
  resposta.
- Métricas de referência do modelo:
  [docs/results/triage_metrics_v2.json](results/triage_metrics_v2.json)
  (chave `logreg_tfidf`).

## Escopo da etapa

1. **API FastAPI** com `POST /predict` e `GET /health` (spec sugerida abaixo)
2. **Dockerfile** — base `python:3.11-slim`, deps via `uv sync --frozen`
   (imagem de serving não precisa de torch/transformers — se a imagem passar
   de ~400 MB, provavelmente o torch entrou; vale separar um grupo de deps)
3. **Benchmark de latência** — script `benchmark.py` medindo média/P95 da API
   completa (baseline para a comparação com ONNX numa etapa futura)
4. **README** — seção de execução da API + análise arquitetural AWS: a
   inferência é **real-time** (triagem na chegada do laudo) → API containerizada
   em ECR + ECS Fargate com CloudWatch; o retreino é **batch** e fica fora da API

Arquivos esperados: `api/main.py`, `api/schemas.py` (Pydantic),
`tests/test_api.py`, `Dockerfile`, `src/benchmark.py` (ou `scripts/`).

## Spec sugerida dos endpoints

### `POST /predict`

```jsonc
// request
{ "text": "Patient presenting acute myocardial infarction..." }

// response 200
{
  "triage_level": "urgente",
  "probabilities": { "atenção": 0.070, "normal": 0.006, "urgente": 0.924 },
  "model_version": "logreg_tfidf_v2"
}
```

- `text` vazio ou só espaços → **422** (validação Pydantic, `min_length` após strip)
- `model_version` na resposta desde já — uma etapa futura vai servir o mesmo
  modelo em ONNX e a comparação precisa ser rastreável

### `GET /health`

```jsonc
{ "status": "ok", "model_loaded": true }
```

Deve retornar 503 se o artefato não carregou — é o probe do container.

## Decisões já tomadas que a API deve respeitar

| Decisão | Implicação para a API |
|---|---|
| `atenção` é regra operacional, não classe clínica | Não descrever como "severidade média" na doc da API; é "incerteza do classificador → revisão humana" |
| Guard-rail da classe `urgente` | Estruture o código para contar predições por classe desde já (um dict/Counter basta) — vira métrica Prometheus na etapa de monitoramento |
| `/metrics` Prometheus vem na etapa de monitoramento | Não precisa implementar agora, mas não ocupe a rota. Quando a integração chegar, os nomes já combinados são: `triage_requests_total{endpoint, http_status}`, `triage_request_latency_seconds{endpoint}` e `triage_predictions_total{triage_level}` (ver [handoff_monitoramento.md](handoff_monitoramento.md)) |
| ONNX Runtime vem na etapa de otimização (spike validado: 0,077 ms, 5,9x mais rápido) | Isole o carregamento/predição numa classe ou módulo (ex.: `api/model.py`) para o backend ser trocável sem tocar nas rotas |
| Idioma do modelo é inglês | Documente no endpoint; não é preciso validar idioma |

## Testes esperados (`tests/test_api.py`)

`httpx` já está no grupo dev — use o `TestClient` do FastAPI:

- `/predict` com texto urgente → 200 e classe válida
- `/predict` com `text` vazio → 422
- `/predict` sem campo `text` → 422
- `/health` → 200 com `model_loaded: true`
- probabilidades somam ~1.0 e têm as 3 chaves

## Checklist de aceite

- [ ] `uv run uvicorn api.main:app` sobe e `/docs` (Swagger) funciona
- [ ] Os testes de API passam junto com os 12 existentes (`uv run pytest`)
- [ ] `docker build` + `docker run -p 8000:8000` → `/predict` responde
- [ ] `benchmark.py` reporta média/P95 (referência local sem HTTP: 0,46 ms)
- [ ] README atualizado (execução + análise AWS)
- [ ] Nenhuma dependência de torch/transformers no caminho da API

## Referências

| Documento | Conteúdo |
|---|---|
| [README.md](../README.md) | Quickstart, contrato do modelo, resultados |
| [triage_metrics_v2.json](results/triage_metrics_v2.json) | Métricas completas do modelo servido |
