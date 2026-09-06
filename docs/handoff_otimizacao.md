# Handoff — Etapa 4: Otimização de Latência (ONNX) e Entrega

> **[2026-09-06] Otimização entregue** (`src/export_onnx.py` +
> `src/experiments/benchmark_onnx.py`): speedup de **7,4x** e divergência de
> 1,9% confirmada como casos limítrofes (margem média 0,019) — resultados em
> `docs/results/onnx_benchmark.json` e seção no README. Pendente apenas o
> vídeo STAR. Documento mantido como registro.

> Guia autossuficiente para quem vai implementar esta etapa. Pode ser colado
> como contexto no seu copilot — contém o estado do repo, os resultados do
> spike já validado e os critérios de aceite.

## Contexto em 1 minuto

O modelo de produção já está treinado: um Pipeline sklearn
(TF-IDF + LogisticRegression) que classifica laudos médicos em `urgente`,
`atenção`, `normal` — 0,3 MB, latência média 0,46 ms. Esta etapa entrega a
**otimização de latência exigida pelo desafio**: exportar para **ONNX
Runtime**, comparar a latência original vs otimizada e documentar. Também
fecha o projeto: roteiro e gravação do **vídeo STAR (≤5 min)**.

A tarefa "treinar o classificador de texto" desta etapa **já está concluída**
(pipeline em `src/training/train_mlp.py`). O trabalho aqui é otimização,
comparação e consolidação.

## Setup e obtenção do modelo

```bash
git clone https://github.com/fabii2607/medical-triage-mlops.git
cd medical-triage-mlops
uv sync    # skl2onnx, onnx e onnxruntime JÁ estão nas dependências
```

O modelo `models/logreg_tfidf_v2.joblib` está fora do git — peça o arquivo ao
time (0,3 MB) ou regenere (precisa dos dados: `python -m src.labeling.pseudolabel`
[~1 min GPU / ~30 min+ CPU] → `python -m src.data.split` →
`python -m src.training.train_mlp --version v2`). Para o benchmark, use
`data/processed/splits/test.csv` (vem junto na regeneração ou no zip do time).

## Spike já validado — ponto de partida

A conversão **funciona**; este código foi executado com sucesso:

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

onnx_model = convert_sklearn(
    model,                                                # Pipeline completo (TF-IDF embutido)
    initial_types=[("medical_abstract", StringTensorType([None, 1]))],
    options={"zipmap": False},
)
# inferência: onnxruntime.InferenceSession + entrada np.array([[texto]], dtype=object)
```

Resultados medidos (test set, 1.685 textos, predição unitária):

| | sklearn | ONNX Runtime |
|---|---|---|
| Latência média | 0,453 ms | **0,077 ms** (5,9x) |
| Latência P95 | 0,691 ms | **0,122 ms** |
| Tamanho | 0,3 MB | 208 KB |
| Concordância de predições | — | **98,16%** (31/1.685 divergem) |

## O trabalho da etapa

1. **`src/export_onnx.py`** — CLI que carrega o joblib, converte e salva
   `models/logreg_tfidf_v2.onnx` (mesma convenção de sufixo).
2. **Investigar os 1,84% de divergência** — comportamento conhecido do
   conversor de `TfidfVectorizer` (tokenização/float32). Roteiro:
   compare `predict_proba` sklearn vs ONNX nos 31 casos divergentes; a
   hipótese é que são probabilidades limítrofes (classes quase empatadas).
   Se confirmado, documente como aceitável com os números; se houver
   divergência grosseira, reporte ao time antes de seguir.
3. **`src/benchmark.py`** — CLI que mede média/P95/max dos dois backends no
   mesmo conjunto (use `data/processed/splits/test.csv`) e imprime a tabela
   comparativa. Salve o resultado em `docs/results/latency_benchmark.json`.
4. **Tabela comparativa no README** — seção "Otimização de latência" com a
   tabela e uma frase de método (predição unitária, CPU, N amostras).
5. **(Opcional, combinar com o dev da API)** — a API foi orientada a isolar o
   backend de predição em um módulo trocável; plugar o ONNX lá rende a
   demonstração completa no vídeo.
6. **Vídeo STAR (≤5 min)** — roteiro exigido pelo desafio:
   - **Situation:** hospital precisa triar laudos por urgência rapidamente
   - **Task:** API leve com CI/CD, monitoramento e otimização de latência
   - **Action:** BioBERT pseudo-rotula → modelo leve destilado → FastAPI em
     Docker → Actions + Airflow → Prometheus/Grafana → ONNX
   - **Result:** demo ao vivo (request na API, dashboard mexendo) + tabela
     de latência sklearn vs ONNX + lições aprendidas

## Decisões já tomadas que esta etapa deve respeitar

| Decisão | Implicação |
|---|---|
| Técnica escolhida: ONNX Runtime | Já validada; quantização fica como plano B/extra, não substituto |
| Comparação é modelo vs modelo, offline | Não precisa da API para os números oficiais (a API mede o dela na Etapa 1) |
| Artefatos fora do git | `.onnx` não entra no git; `latency_benchmark.json` (pequeno) entra em `docs/results/` |
| Sem torch no caminho de inferência | O `.onnx` roda só com `onnxruntime` — mantém a imagem de serving leve |

## Checklist de aceite

- [ ] `uv run python -m src.export_onnx` gera o `.onnx` a partir do joblib
- [ ] `uv run python -m src.benchmark` imprime a tabela e salva o JSON
- [ ] Divergência sklearn vs ONNX investigada e documentada com números
- [ ] README com a seção de otimização e a tabela comparativa
- [ ] Vídeo gravado (≤5 min), link na entrega
- [ ] Testes existentes seguem passando (`uv run pytest`)

## Referências

| Documento | Conteúdo |
|---|---|
| [README.md](../README.md) | Quickstart, arquitetura, resultados dos modelos |
| [triage_metrics_v2.json](results/triage_metrics_v2.json) | Métricas do modelo que será otimizado |
| [handoff_api.md](handoff_api.md) | Etapa 1 (paralela) — ponto de integração opcional do backend ONNX |
