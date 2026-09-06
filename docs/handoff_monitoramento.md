# Handoff — Etapa 3: Monitoramento (Prometheus + Grafana)

> **[2026-09-06] Etapa entregue.** Documento mantido como registro. Diferença
> em relação ao combinado aqui: o contrato de métricas implementado — e
> **adotado oficialmente** — usa o prefixo `medical_triage_*` com labels
> próprias: `medical_triage_requests_total{method,endpoint,status_code}`,
> `medical_triage_request_duration_seconds{method,endpoint}` e
> `medical_triage_predictions_total{triage_level}`. Dashboard, Prometheus e
> README seguem esses nomes; a tabela de contrato abaixo ficou histórica.

> Guia autossuficiente para quem vai implementar esta etapa. Pode ser colado
> como contexto no seu copilot — contém o estado do repo, os contratos e os
> critérios de aceite.

## Contexto em 1 minuto

O projeto é um sistema de triagem de laudos médicos servido por uma API
FastAPI (Etapa 1, em desenvolvimento paralelo) que classifica texto em
`urgente`, `atenção`, `normal`. Esta etapa entrega a **stack local de
observabilidade**: API + Prometheus + Grafana subindo juntos via Docker
Compose, com dashboard de **pelo menos 3 painéis** e um gerador de carga
para popular os gráficos. Entregável do desafio: compose rodando a stack
completa + print/JSON do dashboard.

## Dependência (e como não ficar bloqueado)

A instrumentação final vive dentro da API, que está sendo feita em paralelo.
**~Metade da etapa não depende dela** — comece por:

1. `docker-compose.yml` com Prometheus + Grafana + provisionamento automático
2. Dashboard JSON versionado
3. Gerador de carga
4. **Stub da API** para desenvolver sem esperar: um FastAPI de ~20 linhas que
   expõe `/metrics` (prometheus-client) e um `/predict` fake que sorteia uma
   classe e incrementa as métricas do contrato abaixo. Quando a API real
   chegar, é trocar o serviço no compose — dashboard e Prometheus não mudam.

## Setup

```bash
git clone https://github.com/fabii2607/medical-triage-mlops.git
cd medical-triage-mlops
uv sync            # fastapi, uvicorn e prometheus-client já estão nas deps
```

## Contrato de métricas (combinado com quem faz a API)

Estes nomes são o contrato de integração — use exatamente estes:

| Métrica | Tipo | Labels | O que mede |
|---|---|---|---|
| `triage_requests_total` | Counter | `endpoint`, `http_status` | Total de requisições |
| `triage_request_latency_seconds` | Histogram | `endpoint` | Latência da requisição |
| `triage_predictions_total` | Counter | `triage_level` | Predições por classe |

Instrumentação com `prometheus_client` (Counter, Histogram) exposta em
`GET /metrics` — na API real isso é feito pelo dev da Etapa 1; no stub, por você.

## Painéis do dashboard (mínimo 3 — sugestão de 4)

1. **Requisições/s** — `rate(triage_requests_total[1m])`
2. **Latência p50/p95** — `histogram_quantile(0.95, rate(triage_request_latency_seconds_bucket[5m]))`
3. **Taxa de erro** — razão das séries com `http_status=~"5.."` sobre o total
4. **Predições por classe** — `rate(triage_predictions_total[5m])` por `triage_level`
   — este painel é um **guard-rail de negócio já decidido pelo time**: o
   baseline v1 tinha recall 0 na classe `urgente`; se a série de `urgente`
   ficar zerada em produção, é sinal de regressão grave do modelo.

## Estrutura esperada

```
monitoring/
├── prometheus.yml                      # scrape da api:8000/metrics, interval 5s
└── grafana/
    └── provisioning/
        ├── datasources/datasource.yml  # Prometheus como default
        └── dashboards/
            ├── dashboards.yml
            └── triage_dashboard.json   # dashboard versionado no git
docker-compose.yml                      # raiz: api + prometheus + grafana
scripts/load_test.py                    # gerador de carga
```

- Grafana provisionado **automaticamente** (datasource + dashboard) — subiu,
  está pronto, sem cliques manuais.
- Gerador de carga: script com `httpx` disparando `POST /predict` com textos
  reais de `data/raw/medical_tc_test.csv` (está no git), em loop com
  intervalo aleatório — inclua alguns payloads inválidos para popular o
  painel de erros.

## Decisões já tomadas que esta etapa deve respeitar

| Decisão | Implicação |
|---|---|
| A rota `/metrics` pertence ao Prometheus | A API já foi orientada a não ocupar a rota |
| Imagem da API é leve (sem torch/GPU) | O compose todo roda em qualquer máquina do time |
| Artefatos fora do git | O serviço da API no compose recebe o `models/*.joblib` via volume ou build — combine com o dev da Etapa 1 |
| Dashboard versionado | O JSON exportado do Grafana entra no git (é entregável do desafio) |

## Checklist de aceite

- [ ] `docker compose up` sobe API (ou stub), Prometheus e Grafana juntos
- [ ] Prometheus mostra o target da API como `UP` (`localhost:9090/targets`)
- [ ] Grafana abre com o dashboard já provisionado (`localhost:3000`)
- [ ] Gerador de carga rodando → os 4 painéis se movimentam
- [ ] Painel de predições mostra as 3 classes (inclusive `urgente`)
- [ ] `triage_dashboard.json` versionado no git + print do dashboard para a entrega

## Referências

| Documento | Conteúdo |
|---|---|
| [README.md](../README.md) | Quickstart, arquitetura, resultados |
| [handoff_api.md](handoff_api.md) | Etapa 1 (paralela) — spec da API que será instrumentada |
