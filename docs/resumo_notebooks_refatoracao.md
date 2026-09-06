# Resumo dos Notebooks para Refatoração

> **Verificação (2026-08-19, Erick):** os notebooks foram auditados contra este resumo.
> Os números reportados conferem, mas com correções importantes:
>
> 1. **04 — erro factual:** o limite de execução era **30 minutos** (`MAX_RUNTIME_MINUTES = 30`), não ~10.
> 2. **04 — viés de seleção crítico:** os 720 abstracts processados eram os **primeiros em
>    ordem alfabética** de 11.227 únicos (o `groupby` do pandas ordena) — não uma amostra
>    neutra. O shuffle sugerido abaixo era correção obrigatória, não melhoria opcional.
> 3. **02/03 — decisão descartada:** o notebook 02 desaconselha explicitamente derivar
>    `atenção` de faixa de confiança ("confiança ≠ severidade"); os notebooks 03/04 seguem
>    com a regra mesmo assim. Decisão mantida e documentada como limitação no PLANNING.md.
> 4. **03 — escopo:** a regra foi estudada apenas sobre o piloto de 60 textos.
> 5. Truncamento em 256 tokens afetava **60,3%** dos textos; splits reais eram 504/108/108
>    (só 16 `urgente` no teste — daí o recall 0.000 do baseline).
>
> **Status:** a refatoração sugerida foi aplicada em `src/` e o pipeline foi reexecutado
> completo em GPU (11.227 abstracts, `max_length=512`, shuffle): ver
> `docs/results/triage_metrics_v2.json`. Este documento permanece como registro do estado
> dos notebooks na entrega da Fabi.

## 01_eda.ipynb

Responsável pela análise exploratória do dataset original **Medical Abstracts TC Corpus**.

O notebook:
- carrega treino, teste e labels;
- verifica valores nulos, textos vazios e duplicatas;
- analisa a distribuição das classes médicas;
- analisa o tamanho dos abstracts;
- mostra palavras e bigramas frequentes;
- identifica textos associados a mais de uma categoria;
- verifica repetição de textos entre treino e teste.

### Principal conclusão
O dataset original possui categorias de doenças como target, e não níveis de urgência. Também foi identificado que alguns textos aparecem associados a múltiplas categorias e existem textos repetidos entre treino e teste.

### Refatoração sugerida
Manter este notebook focado apenas no EDA, deixando transformações e lógica reutilizável fora dele.

---

## 02_biobert_pilot.ipynb

Responsável por testar o modelo pré-treinado:

`Yuvrajxms09/biobert-triage-classifier`

A ideia foi utilizar um modelo já treinado para identificar urgência em textos médicos e verificar se ele poderia ser usado para criar pseudo-rótulos para o nosso dataset.

O BioBERT utilizado é binário:

- `urgent`
- `non-urgent`

O notebook:
- carrega o BioBERT;
- seleciona uma pequena amostra dos abstracts;
- gera `urgent_score` e `nonurgent_score`;
- calcula a confiança;
- mostra exemplos mais urgentes, menos urgentes e incertos;
- verifica o impacto do limite de tokens do modelo.

### Refatoração sugerida
Deixar este notebook apenas como experimento/piloto para justificar a escolha do BioBERT, sem processamento completo do dataset.

---

## 03_triage_rules.ipynb

Responsável por adaptar a classificação binária do BioBERT para as três classes solicitadas no projeto:

- `normal`
- `atenção`
- `urgente`

Foi criada uma zona de incerteza usando threshold inicial de `0.70`:

```text
nonurgent_score >= 0.70 -> normal
urgent_score >= 0.70    -> urgente
caso contrário          -> atenção
```

### Observação importante
O BioBERT **não aprendeu a classe `atenção`**. Ela é uma regra operacional criada para representar casos em que o classificador binário ficou incerto.

O notebook também compara diferentes thresholds para analisar como a distribuição das classes muda.

### Refatoração sugerida
Mover a regra de classificação para uma função reutilizável em `src/`, deixando no notebook apenas os experimentos, gráficos e justificativas.

---

## 04_pseudolabeling.ipynb

Responsável por aplicar o BioBERT no dataset e gerar os pseudo-rótulos:

```text
medical_abstract
       ↓
BioBERT
       ↓
urgent_score / nonurgent_score
       ↓
regra de threshold
       ↓
normal / atenção / urgente
```

O notebook:
- consolida abstracts únicos;
- executa o BioBERT em batches;
- gera os scores;
- aplica a regra das três classes;
- salva checkpoints;
- possui limite máximo de execução de aproximadamente **10 minutos**;
- analisa a distribuição resultante;
- mantém as categorias médicas originais para rastreabilidade.

### Resultado atual da execução

Foram processados 720 abstracts, com a seguinte distribuição:

```text
normal  -> 306 (42,5%)
atenção -> 302 (41,9%)
urgente -> 112 (15,6%)
```

### Ponto importante para refatoração
Como existe limite de 10 minutos, o notebook pode processar apenas uma parte do dataframe. Para tornar essa parte mais representativa, o ideal é embaralhar os abstracts antes da inferência:

```python
unique_abstracts = (
    unique_abstracts
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)
```

Também é interessante mover inferência, checkpoint e aplicação da regra para funções reutilizáveis em `src/`.

---

## 05_data_split.ipynb

Responsável por preparar o dataset pseudo-rotulado para treinamento.

O notebook:
- carrega os pseudo-rótulos;
- realiza limpeza mínima;
- remove duplicações de texto;
- cria uma divisão estratificada de:

```text
70% treino
15% validação
15% teste
```

- verifica se o mesmo abstract não aparece em mais de um conjunto;
- salva os arquivos finais em:

```text
data/processed/splits/
├── train.csv
├── validation.csv
└── test.csv
```

O modelo utiliza somente:

```text
X = medical_abstract
y = triage_level
```

Os scores do BioBERT ficam apenas para rastreabilidade e não entram como features.

### Refatoração sugerida
Transformar a criação dos splits em função ou script reproduzível e deixar o notebook focado na validação visual e conferência da distribuição.

---

## 06_mlp_training.ipynb

Responsável por treinar o primeiro modelo próprio do projeto.

Pipeline utilizado:

```text
medical_abstract
       ↓
TF-IDF
       ↓
MLPClassifier
       ↓
normal / atenção / urgente
```

O modelo atual possui uma camada oculta com 64 neurônios.

Foram avaliados:
- Accuracy;
- Balanced Accuracy;
- Macro Precision;
- Macro Recall;
- Macro F1;
- métricas por classe;
- matriz de confusão;
- latência;
- tamanho do modelo.

### Resultado atual do baseline

```text
Accuracy:              0.565
Balanced Accuracy:     0.442
Macro F1:              0.406
Macro Recall:          0.442
Recall urgente:        0.000

Latência média:        4.32 ms
Latência P95:          6.83 ms
Tamanho:               7.53 MB
Treinamento:           3.33 s
```

### Principal conclusão
O modelo ficou leve e rápido, porém ainda não apresentou desempenho suficiente, principalmente porque o recall da classe `urgente` ficou em `0.000`.

Isso indica que o modelo baseline não conseguiu identificar corretamente os casos pseudo-rotulados como urgentes no conjunto de teste.

### Refatoração sugerida
Manter esse modelo como `baseline v1`, sem sobrescrever os resultados.

Depois, testar:
- maior quantidade de dados pseudo-rotulados;
- melhor balanceamento do conjunto de treino;
- novas configurações do MLP;
- Logistic Regression como baseline adicional para TF-IDF.

---

# Arquitetura Geral Construída até Agora

```text
Medical Abstracts
       ↓
01 - EDA
       ↓
02 - Piloto BioBERT
       ↓
03 - Regra de triagem
       ↓
04 - Pseudo-labeling
       ↓
normal / atenção / urgente
       ↓
05 - Train / Validation / Test
       ↓
06 - TF-IDF + MLP
       ↓
Modelo baseline
```

# Sugestão Geral para a Refatoração

A recomendação é deixar os notebooks focados em:

- experimentos;
- gráficos;
- análises;
- documentação das decisões.

A lógica reutilizável pode ser movida para `src/`.

Estrutura sugerida:

```text
src/
├── data/
│   ├── preprocessing.py
│   └── split.py
│
├── labeling/
│   ├── biobert.py
│   └── triage_rules.py
│
├── training/
│   └── train_mlp.py
│
└── evaluation/
    └── metrics.py
```

Assim, a lógica importante deixa de ficar espalhada em células de notebook e o projeto fica mais preparado para:

- CI/CD;
- Airflow;
- FastAPI;
- testes automatizados;
- monitoramento com Prometheus e Grafana;
- comparação entre modelos.
