"""Configuração central do projeto: caminhos e constantes do pipeline."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROCESSED_DIR / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "docs" / "results"

TRAIN_PATH = RAW_DIR / "medical_tc_train.csv"
TEST_PATH = RAW_DIR / "medical_tc_test.csv"
LABELS_PATH = RAW_DIR / "medical_tc_labels.csv"

PSEUDOLABELED_PATH = PROCESSED_DIR / "medical_abstracts_triage_pseudolabeled.csv"
CHECKPOINT_PATH = PROCESSED_DIR / "medical_abstracts_triage_checkpoint.csv"

# Modelo pré-treinado usado para gerar os pseudo-rótulos (ver notebook 02).
PSEUDOLABEL_MODEL = "Yuvrajxms09/biobert-triage-classifier"

# Regra de triagem validada no notebook 03.
CONFIDENCE_THRESHOLD = 0.70

# 512 é o limite do BERT; com 256 (piloto), 60% dos abstracts eram truncados.
MAX_LENGTH = 512
BATCH_SIZE = 32
CHECKPOINT_EVERY = 500

RANDOM_STATE = 42

TRIAGE_LEVELS = ["normal", "atenção", "urgente"]
