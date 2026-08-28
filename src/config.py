"""Caminhos utilizados pelo projeto."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROCESSED_DIR / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "docs" / "results"
PARAMS_PATH = PROJECT_ROOT / "params.yaml"

TRAIN_PATH = RAW_DIR / "medical_tc_train.csv"
TEST_PATH = RAW_DIR / "medical_tc_test.csv"
LABELS_PATH = RAW_DIR / "medical_tc_labels.csv"

PSEUDOLABELED_PATH = PROCESSED_DIR / "medical_abstracts_triage_pseudolabeled.csv"
CHECKPOINT_PATH = PROCESSED_DIR / "medical_abstracts_triage_checkpoint.csv"
