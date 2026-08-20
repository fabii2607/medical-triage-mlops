from src.evaluation.metrics import evaluate_predictions

LABELS = ["normal", "atenção", "urgente"]


def test_predicao_perfeita():
    y = ["normal", "atenção", "urgente", "normal"]
    metrics = evaluate_predictions(y, y, LABELS)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0


def test_classe_nunca_prevista_nao_quebra():
    y_true = ["normal", "urgente", "urgente", "atenção"]
    y_pred = ["normal", "normal", "normal", "atenção"]
    metrics = evaluate_predictions(y_true, y_pred, LABELS)
    assert metrics["per_class"]["urgente"]["recall"] == 0.0
    assert 0.0 < metrics["accuracy"] < 1.0


def test_estrutura_do_resultado():
    y = ["normal", "atenção", "urgente"]
    metrics = evaluate_predictions(y, y, LABELS)
    assert set(metrics["per_class"]) >= set(LABELS)
    assert len(metrics["confusion_matrix"]) == 3
    assert metrics["confusion_matrix_labels"] == LABELS
