"""Exporta o pipeline sklearn de produção para ONNX.

Converte `models/logreg_tfidf.joblib` (TF-IDF + LogisticRegression) em um
único grafo ONNX com o vetorizador embutido — a inferência otimizada precisa
apenas de `onnxruntime`, sem scikit-learn no runtime.

Uso:
    uv run python -m src.export_onnx
"""

from __future__ import annotations

import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

from src.config import MODEL_PATH, MODELS_DIR

ONNX_MODEL_PATH = MODELS_DIR / "logreg_tfidf.onnx"

# Nome do input do grafo ONNX; consumidores fazem
# session.run(None, {ONNX_INPUT_NAME: np.array([[texto]], dtype=object)}).
ONNX_INPUT_NAME = "medical_abstract"


def export_model() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} não encontrado — rode `dvc pull` ou `dvc repro` antes."
        )

    model = joblib.load(MODEL_PATH)

    onnx_model = convert_sklearn(
        model,
        initial_types=[(ONNX_INPUT_NAME, StringTensorType([None, 1]))],
        # zipmap=False devolve as probabilidades como matriz (n, 3) em vez de
        # lista de dicts — mais simples e mais rápido de consumir.
        options={"zipmap": False},
    )

    ONNX_MODEL_PATH.write_bytes(onnx_model.SerializeToString())

    original_kb = MODEL_PATH.stat().st_size / 1024
    exported_kb = ONNX_MODEL_PATH.stat().st_size / 1024
    print(f"Original (joblib): {MODEL_PATH} ({original_kb:.0f} KB)")
    print(f"Exportado (ONNX):  {ONNX_MODEL_PATH} ({exported_kb:.0f} KB)")


if __name__ == "__main__":
    export_model()
