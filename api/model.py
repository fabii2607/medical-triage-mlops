from pathlib import Path

import joblib


class TriageModel:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.model = None

    def load(self) -> None:
        self.model = joblib.load(self.model_path)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")

        prediction = self.model.predict([text])[0]

        probabilities = self.model.predict_proba([text])[0]

        probability_dict = {
            class_name: float(probability)
            for class_name, probability in zip(
                self.model.classes_,
                probabilities,
            )
        }

        return prediction, probability_dict