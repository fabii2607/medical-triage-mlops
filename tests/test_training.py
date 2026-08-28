import joblib
import pandas as pd

from src.parameters import load_params
from src.training.pipeline import build_logreg_pipeline, build_mlp_pipeline
from src.training.train import save_model, train_model


def training_dataset():
    return pd.DataFrame(
        {
            "medical_abstract": [
                "stable routine clinical followup",
                "stable routine patient observation",
                "normal outpatient clinical review",
                "normal outpatient routine followup",
                "moderate symptoms require attention",
                "moderate condition clinical attention",
                "persistent symptoms require observation",
                "persistent condition medical observation",
                "acute severe cardiac emergency",
                "acute severe respiratory emergency",
                "critical cardiac urgent intervention",
                "critical respiratory urgent intervention",
            ],
            "triage_level": ["normal"] * 4 + ["atenção"] * 4 + ["urgente"] * 4,
        }
    )


def test_pipelines_use_params_yaml():
    params = load_params()
    logreg_pipeline = build_logreg_pipeline(params.train)
    mlp_pipeline = build_mlp_pipeline(params.train)

    logreg = logreg_pipeline.named_steps["clf"]
    mlp = mlp_pipeline.named_steps["clf"]
    tfidf = logreg_pipeline.named_steps["tfidf"]

    assert logreg.C == params.train.logistic_regression.C
    assert logreg.class_weight == params.train.logistic_regression.class_weight
    assert logreg.solver == params.train.logistic_regression.solver
    assert mlp.hidden_layer_sizes == params.train.mlp.hidden_layer_sizes
    assert tfidf.ngram_range == params.train.tfidf.ngram_range
    assert tfidf.max_features == params.train.tfidf.max_features


def test_train_and_save_operational_model(tmp_path):
    data = training_dataset()
    model = train_model(data, load_params().train)
    output_path = tmp_path / "logreg_tfidf.joblib"

    save_model(model, output_path)
    loaded_model = joblib.load(output_path)

    assert output_path.exists()
    assert set(loaded_model.classes_) == {"normal", "atenção", "urgente"}
    assert len(loaded_model.predict(["acute severe cardiac emergency"])) == 1
