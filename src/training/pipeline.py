"""Construção dos pipelines de modelos de triagem."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from src.parameters import TfidfParams, TrainParams


def build_tfidf(params: TfidfParams) -> TfidfVectorizer:
    """Cria o TF-IDF com a configuração declarada em params.yaml."""
    return TfidfVectorizer(
        ngram_range=params.ngram_range,
        min_df=params.min_df,
        max_df=params.max_df,
        max_features=params.max_features,
        stop_words=params.stop_words,
        sublinear_tf=params.sublinear_tf,
    )


def build_logreg_pipeline(params: TrainParams) -> Pipeline:
    """Cria o pipeline operacional TF-IDF + Logistic Regression."""
    logreg = params.logistic_regression
    return Pipeline(
        [
            ("tfidf", build_tfidf(params.tfidf)),
            (
                "clf",
                LogisticRegression(
                    C=logreg.C,
                    class_weight=logreg.class_weight,
                    max_iter=logreg.max_iter,
                    solver=logreg.solver,
                    random_state=logreg.random_state,
                ),
            ),
        ]
    )


def build_mlp_pipeline(params: TrainParams) -> Pipeline:
    """Cria o MLP mantido exclusivamente para benchmarks."""
    mlp = params.mlp
    return Pipeline(
        [
            ("tfidf", build_tfidf(params.tfidf)),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=mlp.hidden_layer_sizes,
                    activation=mlp.activation,
                    solver=mlp.solver,
                    alpha=mlp.alpha,
                    learning_rate_init=mlp.learning_rate_init,
                    max_iter=mlp.max_iter,
                    early_stopping=mlp.early_stopping,
                    validation_fraction=mlp.validation_fraction,
                    n_iter_no_change=mlp.n_iter_no_change,
                    random_state=mlp.random_state,
                ),
            ),
        ]
    )
