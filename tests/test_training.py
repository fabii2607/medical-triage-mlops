from src.parameters import load_params
from src.training.train_mlp import build_candidates


def test_candidates_use_params_yaml():
    params = load_params()
    candidates = build_candidates(params.train)

    logreg = candidates["logreg_tfidf"].named_steps["clf"]
    mlp = candidates["mlp_tfidf"].named_steps["clf"]
    tfidf = candidates["logreg_tfidf"].named_steps["tfidf"]

    assert logreg.C == params.train.logistic_regression.C
    assert logreg.class_weight == params.train.logistic_regression.class_weight
    assert logreg.solver == params.train.logistic_regression.solver
    assert mlp.hidden_layer_sizes == params.train.mlp.hidden_layer_sizes
    assert tfidf.ngram_range == params.train.tfidf.ngram_range
    assert tfidf.max_features == params.train.tfidf.max_features
