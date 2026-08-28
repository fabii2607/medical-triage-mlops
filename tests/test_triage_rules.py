import pytest

from src.labeling.triage_rules import map_to_three_levels


def test_urgente_quando_urgent_score_no_threshold():
    assert (
        map_to_three_levels(urgent_score=0.70, nonurgent_score=0.30, threshold=0.70)
        == "urgente"
    )
    assert (
        map_to_three_levels(urgent_score=0.95, nonurgent_score=0.05, threshold=0.70)
        == "urgente"
    )


def test_normal_quando_nonurgent_score_no_threshold():
    assert (
        map_to_three_levels(urgent_score=0.30, nonurgent_score=0.70, threshold=0.70)
        == "normal"
    )
    assert (
        map_to_three_levels(urgent_score=0.05, nonurgent_score=0.95, threshold=0.70)
        == "normal"
    )


def test_atencao_na_zona_de_incerteza():
    assert map_to_three_levels(0.50, 0.50, threshold=0.70) == "atenção"
    assert map_to_three_levels(0.69, 0.31, threshold=0.70) == "atenção"
    assert map_to_three_levels(0.31, 0.69, threshold=0.70) == "atenção"


def test_threshold_customizado():
    assert map_to_three_levels(0.75, 0.25, threshold=0.80) == "atenção"
    assert map_to_three_levels(0.85, 0.15, threshold=0.80) == "urgente"


def test_threshold_ambiguo_rejeitado():
    with pytest.raises(ValueError, match=r"threshold deve ser > 0\.5"):
        map_to_three_levels(0.6, 0.4, threshold=0.5)
    with pytest.raises(ValueError, match=r"threshold deve ser > 0\.5"):
        map_to_three_levels(0.6, 0.4, threshold=0.4)
