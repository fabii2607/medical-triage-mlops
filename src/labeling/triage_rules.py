"""Regra operacional que converte a saída binária do BioBERT em 3 níveis.

O BioBERT é binário (urgent / non-urgent). A classe `atenção` NÃO foi
aprendida pelo modelo: é uma zona de incerteza definida por threshold,
decisão registrada nos notebooks 02 e 03.
"""

from __future__ import annotations

from src.config import CONFIDENCE_THRESHOLD


def map_to_three_levels(
    urgent_score: float,
    nonurgent_score: float,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> str:
    """Mapeia os scores binários para normal / atenção / urgente.

    Com threshold > 0.5 e scores que somam 1, as regras são mutuamente
    exclusivas e a zona de atenção é `1 - threshold <= urgent_score < threshold`.
    """
    if threshold <= 0.5:
        raise ValueError(
            f"threshold deve ser > 0.5 (recebido {threshold}); abaixo disso "
            "as faixas urgente/normal se sobrepõem e a regra fica ambígua."
        )

    if urgent_score >= threshold:
        return "urgente"

    if nonurgent_score >= threshold:
        return "normal"

    return "atenção"
