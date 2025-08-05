"""Fonctions de régression simple."""
from __future__ import annotations

from typing import Tuple

import pandas as pd
import statsmodels.api as sm


def simple_linear_regression(df: pd.DataFrame, x_col: str, y_col: str) -> Tuple[float, float]:
    """Réalise une régression linéaire simple et retourne p-value et R².

    Parameters
    ----------
    df : pd.DataFrame
        Données d'entrée.
    x_col : str
        Nom de la variable explicative.
    y_col : str
        Nom de la variable cible.

    Returns
    -------
    Tuple[float, float]
        Tuple `(p_value, r2)` du modèle.
    """
    x = sm.add_constant(df[x_col])
    y = df[y_col]
    model = sm.OLS(y, x).fit()
    return float(model.f_pvalue), float(model.rsquared)
