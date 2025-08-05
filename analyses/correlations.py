"""Fonctions de détection des corrélations."""
from __future__ import annotations

from itertools import combinations
from typing import List, Tuple

import pandas as pd
from scipy.stats import pearsonr


def pearson_correlations(df: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """Calcule les corrélations de Pearson significatives entre paires de colonnes.

    Parameters
    ----------
    df : pd.DataFrame
        Données numériques.
    threshold : float, optional
        Seuil de p-value pour retenir la corrélation, by default 0.05.

    Returns
    -------
    pd.DataFrame
        Tableau contenant les colonnes `var1`, `var2`, `corr` et `p_value`
        pour chaque paire significativement corrélée.
    """
    numeric_df = df.select_dtypes(include="number").dropna()
    rows = []
    for col1, col2 in combinations(numeric_df.columns, 2):
        corr, p = pearsonr(numeric_df[col1], numeric_df[col2])
        if p < threshold:
            rows.append({"var1": col1, "var2": col2, "corr": corr, "p_value": p})
    return pd.DataFrame(rows)
