"""Fonctions d'analyses descriptives.

Détection de valeurs aberrantes selon la méthode de l'IQR.
"""
from __future__ import annotations

from typing import List

import pandas as pd


def detect_outliers_iqr(df: pd.DataFrame, column: str) -> pd.Series:
    """Retourne une série booléenne indiquant les valeurs aberrantes pour *column*.

    La méthode utilise l'IQR (interquartile range) :
        - Q1 = 25e percentile
        - Q3 = 75e percentile
        - IQR = Q3 - Q1
        - Limites = [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]

    Parameters
    ----------
    df : pd.DataFrame
        Données d'entrée.
    column : str
        Colonne sur laquelle détecter les outliers.

    Returns
    -------
    pd.Series
        Série booléenne de même taille que *df[column]* où `True` indique
        un outlier.
    """
    series = df[column].dropna()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (df[column] < lower) | (df[column] > upper)
