"""Fonctions de clustering."""
from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.cluster import KMeans


def kmeans_clustering(df: pd.DataFrame, n_clusters: int = 3) -> pd.Series:
    """Applique un clustering K-means sur les variables numériques.

    Parameters
    ----------
    df : pd.DataFrame
        Données d'entrée.
    n_clusters : int, optional
        Nombre de clusters, by default 3.

    Returns
    -------
    pd.Series
        Étiquettes de cluster pour chaque ligne.
    """
    numeric_df = df.select_dtypes(include="number").dropna()
    model = KMeans(n_clusters=n_clusters, n_init=10)
    labels = model.fit_predict(numeric_df)
    return pd.Series(labels, index=numeric_df.index, name="cluster")
