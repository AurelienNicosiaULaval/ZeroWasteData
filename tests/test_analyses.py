"""Tests unitaires des fonctions d'analyse."""
from __future__ import annotations

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np

from analyses.correlations import pearson_correlations
from analyses.stats_descriptives import detect_outliers_iqr
from analyses.regressions import simple_linear_regression


def test_pearson_correlations():
    np.random.seed(0)
    x = np.arange(50)
    y = x + np.random.normal(scale=0.1, size=50)
    z = np.random.normal(size=50)
    df = pd.DataFrame({"x": x, "y": y, "z": z})
    corr_df = pearson_correlations(df, threshold=0.01)
    assert ((corr_df["var1"] == "x") & (corr_df["var2"] == "y")).any()


def test_detect_outliers_iqr():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 100]})
    mask = detect_outliers_iqr(df, "a")
    assert mask.iloc[-1]
    assert mask.sum() == 1


def test_simple_linear_regression():
    x = np.arange(100)
    y = 3 * x + 5
    df = pd.DataFrame({"x": x, "y": y})
    pval, r2 = simple_linear_regression(df, "x", "y")
    assert pval < 0.05
    assert r2 > 0.99
