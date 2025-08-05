"""Modules d'analyses statistiques."""
from .stats_descriptives import detect_outliers_iqr
from .correlations import pearson_correlations
from .regressions import simple_linear_regression
from .clustering import kmeans_clustering

__all__ = [
    "detect_outliers_iqr",
    "pearson_correlations",
    "simple_linear_regression",
    "kmeans_clustering",
]
