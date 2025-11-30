"""Fonctions de statistiques descriptives."""

from __future__ import annotations

from typing import Dict, Any, Optional
import pandas as pd
import streamlit as st
from plotnine import (
    ggplot,
    aes,
    geom_boxplot,
    geom_histogram,
    theme_minimal,
    labs,
    theme,
    element_text,
)
from .base import BaseAnalysis


def detect_outliers_iqr(df: pd.DataFrame, col: str) -> pd.Series:
    """Détecte les outliers via la méthode IQR."""
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return (df[col] < lower_bound) | (df[col] > upper_bound)


class OutlierAnalysis(BaseAnalysis):
    @property
    def name(self) -> str:
        return "Valeurs aberrantes"

    @property
    def category(self) -> str:
        return "Descriptive/Exploratoire"

    @property
    def description(self) -> str:
        return "Détection des valeurs aberrantes utilisant la méthode de l'écart interquartile (IQR)."

    def check_applicability(self, df: pd.DataFrame) -> bool:
        return len(df.select_dtypes(include="number").columns) > 0

    def run(self, df: pd.DataFrame) -> Dict[str, int]:
        num_cols = df.select_dtypes(include="number").columns
        results = {}
        for col in num_cols:
            mask = detect_outliers_iqr(df, col)
            if mask.any():
                results[col] = int(mask.sum())
        return results

    def render_streamlit(
        self, df: pd.DataFrame, result: Dict[str, int]
    ) -> Optional[str]:
        st.write("### Valeurs aberrantes")
        if not result:
            st.write("Aucune valeur aberrante détectée.")
            return None

        st.write(f"Colonnes avec outliers : {', '.join(result.keys())}")

        with st.expander("Voir les détails et graphiques"):
            for col, count in result.items():
                st.write(f"**{col}**: {count} outliers")

                # Plotnine boxplot
                p = (
                    ggplot(df, aes(x=0, y=col))
                    + geom_boxplot(fill="steelblue", alpha=0.7)
                    + theme_minimal()
                    + labs(title=f"Boxplot de {col}", x="", y=col)
                    + theme(axis_text_x=element_text(size=0))  # Hide x axis text
                )
                st.pyplot(p.draw())

        return "\n".join(f"- {k}: {v} outliers" for k, v in result.items())


class DistributionAnalysis(BaseAnalysis):
    @property
    def name(self) -> str:
        return "Distribution des données"

    @property
    def category(self) -> str:
        return "Descriptive/Exploratoire"

    @property
    def description(self) -> str:
        return (
            "Visualisation de la distribution des variables numériques (Histogrammes)."
        )

    def check_applicability(self, df: pd.DataFrame) -> bool:
        return len(df.select_dtypes(include="number").columns) > 0

    def run(self, df: pd.DataFrame) -> None:
        # Pas de calcul complexe, juste de la visu
        return None

    def render_streamlit(self, df: pd.DataFrame, result: Any) -> Optional[str]:
        st.write("### Distribution des données")
        num_cols = df.select_dtypes(include="number").columns

        col_to_plot = st.selectbox("Choisir une variable", num_cols, key="dist_col")

        # Plotnine histogram
        p = (
            ggplot(df, aes(x=col_to_plot))
            + geom_histogram(fill="steelblue", color="white", bins=30, alpha=0.7)
            + theme_minimal()
            + labs(title=f"Distribution de {col_to_plot}", x=col_to_plot, y="Fréquence")
        )
        st.pyplot(p.draw())

        return f"Distribution analysée pour {col_to_plot}."
