"""Fonctions de détection des corrélations."""

from __future__ import annotations

from itertools import combinations
from typing import Optional
import pandas as pd
import streamlit as st
from plotnine import (
    ggplot,
    aes,
    geom_tile,
    scale_fill_gradient2,
    theme_minimal,
    labs,
    geom_text,
    theme,
    element_text,
)
from scipy.stats import pearsonr
from .base import BaseAnalysis


def pearson_correlations(df: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """Calcule les corrélations de Pearson significatives entre paires de colonnes."""
    numeric_df = df.select_dtypes(include="number").dropna()
    rows = []
    for col1, col2 in combinations(numeric_df.columns, 2):
        corr, p = pearsonr(numeric_df[col1], numeric_df[col2])
        if p < threshold:
            rows.append({"var1": col1, "var2": col2, "corr": corr, "p_value": p})
    return pd.DataFrame(rows)


class CorrelationAnalysis(BaseAnalysis):
    @property
    def name(self) -> str:
        return "Corrélation linéaire (Pearson)"

    @property
    def category(self) -> str:
        return "Corrélationnelle"

    @property
    def description(self) -> str:
        return (
            "Analyse des corrélations linéaires (Pearson) entre variables numériques."
        )

    def check_applicability(self, df: pd.DataFrame) -> bool:
        return len(df.select_dtypes(include="number").columns) >= 2

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        return pearson_correlations(df)

    def render_streamlit(self, df: pd.DataFrame, result: pd.DataFrame) -> Optional[str]:
        st.write("### Corrélations significatives")

        if result.empty:
            st.write("Aucune corrélation significative détectée.")
            return None

        st.write("Paires de variables significativement corrélées :")
        st.dataframe(result)

        with st.expander("Voir la matrice de corrélation"):
            # Prepare data for plotnine heatmap
            corr_matrix = df.select_dtypes(include="number").corr().reset_index()
            corr_melted = corr_matrix.melt(
                id_vars="index", var_name="variable", value_name="correlation"
            )

            p = (
                ggplot(corr_melted, aes(x="index", y="variable", fill="correlation"))
                + geom_tile(color="white")
                + scale_fill_gradient2(
                    low="blue", mid="white", high="red", midpoint=0, limit=(-1, 1)
                )
                + geom_text(aes(label="correlation"), format_string="{:.2f}", size=8)
                + theme_minimal()
                + labs(title="Matrice de Corrélation", x="", y="")
                + theme(axis_text_x=element_text(rotation=45, hjust=1))
            )
            st.pyplot(p.draw())

        return result.to_markdown(index=False)

    def generate_code(self, df_name: str = "df", **kwargs) -> str:
        return f"""
# Analyse de Corrélation
from scipy.stats import pearsonr
from itertools import combinations
import pandas as pd
from plotnine import ggplot, aes, geom_tile, scale_fill_gradient2, theme_minimal, labs, geom_text, theme, element_text

# Calcul des corrélations
numeric_df = {df_name}.select_dtypes(include="number").dropna()
rows = []
for col1, col2 in combinations(numeric_df.columns, 2):
    corr, p = pearsonr(numeric_df[col1], numeric_df[col2])
    if p < 0.05:
        rows.append({{"var1": col1, "var2": col2, "corr": corr, "p_value": p}})
print(pd.DataFrame(rows))

# Matrice de corrélation (Heatmap)
corr_matrix = numeric_df.corr().reset_index()
corr_melted = corr_matrix.melt(id_vars='index', var_name='variable', value_name='correlation')

p = (ggplot(corr_melted, aes(x='index', y='variable', fill='correlation'))
     + geom_tile(color="white")
     + scale_fill_gradient2(low="blue", mid="white", high="red", midpoint=0, limit=(-1, 1))
     + geom_text(aes(label='correlation'), format_string='{{:.2f}}', size=8)
     + theme_minimal()
     + labs(title="Matrice de Corrélation", x="", y="")
     + theme(axis_text_x=element_text(rotation=45, hjust=1))
    )
print(p)
"""
