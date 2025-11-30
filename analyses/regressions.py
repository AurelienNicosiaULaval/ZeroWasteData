"""Fonctions de régression simple."""

from __future__ import annotations

from typing import Tuple, Optional, Any
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from plotnine import ggplot, aes, geom_point, geom_smooth, theme_minimal, labs
from .base import BaseAnalysis


def simple_linear_regression(
    df: pd.DataFrame, x_col: str, y_col: str
) -> Tuple[float, float]:
    """Réalise une régression linéaire simple et retourne p-value et R²."""
    # Drop NA for regression
    data = df[[x_col, y_col]].dropna()
    x = sm.add_constant(data[x_col])
    y = data[y_col]
    model = sm.OLS(y, x).fit()
    return float(model.f_pvalue), float(model.rsquared)


class SimpleLinearRegressionAnalysis(BaseAnalysis):
    @property
    def name(self) -> str:
        return "Régression linéaire simple"

    @property
    def category(self) -> str:
        return "Modélisation prédictive simple"

    @property
    def description(self) -> str:
        return "Modélisation de la relation linéaire entre deux variables numériques."

    def check_applicability(self, df: pd.DataFrame) -> bool:
        return len(df.select_dtypes(include="number").columns) >= 2

    def run(self, df: pd.DataFrame) -> None:
        # On ne lance pas tout d'un coup, c'est interactif
        return None

    def render_streamlit(self, df: pd.DataFrame, result: Any) -> Optional[str]:
        st.write("### Régression linéaire simple")
        num_cols = df.select_dtypes(include="number").columns

        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Variable explicative (X)", num_cols, key="reg_x")
        with col2:
            y_col = st.selectbox(
                "Variable cible (Y)", [c for c in num_cols if c != x_col], key="reg_y"
            )

        if x_col and y_col:
            pval, r2 = simple_linear_regression(df, x_col, y_col)

            st.write(f"Modèle : **{y_col} ~ {x_col}**")
            st.write(f"- R² : {r2:.3f}")
            st.write(f"- p-value : {pval:.3g}")

            with st.expander("Voir le graphique"):
                p = (
                    ggplot(df, aes(x=x_col, y=y_col))
                    + geom_point(alpha=0.6)
                    + geom_smooth(method="lm", color="red")
                    + theme_minimal()
                    + labs(title=f"Régression : {y_col} vs {x_col}")
                )
                st.pyplot(p.draw())

            return f"Régression {y_col} ~ {x_col} : R²={r2:.3f}, p-value={pval:.3g}"
        return None
