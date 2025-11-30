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

    def render_streamlit(
        self, df: pd.DataFrame, result: Any
    ) -> tuple[Optional[str], Optional[Any]]:
        st.write("### Régression Linéaire Simple")

        num_cols = df.select_dtypes(include="number").columns
        c1, c2 = st.columns(2)
        x_col = c1.selectbox("Variable X (Indépendante)", num_cols, key="reg_x")
        y_col = c2.selectbox(
            "Variable Y (Dépendante)", [c for c in num_cols if c != x_col], key="reg_y"
        )

        if not x_col or not y_col:
            return None, None

        # Run regression
        data = df[[x_col, y_col]].dropna()
        X = sm.add_constant(data[x_col])
        y = data[y_col]
        model = sm.OLS(y, X).fit()

        st.write(model.summary())

        # Plot
        p = (
            ggplot(data, aes(x=x_col, y=y_col))
            + geom_point(alpha=0.6)
            + geom_smooth(method="lm", color="red")
            + theme_minimal()
            + labs(title=f"Régression : {y_col} vs {x_col}")
        )
        st.pyplot(p.draw())

        return f"Régression linéaire : {y_col} ~ {x_col}. R2 = {model.rsquared:.3f}.", p

    def generate_code(self, df_name: str = "df", **kwargs) -> str:
        x_col = kwargs.get("x_col", "X")
        y_col = kwargs.get("y_col", "Y")
        return f"""
# Régression Linéaire Simple
import statsmodels.api as sm
from plotnine import ggplot, aes, geom_point, geom_smooth, theme_minimal, labs

x_col = '{x_col}'
y_col = '{y_col}'

# Préparation des données
data = {df_name}[[x_col, y_col]].dropna()
X = sm.add_constant(data[x_col])
y = data[y_col]

# Modèle
model = sm.OLS(y, X).fit()
print(f"R²: {{model.rsquared:.3f}}")
print(f"p-value: {{model.f_pvalue:.3g}}")

# Graphique
p = (ggplot(data, aes(x=x_col, y=y_col))
     + geom_point(alpha=0.6)
     + geom_smooth(method="lm", color="red")
     + theme_minimal()
     + labs(title=f"Régression : {{y_col}} vs {{x_col}}")
    )
print(p)
"""

    def generate_r_code(self, df_name: str = "df", **kwargs) -> str:
        x_col = kwargs.get("x_col", "X")
        y_col = kwargs.get("y_col", "Y")
        return f"""
# Régression Linéaire Simple (R)
library(ggplot2)

x_col <- "{x_col}"
y_col <- "{y_col}"

# Modèle
formula <- as.formula(paste(y_col, "~", x_col))
model <- lm(formula, data = {df_name})
summary(model)

# Graphique
ggplot({df_name}, aes_string(x = x_col, y = y_col)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", color = "red") +
  theme_minimal() +
  labs(title = paste("Régression :", y_col, "vs", x_col))
"""
