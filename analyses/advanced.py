from typing import Optional, Any
import pandas as pd
import streamlit as st
import statsmodels.api as sm
from plotnine import (
    ggplot,
    aes,
    geom_line,
    theme_minimal,
    labs,
    geom_smooth,
    geom_point,
)
from .base import BaseAnalysis


class LogisticRegressionAnalysis(BaseAnalysis):
    @property
    def name(self) -> str:
        return "Régression Logistique"

    @property
    def category(self) -> str:
        return "Modélisation prédictive simple"

    @property
    def description(self) -> str:
        return (
            "Modélisation d'une variable binaire en fonction d'une variable numérique."
        )

    def check_applicability(self, df: pd.DataFrame) -> bool:
        # Need 1 binary target and 1 numeric predictor
        num_cols = df.select_dtypes(include="number").columns
        cat_cols = df.select_dtypes(include=["object", "category", "number"]).columns
        has_binary = any(df[c].nunique() == 2 for c in cat_cols)
        return len(num_cols) > 0 and has_binary

    def run(self, df: pd.DataFrame) -> Any:
        return None

    def render_streamlit(self, df: pd.DataFrame, result: Any) -> Optional[str]:
        st.write("### Régression Logistique")

        cols = df.columns
        binary_cols = [c for c in cols if df[c].nunique() == 2]
        num_cols = df.select_dtypes(include="number").columns

        col1, col2 = st.columns(2)
        with col1:
            target_col = st.selectbox(
                "Variable cible (Binaire)", binary_cols, key="logreg_y"
            )
        with col2:
            feature_col = st.selectbox(
                "Variable explicative (Numérique)", num_cols, key="logreg_x"
            )

        if target_col and feature_col:
            # Prepare data
            data = df[[target_col, feature_col]].dropna()
            # Encode target to 0/1 if not already
            y = data[target_col]
            if y.dtype == "object" or y.dtype.name == "category":
                y = pd.Categorical(y).codes

            x = sm.add_constant(data[feature_col])

            try:
                model = sm.Logit(y, x).fit(disp=0)

                st.write(f"Modèle : **{target_col} ~ {feature_col}**")
                st.write(f"- Pseudo R-squared: {model.prsquared:.3f}")
                st.write(f"- p-value ({feature_col}): {model.pvalues[feature_col]:.3g}")

                if model.pvalues[feature_col] < 0.05:
                    st.success("Relation significative !")

                # Plot
                plot_data = data.copy()
                plot_data["y_encoded"] = y

                p = (
                    ggplot(plot_data, aes(x=feature_col, y="y_encoded"))
                    + geom_point(alpha=0.5)
                    + geom_smooth(
                        method="glm", method_args={"family": "binomial"}, color="red"
                    )
                    + theme_minimal()
                    + labs(
                        title=f"Régression Logistique : {target_col} vs {feature_col}",
                        y="Probabilité",
                    )
                )
                st.pyplot(p.draw())

                return f"Régression Logistique ({target_col} ~ {feature_col}) : Pseudo R2={model.prsquared:.3f}"

            except Exception as e:
                st.error(f"Erreur lors de l'ajustement du modèle : {e}")
                return None
        return None

    def generate_code(self, df_name: str = "df", **kwargs) -> str:
        target_col = kwargs.get("target_col", "Target")
        feature_col = kwargs.get("feature_col", "Feature")
        return f"""
# Régression Logistique
import statsmodels.api as sm
import pandas as pd
from plotnine import ggplot, aes, geom_point, geom_smooth, theme_minimal, labs

target_col = '{target_col}'
feature_col = '{feature_col}'

# Préparation
data = {df_name}[[target_col, feature_col]].dropna()
y = data[target_col]
# Encodage si nécessaire
if y.dtype == 'object' or y.dtype.name == 'category':
    y = pd.Categorical(y).codes
x = sm.add_constant(data[feature_col])

# Modèle
model = sm.Logit(y, x).fit(disp=0)
print(f"Pseudo R2: {{model.prsquared:.3f}}")

# Graphique
plot_data = data.copy()
plot_data['y_encoded'] = y

p = (ggplot(plot_data, aes(x=feature_col, y='y_encoded'))
     + geom_point(alpha=0.5)
     + geom_smooth(method="glm", method_args={{'family': 'binomial'}}, color="red")
     + theme_minimal()
     + labs(title=f"Régression Logistique : {{target_col}} vs {{feature_col}}", y="Probabilité")
    )
print(p)
"""


class TimeSeriesAnalysis(BaseAnalysis):
    @property
    def name(self) -> str:
        return "Analyse de Séries Temporelles"

    @property
    def category(self) -> str:
        return "Analyses temporelles"

    @property
    def description(self) -> str:
        return "Visualisation de l'évolution d'une variable numérique dans le temps."

    def check_applicability(self, df: pd.DataFrame) -> bool:
        # Check for datetime column
        # Heuristic: look for 'date' or 'time' in name or datetime dtype
        has_date = False
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                has_date = True
                break
            if "date" in col.lower() or "time" in col.lower() or "year" in col.lower():
                has_date = True
                break

        has_num = len(df.select_dtypes(include="number").columns) > 0
        return has_date and has_num

    def run(self, df: pd.DataFrame) -> Any:
        return None

    def render_streamlit(self, df: pd.DataFrame, result: Any) -> Optional[str]:
        st.write("### Analyse de Séries Temporelles")

        # Identify potential date columns
        date_candidates = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_candidates.append(col)
            elif (
                "date" in col.lower() or "time" in col.lower() or "year" in col.lower()
            ):
                date_candidates.append(col)

        num_cols = df.select_dtypes(include="number").columns

        col1, col2 = st.columns(2)
        with col1:
            date_col = st.selectbox(
                "Variable temporelle", date_candidates, key="ts_date"
            )
        with col2:
            val_col = st.selectbox("Variable à suivre", num_cols, key="ts_val")

        if date_col and val_col:
            # Ensure date is datetime
            plot_data = df[[date_col, val_col]].copy().dropna()
            try:
                plot_data[date_col] = pd.to_datetime(plot_data[date_col])
            except Exception:
                st.warning(f"Impossible de convertir {date_col} en date.")
                return None

            p = (
                ggplot(plot_data, aes(x=date_col, y=val_col))
                + geom_line(color="steelblue")
                + theme_minimal()
                + labs(title=f"Évolution de {val_col} dans le temps", x="Date")
            )
            st.pyplot(p.draw())

            return f"Série temporelle analysée pour {val_col}."
        return None

    def generate_code(self, df_name: str = "df", **kwargs) -> str:
        date_col = kwargs.get("date_col", "Date")
        val_col = kwargs.get("val_col", "Value")
        return f"""
# Analyse de Séries Temporelles
import pandas as pd
from plotnine import ggplot, aes, geom_line, theme_minimal, labs

date_col = '{date_col}'
val_col = '{val_col}'

# Conversion en date
plot_data = {df_name}[[date_col, val_col]].copy().dropna()
plot_data[date_col] = pd.to_datetime(plot_data[date_col])

# Graphique
p = (ggplot(plot_data, aes(x=date_col, y=val_col))
     + geom_line(color="steelblue")
     + theme_minimal()
     + labs(title=f"Évolution de {{val_col}} dans le temps", x="Date")
)
print(p)
"""
