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


@st.cache_data
def fit_logistic_regression(df: pd.DataFrame, target_col: str, feature_col: str) -> Any:
    data = df[[target_col, feature_col]].dropna()
    y = data[target_col]
    if y.dtype == "object" or y.dtype.name == "category":
        y = pd.Categorical(y).codes
    x = sm.add_constant(data[feature_col])
    return sm.Logit(y, x).fit(disp=0)


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

    def render_streamlit(
        self, df: pd.DataFrame, result: Any
    ) -> tuple[Optional[str], Optional[Any]]:
        st.write("### Régression Logistique")

        num_cols = df.select_dtypes(include="number").columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns

        # Target should be binary (or categorical)
        c1, c2 = st.columns(2)
        target_col = c1.selectbox("Cible (Binaire)", cat_cols, key="logreg_y")
        feature_col = c2.selectbox("Feature (Numérique)", num_cols, key="logreg_x")

        if not target_col or not feature_col:
            return None, None

        try:
            model = fit_logistic_regression(df, target_col, feature_col)
            st.write(model.summary())

            # Plot
            data = df[[target_col, feature_col]].dropna()
            y = data[target_col]
            if y.dtype == "object" or y.dtype.name == "category":
                y = pd.Categorical(y).codes

            plot_data = data.copy()
            plot_data["y_encoded"] = y

            # Downsample for plotting if too large
            if len(plot_data) > 5000:
                plot_data = plot_data.sample(5000, random_state=42)
                st.caption(
                    "⚠️ Données échantillonnées (5000 points) pour accélérer l'affichage."
                )

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

            return (
                f"Régression Logistique ({target_col} ~ {feature_col}) : Pseudo R2={model.prsquared:.3f}",
                p,
            )

        except Exception as e:
            st.error(f"Erreur lors de l'ajustement du modèle : {e}")
            return None, None

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

    def generate_r_code(self, df_name: str = "df", **kwargs) -> str:
        target_col = kwargs.get("target_col", "Target")
        feature_col = kwargs.get("feature_col", "Feature")
        return f"""
# Régression Logistique (R)
library(ggplot2)

target_col <- "{target_col}"
feature_col <- "{feature_col}"

# Modèle
# Assurez-vous que la cible est un facteur ou 0/1
model <- glm(as.formula(paste(target_col, "~", feature_col)), data = {df_name}, family = binomial)
summary(model)

# Graphique
ggplot({df_name}, aes_string(x = feature_col, y = target_col)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "glm", method.args = list(family = "binomial"), color = "red") +
  theme_minimal() +
  labs(title = paste("Régression Logistique :", target_col, "vs", feature_col), y = "Probabilité")
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

    def render_streamlit(
        self, df: pd.DataFrame, result: Any
    ) -> tuple[Optional[str], Optional[Any]]:
        st.write("### Analyse de Séries Temporelles")

        # Find date cols
        date_cols = [
            c
            for c in df.columns
            if pd.api.types.is_datetime64_any_dtype(df[c]) or "date" in c.lower()
        ]
        num_cols = df.select_dtypes(include="number").columns

        if not date_cols:
            st.warning("Aucune colonne de date trouvée.")
            return None, None

        c1, c2 = st.columns(2)
        date_col = c1.selectbox("Colonne Date", date_cols, key="ts_date")
        val_col = c2.selectbox("Valeur", num_cols, key="ts_val")

        if not date_col or not val_col:
            return None, None

        # Plot
        plot_data = df[[date_col, val_col]].copy().dropna()
        # Ensure date is datetime
        plot_data[date_col] = pd.to_datetime(plot_data[date_col])

        p = (
            ggplot(plot_data, aes(x=date_col, y=val_col))
            + geom_line(color="steelblue")
            + theme_minimal()
            + labs(title=f"Évolution de {val_col} dans le temps", x="Date")
        )
        st.pyplot(p.draw())

        return f"Série temporelle : {val_col} vs {date_col}.", p

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

    def generate_r_code(self, df_name: str = "df", **kwargs) -> str:
        date_col = kwargs.get("date_col", "Date")
        val_col = kwargs.get("val_col", "Value")
        return f"""
# Analyse de Séries Temporelles (R)
library(ggplot2)
library(lubridate)

date_col <- "{date_col}"
val_col <- "{val_col}"

# Conversion en date (si nécessaire)
# {df_name}[[date_col]] <- ymd({df_name}[[date_col]])

# Graphique
ggplot({df_name}, aes_string(x = date_col, y = val_col)) +
  geom_line(color = "steelblue") +
  theme_minimal() +
  labs(title = paste("Évolution de", val_col, "dans le temps"), x = "Date")
"""
