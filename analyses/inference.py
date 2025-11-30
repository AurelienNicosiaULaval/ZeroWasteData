from typing import Optional, Any
import pandas as pd
import streamlit as st
from scipy import stats
from plotnine import (
    ggplot,
    aes,
    geom_boxplot,
    geom_bar,
    theme_minimal,
    labs,
    theme,
    element_text,
)
from .base import BaseAnalysis


class TTestAnalysis(BaseAnalysis):
    @property
    def name(self) -> str:
        return "Test t de Student"

    @property
    def category(self) -> str:
        return "Inférentielle"

    @property
    def description(self) -> str:
        return "Comparaison de moyennes entre deux groupes."

    def check_applicability(self, df: pd.DataFrame) -> bool:
        # Need at least 1 numeric and 1 categorical with 2 levels
        num_cols = df.select_dtypes(include="number").columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        has_binary_cat = any(df[c].nunique() == 2 for c in cat_cols)
        return len(num_cols) > 0 and has_binary_cat

    def run(self, df: pd.DataFrame) -> Any:
        return None

    def render_streamlit(
        self, df: pd.DataFrame, result: Any
    ) -> tuple[Optional[str], Optional[Any]]:
        st.write("### Test t de Student")

        num_cols = df.select_dtypes(include="number").columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns

        c1, c2 = st.columns(2)
        group_col = c1.selectbox(
            "Variable de Groupe (2 niveaux)", cat_cols, key="ttest_grp"
        )
        val_col = c2.selectbox("Variable Numérique", num_cols, key="ttest_val")

        if not group_col or not val_col:
            return None, None

        groups = df[group_col].unique()
        if len(groups) != 2:
            st.warning("La variable de groupe doit avoir exactement 2 niveaux.")
            return None, None

        g1 = df[df[group_col] == groups[0]][val_col].dropna()
        g2 = df[df[group_col] == groups[1]][val_col].dropna()

        t_stat, p_val = stats.ttest_ind(g1, g2)
        st.metric("p-value", f"{p_val:.4g}")

        if p_val < 0.05:
            st.success("Différence significative !")
        else:
            st.info("Pas de différence significative.")

        # Plot
        p = (
            ggplot(df, aes(x=group_col, y=val_col, fill=group_col))
            + geom_boxplot(alpha=0.7)
            + theme_minimal()
            + labs(title=f"Boxplot : {val_col} par {group_col}")
        )
        st.pyplot(p.draw())

        return f"Test t : {val_col} par {group_col}. p-value = {p_val:.4g}.", p

    def generate_code(self, df_name: str = "df", **kwargs) -> str:
        group_col = kwargs.get("group_col", "Group")
        val_col = kwargs.get("val_col", "Value")
        return f"""
# Test t de Student
from scipy import stats
from plotnine import ggplot, aes, geom_boxplot, theme_minimal, labs

group_col = '{group_col}'
val_col = '{val_col}'

groups = {df_name}[group_col].unique()
g1 = {df_name}[{df_name}[group_col] == groups[0]][val_col].dropna()
g2 = {df_name}[{df_name}[group_col] == groups[1]][val_col].dropna()

t_stat, p_val = stats.ttest_ind(g1, g2)
print(f"t-statistic: {{t_stat:.3f}}")
print(f"p-value: {{p_val:.3g}}")

# Graphique
p = (ggplot({df_name}, aes(x=group_col, y=val_col, fill=group_col))
     + geom_boxplot(alpha=0.7)
     + theme_minimal()
     + labs(title=f"Boxplot : {{val_col}} par {{group_col}}")
    )
print(p)
"""

    def generate_r_code(self, df_name: str = "df", **kwargs) -> str:
        group_col = kwargs.get("group_col", "Group")
        val_col = kwargs.get("val_col", "Value")
        return f"""
# Test t de Student (R)
library(ggplot2)

group_col <- "{group_col}"
val_col <- "{val_col}"

# Test t
t_test <- t.test(as.formula(paste(val_col, "~", group_col)), data = {df_name})
print(t_test)

# Graphique
ggplot({df_name}, aes_string(x = group_col, y = val_col, fill = group_col)) +
  geom_boxplot(alpha = 0.7) +
  theme_minimal() +
  labs(title = paste("Boxplot :", val_col, "par", group_col))
"""


class ANOVAAnalysis(BaseAnalysis):
    @property
    def name(self) -> str:
        return "ANOVA à un facteur"

    @property
    def category(self) -> str:
        return "Inférentielle"

    @property
    def description(self) -> str:
        return "Comparaison de moyennes entre plus de deux groupes."

    def check_applicability(self, df: pd.DataFrame) -> bool:
        # Need at least 1 numeric and 1 categorical with > 2 levels
        num_cols = df.select_dtypes(include="number").columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        has_multi_cat = any(df[c].nunique() > 2 for c in cat_cols)
        return len(num_cols) > 0 and has_multi_cat

    def run(self, df: pd.DataFrame) -> Any:
        return None

    def render_streamlit(
        self, df: pd.DataFrame, result: Any
    ) -> tuple[Optional[str], Optional[Any]]:
        st.write("### ANOVA à un facteur")

        num_cols = df.select_dtypes(include="number").columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns

        c1, c2 = st.columns(2)
        group_col = c1.selectbox(
            "Variable de Groupe (>2 niveaux)", cat_cols, key="anova_grp"
        )
        val_col = c2.selectbox("Variable Numérique", num_cols, key="anova_val")

        if not group_col or not val_col:
            return None, None

        groups = [
            df[df[group_col] == g][val_col].dropna() for g in df[group_col].unique()
        ]
        f_stat, p_val = stats.f_oneway(*groups)

        st.metric("p-value", f"{p_val:.4g}")

        if p_val < 0.05:
            st.success("Différence significative entre les groupes !")
        else:
            st.info("Pas de différence significative.")

        # Plot
        p = (
            ggplot(df, aes(x=group_col, y=val_col, fill=group_col))
            + geom_boxplot(alpha=0.7)
            + theme_minimal()
            + labs(title=f"Boxplot : {val_col} par {group_col}")
            + theme(axis_text_x=element_text(rotation=45, hjust=1))
        )
        st.pyplot(p.draw())

        return f"ANOVA : {val_col} par {group_col}. p-value = {p_val:.4g}.", p

    def generate_code(self, df_name: str = "df", **kwargs) -> str:
        group_col = kwargs.get("group_col", "Group")
        val_col = kwargs.get("val_col", "Value")
        return f"""
# ANOVA à un facteur
from scipy import stats
from plotnine import ggplot, aes, geom_boxplot, theme_minimal, labs, theme, element_text

group_col = '{group_col}'
val_col = '{val_col}'

groups = [{df_name}[{df_name}[group_col] == g][val_col].dropna() for g in {df_name}[group_col].unique()]
f_stat, p_val = stats.f_oneway(*groups)

print(f"F-statistic: {{f_stat:.3f}}")
print(f"p-value: {{p_val:.3g}}")

# Graphique
p = (ggplot({df_name}, aes(x=group_col, y=val_col, fill=group_col))
     + geom_boxplot(alpha=0.7)
     + theme_minimal()
     + labs(title=f"Boxplot : {{val_col}} par {{group_col}}")
     + theme(axis_text_x=element_text(rotation=45, hjust=1))
    )
print(p)
"""

    def generate_r_code(self, df_name: str = "df", **kwargs) -> str:
        group_col = kwargs.get("group_col", "Group")
        val_col = kwargs.get("val_col", "Value")
        return f"""
# ANOVA à un facteur (R)
library(ggplot2)

group_col <- "{group_col}"
val_col <- "{val_col}"

# ANOVA
anova_res <- aov(as.formula(paste(val_col, "~", group_col)), data = {df_name})
summary(anova_res)

# Graphique
ggplot({df_name}, aes_string(x = group_col, y = val_col, fill = group_col)) +
  geom_boxplot(alpha = 0.7) +
  theme_minimal() +
  labs(title = paste("Boxplot :", val_col, "par", group_col)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
"""


class ChiSquareAnalysis(BaseAnalysis):
    @property
    def name(self) -> str:
        return "Test du Chi-carré"

    @property
    def category(self) -> str:
        return "Inférentielle"

    @property
    def description(self) -> str:
        return "Test d'indépendance entre deux variables catégorielles."

    def check_applicability(self, df: pd.DataFrame) -> bool:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        return len(cat_cols) >= 2

    def run(self, df: pd.DataFrame) -> Any:
        return None

    def render_streamlit(
        self, df: pd.DataFrame, result: Any
    ) -> tuple[Optional[str], Optional[Any]]:
        st.write("### Test du Chi-carré")

        cat_cols = df.select_dtypes(include=["object", "category"]).columns

        c1, c2 = st.columns(2)
        var1 = c1.selectbox("Variable 1", cat_cols, key="chi2_v1")
        var2 = c2.selectbox(
            "Variable 2", [c for c in cat_cols if c != var1], key="chi2_v2"
        )

        if not var1 or not var2:
            return None, None

        contingency_table = pd.crosstab(df[var1], df[var2])
        chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)

        st.metric("p-value", f"{p_val:.4g}")

        if p_val < 0.05:
            st.success("Dépendance significative !")
        else:
            st.info("Indépendance (pas de lien significatif).")

        # Plot
        p = (
            ggplot(df, aes(x=var1, fill=var2))
            + geom_bar(position="fill")
            + theme_minimal()
            + labs(title=f"Distribution de {var2} par {var1}", y="Proportion")
        )
        st.pyplot(p.draw())

        return f"Chi-carré : {var1} vs {var2}. p-value = {p_val:.4g}.", p

    def generate_code(self, df_name: str = "df", **kwargs) -> str:
        var1 = kwargs.get("var1", "Variable1")
        var2 = kwargs.get("var2", "Variable2")
        return f"""
# Test du Chi-carré
from scipy import stats
import pandas as pd
from plotnine import ggplot, aes, geom_bar, theme_minimal, labs

var1 = '{var1}'
var2 = '{var2}'

# Table de contingence
contingency_table = pd.crosstab({df_name}[var1], {df_name}[var2])
chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)

print(f"Chi2: {{chi2:.3f}}")
print(f"p-value: {{p_val:.3g}}")

# Graphique
p = (ggplot({df_name}, aes(x=var1, fill=var2))
     + geom_bar(position="fill")
     + theme_minimal()
     + labs(title=f"Distribution de {{var2}} par {{var1}}", y="Proportion")
)
print(p)
"""

    def generate_r_code(self, df_name: str = "df", **kwargs) -> str:
        var1 = kwargs.get("var1", "Variable1")
        var2 = kwargs.get("var2", "Variable2")
        return f"""
# Test du Chi-carré (R)
library(ggplot2)

var1 <- "{var1}"
var2 <- "{var2}"

# Table de contingence et Test
tbl <- table({df_name}[[var1]], {df_name}[[var2]])
chisq_res <- chisq.test(tbl)
print(chisq_res)

# Graphique
ggplot({df_name}, aes_string(x = var1, fill = var2)) +
  geom_bar(position = "fill") +
  theme_minimal() +
  labs(title = paste("Distribution de", var2, "par", var1), y = "Proportion")
"""
