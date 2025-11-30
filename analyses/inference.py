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

    def render_streamlit(self, df: pd.DataFrame, result: Any) -> Optional[str]:
        st.write("### Test t de Student")

        num_cols = df.select_dtypes(include="number").columns
        cat_cols = [
            c
            for c in df.select_dtypes(include=["object", "category"]).columns
            if df[c].nunique() == 2
        ]

        col1, col2 = st.columns(2)
        with col1:
            group_col = st.selectbox(
                "Variable de groupe (2 niveaux)", cat_cols, key="ttest_grp"
            )
        with col2:
            val_col = st.selectbox("Variable numérique", num_cols, key="ttest_val")

        if group_col and val_col:
            groups = df[group_col].unique()
            g1 = df[df[group_col] == groups[0]][val_col].dropna()
            g2 = df[df[group_col] == groups[1]][val_col].dropna()

            t_stat, p_val = stats.ttest_ind(g1, g2)

            st.write(
                f"Comparaison de **{val_col}** par **{group_col}** ({groups[0]} vs {groups[1]})"
            )
            st.write(f"- t-statistic: {t_stat:.3f}")
            st.write(f"- p-value: {p_val:.3g}")

            if p_val < 0.05:
                st.success("Différence significative !")
            else:
                st.info("Pas de différence significative.")

            p = (
                ggplot(df, aes(x=group_col, y=val_col, fill=group_col))
                + geom_boxplot(alpha=0.7)
                + theme_minimal()
                + labs(title=f"Boxplot : {val_col} par {group_col}")
            )
            st.pyplot(p.draw())

            return f"Test t ({val_col} ~ {group_col}) : p-value={p_val:.3g}"
        return None


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

    def render_streamlit(self, df: pd.DataFrame, result: Any) -> Optional[str]:
        st.write("### ANOVA à un facteur")

        num_cols = df.select_dtypes(include="number").columns
        cat_cols = [
            c
            for c in df.select_dtypes(include=["object", "category"]).columns
            if df[c].nunique() > 2
        ]

        col1, col2 = st.columns(2)
        with col1:
            group_col = st.selectbox(
                "Variable de groupe (>2 niveaux)", cat_cols, key="anova_grp"
            )
        with col2:
            val_col = st.selectbox("Variable numérique", num_cols, key="anova_val")

        if group_col and val_col:
            groups = [
                df[df[group_col] == g][val_col].dropna() for g in df[group_col].unique()
            ]
            f_stat, p_val = stats.f_oneway(*groups)

            st.write(f"Comparaison de **{val_col}** par **{group_col}**")
            st.write(f"- F-statistic: {f_stat:.3f}")
            st.write(f"- p-value: {p_val:.3g}")

            if p_val < 0.05:
                st.success("Différence significative entre les groupes !")
            else:
                st.info("Pas de différence significative.")

            p = (
                ggplot(df, aes(x=group_col, y=val_col, fill=group_col))
                + geom_boxplot(alpha=0.7)
                + theme_minimal()
                + labs(title=f"Boxplot : {val_col} par {group_col}")
                + theme(axis_text_x=element_text(rotation=45, hjust=1))
            )
            st.pyplot(p.draw())

            return f"ANOVA ({val_col} ~ {group_col}) : p-value={p_val:.3g}"
        return None


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

    def render_streamlit(self, df: pd.DataFrame, result: Any) -> Optional[str]:
        st.write("### Test du Chi-carré")

        cat_cols = df.select_dtypes(include=["object", "category"]).columns

        col1, col2 = st.columns(2)
        with col1:
            var1 = st.selectbox("Variable 1", cat_cols, key="chi2_v1")
        with col2:
            var2 = st.selectbox(
                "Variable 2", [c for c in cat_cols if c != var1], key="chi2_v2"
            )

        if var1 and var2:
            contingency_table = pd.crosstab(df[var1], df[var2])
            chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)

            st.write(f"Association entre **{var1}** et **{var2}**")
            st.write(f"- Chi2: {chi2:.3f}")
            st.write(f"- p-value: {p_val:.3g}")

            if p_val < 0.05:
                st.success("Association significative !")
            else:
                st.info("Pas d'association significative.")

            # Plot bar chart
            p = (
                ggplot(df, aes(x=var1, fill=var2))
                + geom_bar(position="fill")
                + theme_minimal()
                + labs(title=f"Distribution de {var2} par {var1}", y="Proportion")
            )
            st.pyplot(p.draw())

            return f"Chi-carré ({var1} vs {var2}) : p-value={p_val:.3g}"
        return None
