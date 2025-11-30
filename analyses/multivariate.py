from typing import Optional, Any
import pandas as pd
import streamlit as st
from plotnine import ggplot, aes, geom_point, theme_minimal, labs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from .base import BaseAnalysis


class PCAAnalysis(BaseAnalysis):
    @property
    def name(self) -> str:
        return "Analyse en composantes principales (PCA)"

    @property
    def category(self) -> str:
        return "Analyses multivariées avancées"

    @property
    def description(self) -> str:
        return "Réduction de dimension pour visualiser la structure des données multivariées."

    def check_applicability(self, df: pd.DataFrame) -> bool:
        return len(df.select_dtypes(include="number").columns) >= 3

    def run(self, df: pd.DataFrame) -> Any:
        # Interactive
        return None

    def render_streamlit(self, df: pd.DataFrame, result: Any) -> Optional[str]:
        st.write("### Analyse en Composantes Principales (PCA)")

        num_cols = df.select_dtypes(include="number").columns
        selected_cols = st.multiselect(
            "Variables à inclure", num_cols, default=list(num_cols)[:5], key="pca_cols"
        )

        if len(selected_cols) < 2:
            st.warning("Veuillez sélectionner au moins 2 variables.")
            return None

        data = df[selected_cols].dropna()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        pca = PCA(n_components=2)
        components = pca.fit_transform(scaled_data)

        explained_variance = pca.explained_variance_ratio_

        st.write(
            f"Variance expliquée : PC1 ({explained_variance[0]:.1%}), PC2 ({explained_variance[1]:.1%})"
        )

        # Create DF for plotting
        pca_df = pd.DataFrame(data=components, columns=["PC1", "PC2"])

        p = (
            ggplot(pca_df, aes(x="PC1", y="PC2"))
            + geom_point(alpha=0.7, color="purple")
            + theme_minimal()
            + labs(
                title="Projection PCA",
                x=f"PC1 ({explained_variance[0]:.1%})",
                y=f"PC2 ({explained_variance[1]:.1%})",
            )
        )
        st.pyplot(p.draw())

        return f"PCA réalisée sur {len(selected_cols)} variables. Variance expliquée cumulée (2 axes) : {sum(explained_variance):.1%}."
