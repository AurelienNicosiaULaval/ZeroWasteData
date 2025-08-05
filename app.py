"""Application Streamlit pour l'exploration automatique des données."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

from analyses.correlations import pearson_correlations
from analyses.regressions import simple_linear_regression
from analyses.stats_descriptives import detect_outliers_iqr
from analyses.clustering import kmeans_clustering
from utils.report import generate_report

DATA_DIR = Path(__file__).resolve().parent / "data"
CSV_ANALYSES = DATA_DIR / "table_analyses.csv"
JSON_ANALYSES = DATA_DIR / "table_analyses.json"


@st.cache_data
def load_analyses_map() -> List[Dict[str, str]]:
    """Charge la table des analyses depuis le JSON ou le CSV."""
    if JSON_ANALYSES.exists():
        with JSON_ANALYSES.open("r", encoding="utf-8") as f:
            return json.load(f)
    df = pd.read_csv(CSV_ANALYSES)
    records = []
    for row in df.to_dict(orient="records"):
        records.append(
            {
                "category": row["Catégorie"],
                "analysis": row["Type d'analyse"],
                "condition": row["Condition automatique à tester"],
                "result": row["Résultat d'intérêt potentiel à signaler"],
            }
        )
    with JSON_ANALYSES.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return records


def main() -> None:
    st.set_page_config(page_title="Données zéro déchet", layout="wide")
    st.title("♻️ Données zéro déchet")

    analyses_map = load_analyses_map()
    analysis_names = [a["analysis"] for a in analyses_map]

    uploaded_file = st.file_uploader("Uploader un fichier CSV ou Excel", type=["csv", "xlsx"])
    if uploaded_file is None:
        st.info("Veuillez sélectionner un fichier de données.")
        return

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Analyses déjà réalisées")
    done = st.multiselect("Sélectionner les analyses déjà effectuées", analysis_names)
    remaining = [a for a in analyses_map if a["analysis"] not in done]

    if st.button("Scanner"):
        sections: Dict[str, str] = {}

        if any(a["analysis"] == "Valeurs aberrantes" for a in remaining):
            st.write("### Valeurs aberrantes")
            results = {}
            for col in df.select_dtypes(include="number").columns:
                mask = detect_outliers_iqr(df, col)
                if mask.any():
                    results[col] = int(mask.sum())
                    st.write(f"{col}: {int(mask.sum())} outliers")
                    import matplotlib.pyplot as plt
                    plt.figure()
                    df[col].plot.box()
                    st.pyplot(plt.gcf())
            if results:
                sections["Valeurs aberrantes"] = "\n".join(f"- {k}: {v}" for k, v in results.items())

        if any(a["analysis"] == "Corrélation linéaire (Pearson)" for a in remaining):
            st.write("### Corrélations significatives")
            corr_df = pearson_correlations(df)
            if not corr_df.empty:
                st.dataframe(corr_df)
                sections["Corrélations"] = corr_df.to_markdown(index=False)
                import seaborn as sns
                import matplotlib.pyplot as plt
                plt.figure(figsize=(6, 4))
                sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, cmap="viridis")
                st.pyplot(plt.gcf())

        if any(a["analysis"] == "Régression linéaire simple" for a in remaining):
            st.write("### Régression linéaire simple")
            num_cols = df.select_dtypes(include="number").columns
            if len(num_cols) >= 2:
                x_col, y_col = num_cols[:2]
                pval, r2 = simple_linear_regression(df, x_col, y_col)
                st.write(f"Modèle {y_col} ~ {x_col} : p-value={pval:.3g}, R²={r2:.3f}")
                sections["Régression"] = f"{y_col} ~ {x_col} : p-value={pval:.3g}, R²={r2:.3f}"
                import seaborn as sns
                import matplotlib.pyplot as plt
                plt.figure()
                sns.regplot(x=df[x_col], y=df[y_col])
                st.pyplot(plt.gcf())

        if sections:
            html = generate_report(sections)
            st.download_button(
                "Télécharger rapport", data=html, file_name="rapport.html", mime="text/html"
            )


if __name__ == "__main__":
    main()
