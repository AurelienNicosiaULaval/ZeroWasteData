"""Application Streamlit pour l'exploration automatique des données."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from analyses.manager import AnalysisManager
from analyses.stats_descriptives import OutlierAnalysis, DistributionAnalysis
from analyses.correlations import CorrelationAnalysis
from analyses.regressions import SimpleLinearRegressionAnalysis
from analyses.multivariate import PCAAnalysis
from analyses.inference import TTestAnalysis, ANOVAAnalysis, ChiSquareAnalysis
from analyses.advanced import LogisticRegressionAnalysis, TimeSeriesAnalysis
from utils.report import generate_report, generate_pdf_report


def load_data(file) -> pd.DataFrame:
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)


def main() -> None:
    st.set_page_config(page_title="Données zéro déchet", layout="wide", page_icon="♻️")

    # --- Sidebar ---
    with st.sidebar:
        st.title("♻️ ZeroWasteData")
        st.markdown(
            """
            **Réutilisez vos données !**

            Cette application vous aide à explorer de nouvelles perspectives
            sur vos jeux de données existants.
            """
        )
        uploaded_file = st.file_uploader("Importer un fichier", type=["csv", "xlsx"])

    if uploaded_file is None:
        st.info(
            "👋 Bienvenue ! Veuillez importer un fichier CSV ou Excel dans la barre latérale pour commencer."
        )
        return

    try:
        df = load_data(uploaded_file)
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {e}")
        return

    # --- Sidebar: Column Selection ---
    with st.sidebar:
        st.markdown("---")
        st.subheader("Filtrage des colonnes")
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect(
            "Sélectionner les variables à conserver",
            options=all_columns,
            default=all_columns,
            help="Désélectionnez les colonnes que vous ne souhaitez pas inclure dans les analyses.",
        )

        if not selected_columns:
            st.warning("Veuillez sélectionner au moins une colonne pour continuer.")
            return

        df = df[selected_columns]

    # --- Analysis Manager Setup ---
    manager = AnalysisManager()
    # Descriptive
    manager.register_analysis(OutlierAnalysis)
    manager.register_analysis(DistributionAnalysis)
    # Correlation
    manager.register_analysis(CorrelationAnalysis)
    # Inference
    manager.register_analysis(TTestAnalysis)
    manager.register_analysis(ANOVAAnalysis)
    manager.register_analysis(ChiSquareAnalysis)
    # Predictive
    manager.register_analysis(SimpleLinearRegressionAnalysis)
    manager.register_analysis(LogisticRegressionAnalysis)
    # Advanced
    manager.register_analysis(PCAAnalysis)
    manager.register_analysis(TimeSeriesAnalysis)

    applicable_analyses = manager.get_applicable_analyses(df)
    applicable_names = [a.name for a in applicable_analyses]

    # --- Sidebar: Already Done ---
    with st.sidebar:
        st.markdown("---")
        st.subheader("Historique")
        already_done = st.multiselect(
            "Quelles analyses avez-vous déjà réalisées ?",
            options=applicable_names,
            help="Sélectionnez les analyses que vous avez déjà faites pour que nous puissions prioriser de nouvelles pistes.",
        )

    # Split analyses
    suggested_analyses = [a for a in applicable_analyses if a.name not in already_done]
    done_analyses = [a for a in applicable_analyses if a.name in already_done]

    # --- Main Layout ---
    st.title(f"Analyse de : {uploaded_file.name}")

    # Tabs
    tab_data, tab_suggestions, tab_report = st.tabs(
        ["📊 Données", "💡 Suggestions d'Analyses", "📝 Rapport"]
    )

    with tab_data:
        st.subheader("Aperçu des données")
        col1, col2, col3 = st.columns(3)
        col1.metric("Lignes", df.shape[0])
        col2.metric("Colonnes", df.shape[1])
        col3.metric(
            "Variables numériques", len(df.select_dtypes(include="number").columns)
        )

        st.dataframe(df.head())

        with st.expander("Informations détaillées"):
            st.write(df.dtypes.astype(str).to_frame(name="Type"))
            st.write("Valeurs manquantes :")
            st.write(df.isnull().sum().to_frame(name="Count"))

    with tab_suggestions:
        st.subheader("Analyses suggérées")
        st.markdown(
            "Basé sur la structure de vos données et votre historique, voici ce que nous vous conseillons d'explorer :"
        )

        # Helper to render analyses list
        def render_analyses_list(analyses_list, key_prefix):
            categories = {}
            for analysis in analyses_list:
                cat = analysis.category
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(analysis)

            for cat, analyses in categories.items():
                st.markdown(f"#### {cat}")
                for analysis in analyses:
                    with st.expander(f"📌 {analysis.name}"):
                        st.markdown(f"_{analysis.description}_")

                        # Run and Render
                        try:
                            result = analysis.run(df)
                            report_content = analysis.render_streamlit(df, result)

                            if report_content:
                                if st.button(
                                    f"Ajouter '{analysis.name}' au rapport",
                                    key=f"{key_prefix}_btn_{analysis.name}",
                                ):
                                    st.session_state.report_sections[analysis.name] = (
                                        report_content
                                    )
                                    st.success("Ajouté au rapport !")
                        except Exception as e:
                            st.error(
                                f"Une erreur est survenue lors de l'exécution de cette analyse : {e}"
                            )

        # Initialize session state for report sections if not exists
        if "report_sections" not in st.session_state:
            st.session_state.report_sections = {}

        if suggested_analyses:
            render_analyses_list(suggested_analyses, "sugg")
        else:
            st.info(
                "Vous avez exploré toutes les pistes suggérées pour ce jeu de données ! Bravo !"
            )

        if done_analyses:
            st.markdown("---")
            with st.expander("✅ Analyses déjà réalisées (cliquer pour voir)"):
                st.markdown(
                    "Ces analyses sont masquées car vous avez indiqué les avoir déjà faites."
                )
                render_analyses_list(done_analyses, "done")

    with tab_report:
        st.subheader("Génération de rapport")

        if st.session_state.report_sections:
            st.write(
                f"Sections incluses : {', '.join(st.session_state.report_sections.keys())}"
            )

            if st.button("Vider le rapport"):
                st.session_state.report_sections = {}
                st.rerun()

            html = generate_report(st.session_state.report_sections)
            st.download_button(
                "📥 Télécharger HTML",
                data=html,
                file_name="rapport_zerowaste.html",
                mime="text/html",
            )

            pdf = generate_pdf_report(st.session_state.report_sections)
            st.download_button(
                "📥 Télécharger PDF",
                data=pdf,
                file_name="rapport_zerowaste.pdf",
                mime="application/pdf",
            )

            st.markdown("---")
            st.markdown("### Prévisualisation")
            for title, content in st.session_state.report_sections.items():
                st.markdown(f"#### {title}")
                st.markdown(content)
        else:
            st.info(
                "Aucune analyse n'a encore été ajoutée au rapport. Allez dans l'onglet 'Suggestions' et cliquez sur 'Ajouter au rapport' pour les analyses qui vous intéressent."
            )


if __name__ == "__main__":
    main()
