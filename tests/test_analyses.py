import unittest
import pandas as pd
import numpy as np
from analyses.manager import AnalysisManager
from analyses.stats_descriptives import OutlierAnalysis, DistributionAnalysis
from analyses.correlations import CorrelationAnalysis
from analyses.regressions import SimpleLinearRegressionAnalysis
from analyses.multivariate import PCAAnalysis
from analyses.inference import TTestAnalysis, ANOVAAnalysis, ChiSquareAnalysis
from analyses.advanced import LogisticRegressionAnalysis, TimeSeriesAnalysis


class TestAnalyses(unittest.TestCase):
    def setUp(self):
        # Create a sample dataframe
        np.random.seed(42)
        self.df = pd.DataFrame(
            {
                "A": np.random.normal(0, 1, 100),
                "B": np.random.normal(0, 1, 100),
                "C": np.random.choice(["x", "y"], 100),
                "D": np.random.choice(["a", "b", "c"], 100),
                "Date": pd.date_range(start="1/1/2022", periods=100),
            }
        )
        # Add some correlation
        self.df["E"] = self.df["A"] * 2 + np.random.normal(0, 0.1, 100)
        # Binary target
        self.df["Target"] = np.where(self.df["A"] > 0, 1, 0)

        self.manager = AnalysisManager()
        self.manager.register_analysis(OutlierAnalysis)
        self.manager.register_analysis(DistributionAnalysis)
        self.manager.register_analysis(CorrelationAnalysis)
        self.manager.register_analysis(SimpleLinearRegressionAnalysis)
        self.manager.register_analysis(PCAAnalysis)
        self.manager.register_analysis(TTestAnalysis)
        self.manager.register_analysis(ANOVAAnalysis)
        self.manager.register_analysis(ChiSquareAnalysis)
        self.manager.register_analysis(LogisticRegressionAnalysis)
        self.manager.register_analysis(TimeSeriesAnalysis)

    def test_manager_applicability(self):
        applicable = self.manager.get_applicable_analyses(self.df)
        names = [a.name for a in applicable]
        self.assertIn("Valeurs aberrantes", names)
        self.assertIn("Corrélation linéaire (Pearson)", names)
        self.assertIn("Test t de Student", names)
        self.assertIn("ANOVA à un facteur", names)
        self.assertIn("Test du Chi-carré", names)
        self.assertIn("Régression Logistique", names)
        self.assertIn("Analyse de Séries Temporelles", names)

    def test_outlier_analysis(self):
        analysis = OutlierAnalysis()
        self.df.loc[0, "A"] = 100
        result = analysis.run(self.df)
        self.assertIn("A", result)

    def test_correlation_analysis(self):
        analysis = CorrelationAnalysis()
        result = analysis.run(self.df)
        self.assertFalse(result.empty)

    def test_inference_analyses(self):
        # T-Test
        ttest = TTestAnalysis()
        self.assertTrue(ttest.check_applicability(self.df))

        # ANOVA
        anova = ANOVAAnalysis()
        self.assertTrue(anova.check_applicability(self.df))

        # Chi2
        chi2 = ChiSquareAnalysis()
        self.assertTrue(chi2.check_applicability(self.df))

    def test_advanced_analyses(self):
        # LogReg
        logreg = LogisticRegressionAnalysis()
        self.assertTrue(logreg.check_applicability(self.df))

        # TimeSeries
        ts = TimeSeriesAnalysis()
        self.assertTrue(ts.check_applicability(self.df))


if __name__ == "__main__":
    unittest.main()
