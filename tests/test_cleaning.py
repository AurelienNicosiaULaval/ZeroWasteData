import unittest
import pandas as pd
import numpy as np
from utils.cleaning import (
    check_quality,
    clean_duplicates,
    impute_missing,
    convert_types,
)


class TestCleaning(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "A": [1, 2, 2, np.nan, 5],
                "B": ["a", "b", "b", "c", "d"],
                "C": ["10€", "20", "20", "30", "40"],
                "Date": [
                    "2021-01-01",
                    "2021-01-02",
                    "2021-01-02",
                    "invalid",
                    "2021-01-05",
                ],
            }
        )

    def test_check_quality(self):
        report = check_quality(self.df)
        self.assertEqual(report["n_rows"], 5)
        self.assertEqual(report["duplicates"], 1)
        self.assertEqual(report["missing_values"]["A"], 1)

    def test_clean_duplicates(self):
        df_clean = clean_duplicates(self.df)
        self.assertEqual(len(df_clean), 4)

    def test_impute_missing(self):
        df_imputed = impute_missing(self.df, strategy="mean", cols=["A"])
        self.assertFalse(df_imputed["A"].isnull().any())
        self.assertEqual(df_imputed.loc[3, "A"], 2.5)  # (1+2+2+5)/4 = 2.5

    def test_convert_types(self):
        df_converted = convert_types(self.df)
        self.assertTrue(pd.api.types.is_numeric_dtype(df_converted["C"]))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_converted["Date"]))
        self.assertEqual(df_converted.loc[0, "C"], 10)


if __name__ == "__main__":
    unittest.main()
