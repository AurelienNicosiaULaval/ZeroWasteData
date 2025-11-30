import pandas as pd
from typing import Dict, List, Any


def check_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyse la qualité des données."""
    report = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": df.duplicated().sum(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
    return report


def clean_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les doublons complets."""
    return df.drop_duplicates()


def impute_missing(
    df: pd.DataFrame, strategy: str = "mean", cols: List[str] = None
) -> pd.DataFrame:
    """Impute les valeurs manquantes."""
    df_clean = df.copy()

    if cols is None:
        cols = df.select_dtypes(include="number").columns

    for col in cols:
        if col in df_clean.columns:
            if strategy == "mean":
                val = df_clean[col].mean()
            elif strategy == "median":
                val = df_clean[col].median()
            elif strategy == "mode":
                val = df_clean[col].mode()[0]
            elif strategy == "drop":
                df_clean = df_clean.dropna(subset=[col])
                continue
            else:
                val = 0

            df_clean[col] = df_clean[col].fillna(val)

    return df_clean


def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """Tente de convertir les types de colonnes."""
    df_clean = df.copy()

    # Try to convert to numeric
    for col in df_clean.select_dtypes(include="object").columns:
        try:
            # Check if it looks like a number (e.g. "1,000", "12€")
            # Simple heuristic: remove common non-numeric chars and try conversion
            cleaned_col = (
                df_clean[col].astype(str).str.replace(r"[€$£, ]", "", regex=True)
            )
            df_clean[col] = pd.to_numeric(cleaned_col)
        except (ValueError, TypeError):
            pass

    # Try to convert to datetime
    for col in df_clean.select_dtypes(include="object").columns:
        if "date" in col.lower() or "time" in col.lower():
            try:
                # Use errors='coerce' to handle mixed formats/invalid values
                df_clean[col] = pd.to_datetime(df_clean[col], errors="coerce")
            except (ValueError, TypeError):
                pass

    return df_clean
