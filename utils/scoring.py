import pandas as pd


def calculate_demo_power(df: pd.DataFrame) -> int:
    """
    Calcule la puissance de démonstration du jeu de données (0-100).
    Basé sur la variété des types de colonnes, le nombre de lignes, etc.
    """
    score = 0

    # Taille
    if len(df) > 100:
        score += 10
    if len(df) > 1000:
        score += 10

    # Colonnes
    num_cols = len(df.select_dtypes(include="number").columns)
    cat_cols = len(df.select_dtypes(include=["object", "category"]).columns)
    date_cols = 0
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower():
            date_cols += 1

    if num_cols >= 2:
        score += 20
    if cat_cols >= 1:
        score += 10
    if cat_cols >= 2:
        score += 10
    if date_cols >= 1:
        score += 20

    # Qualité (pas trop de manquants)
    missing_pct = df.isnull().mean().mean()
    if missing_pct < 0.1:
        score += 20

    return min(100, score)


def calculate_zero_waste_score(done_count: int, total_count: int) -> int:
    """
    Calcule le score Zero Waste (0-100) basé sur l'utilisation des analyses possibles.
    """
    if total_count == 0:
        return 0
    return int((done_count / total_count) * 100)


def calculate_eco_impact(df: pd.DataFrame) -> str:
    """
    Estime l'impact écologique (stockage).
    """
    bytes_usage = df.memory_usage(deep=True).sum()
    mb_usage = bytes_usage / (1024 * 1024)

    # Estimation très grossière : 1MB stocké = ~0.00001 gCO2/an (négligeable mais symbolique)
    # On va plutôt donner une équivalence ludique
    if mb_usage < 1:
        return f"{mb_usage:.2f} MB (Très léger 🍃)"
    elif mb_usage < 10:
        return f"{mb_usage:.2f} MB (Léger 🌱)"
    else:
        return f"{mb_usage:.2f} MB (Lourd 🌳)"
