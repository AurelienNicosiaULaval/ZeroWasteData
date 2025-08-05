# Données zéro déchet

Prototype d'application Streamlit pour explorer un jeu de données et proposer
les analyses statistiques encore non réalisées.

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

```bash
streamlit run app.py
```

1. Charger un fichier CSV ou Excel.
2. Indiquer les analyses déjà effectuées.
3. Cliquer sur "Scanner" pour lancer les analyses restantes.
4. Télécharger le rapport généré en HTML.

## Tests

```bash
pytest
```

Compatible Python 3.10.
