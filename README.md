# ZeroWasteData

Streamlit prototype for **zero-waste data exploration**: upload a dataset, declare which analyses you have already done, and the app suggests the remaining statistical analyses.

Built for teaching and reproducible workflows: it generates **Python and R code snippets** and lets you export a **HTML report** summarizing results and next steps.

## Live demo

- https://zerowastedata.streamlit.app/

## What it does

- Data ingest: CSV or Excel
- Data quality and typing: type conversion, missing values, duplicates
- Analysis catalogue (auto-selection based on variables):
  - descriptive statistics (distributions, outliers)
  - correlations
  - simple linear regression
  - principal component analysis (PCA)
  - inferential tests: t-test, ANOVA, chi-square
  - logistic regression
  - time series analysis
- Scoring utilities (see `utils/scoring.py`): demo power, zero-waste score, eco impact
- Report generation: export a downloadable HTML report
- Caching for performance (Streamlit cache)

## Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Python 3.10 recommended (project tested with 3.10).

## Run

```bash
streamlit run app.py
```

Then:

1. Upload a CSV or Excel file.
2. Declare which analyses you already ran.
3. Click **Scan** to run the remaining analyses.
4. Download the generated **HTML report**.

## Tests

```bash
pytest
```

## Project structure

- `app.py` — Streamlit app entry point
- `analyses/` — analysis modules (outliers, distributions, correlations, regression, PCA, hypothesis tests, time series)
- `utils/` — cleaning, scoring, report generation utilities
- `tests/` — Pytest suite
- `data/` — example datasets

## Roadmap (short)

- add more analysis types and diagnostics
- improve HTML report layout and narrative
- add export options (code bundles, CLI)
