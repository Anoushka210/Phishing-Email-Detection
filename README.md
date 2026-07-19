# PhishGuard

**Explainable Multi-Model Phishing Email Detection with Interactive Risk Analysis**

A machine learning system that detects phishing emails using NLP-based
text analysis and content-derived structural features, with an
interactive Streamlit dashboard for live checking, batch analysis,
model comparison, and explainability.

Built entirely with free, local tools — no paid APIs used anywhere in
this project.

---

## Project structure
```
phishguard/
├── app.py                      # Streamlit entry point (Home page)
├── pages/                      # Dashboard pages
│   ├── 1_Live_Checker.py
│   ├── 2_Batch_Analysis.py
│   ├── 3_Model_Comparison.py
│   └── 4_Explainability.py
├── src/
│   ├── config.py                # Paths and constants
│   ├── data_merger.py           # Merge raw Kaggle sources into one schema
│   ├── check_data.py            # Sanity checks on merged data
│   ├── cleaning.py               # HTML stripping, normalization, dedup
│   ├── feature_engineering.py    # Structural feature extraction + TF-IDF prep
│   ├── train.py                  # Train/evaluate text corpus models
│   ├── train_metadata.py         # Train/evaluate metadata-only models
│   ├── explain.py                # SHAP explainability plots
│   └── inference.py              # Shared prediction pipeline for the dashboard
├── models/                     # Trained model files (.joblib) - NOT in GitHub, see below
├── reports/
│   ├── model_comparison.csv         # NOT in GitHub, see below
│   ├── metadata_model_comparison.csv # NOT in GitHub, see below
│   └── figures/                      # SHAP plots - NOT in GitHub, see below
├── data_card.md                 # Full documentation of sources, cleaning, results
├── requirements.txt
└── .streamlit/config.toml       # Theme + upload size config
```
---

## Getting the data

Raw and processed CSV files, trained model files, and SHAP figure
outputs are **not included in this GitHub repository** — they're
excluded via `.gitignore` to keep the repo lightweight for Streamlit
Cloud deployment.

**These files are provided separately in the capstone submission zip.**
To run the pipeline from scratch locally instead, place the following
manually-downloaded Kaggle datasets at these paths, then run the
pipeline scripts below in order:
```bash
data/raw/naserabdullahalam/phishing_email.csv
data/raw/ethancratchley/email_phishing_data.csv
```

- naserabdullahalam source: kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
- ethancratchley source: kaggle.com/datasets/ethancratchley/email-phishing-dataset

## Setup

```bash
conda create -n phishguard python=3.11 -y
conda activate phishguard
pip install -r requirements.txt
```

**macOS users:** no special OpenMP install needed — this project uses
scikit-learn's `HistGradientBoostingClassifier` instead of XGBoost
specifically to avoid the `libomp` compiled-library dependency that
XGBoost requires on Mac.

## Running the full pipeline (if rebuilding from raw data)

```bash
python -m src.data_merger          # Phase 1: merge raw sources
python -m src.check_data           # Phase 1: sanity check
python -m src.cleaning             # Phase 2: clean text
python -m src.feature_engineering  # Phase 3: extract features
python -m src.train                # Phase 4: train text corpus models
python -m src.train_metadata       # Phase 6: train metadata-only models
python -m src.explain              # Phase 5: generate SHAP explanations
```

## Running the dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Four pages available via the sidebar:

- **Live Checker** — paste an email, get an instant multi-model verdict with key signal breakdown
- **Batch Analysis** — upload a CSV, classify all rows, download results
- **Model Comparison** — metrics table + charts for both the text corpus and metadata-only experiments
- **Explainability** — SHAP global feature importance and sample explanations

## Key results

| Experiment | Best model | Accuracy | F1 | False negatives |
|---|---|---|---|---|
| Text corpus (TF-IDF + structural) | HistGradientBoosting | 97.80% | 0.979 | 78 / ~8,569 phishing |
| Metadata-only (structural features alone) | Random Forest | 84.93% | 0.573 | 587 / ~1,216 phishing |

**Key finding:** email content carries substantially more
phishing-discriminative signal than structural metadata alone for this
dataset — see `data_card.md` for full methodology and results.

## Known issues
- `pandas.Styler.highlight_max()` passed to `st.dataframe()` causes a
  segmentation fault in this environment (pyarrow's Styler-to-Arrow
  conversion crashes). Avoid pandas `.style` with `st.dataframe()`; use
  plain DataFrames or `st.column_config` for visual formatting instead.
- XGBoost was deliberately avoided in favor of
  `HistGradientBoostingClassifier` to keep the project free of compiled
  external library dependencies (XGBoost needs `libomp` on macOS,
  which isn't installed by default) — see `data_card.md` for the
  performance comparison confirming this substitution didn't cost
  accuracy.

## Full methodology, limitations, and dataset documentation

See `data_card.md` for complete details on data sources, cleaning
steps, feature engineering rationale, and full model comparison tables
for both experiments.