# PhishGuard — Data Card

## Note on data availability
Raw and processed CSV files are NOT included in this GitHub repository
(excluded via .gitignore to keep repo size manageable for Streamlit Cloud
deployment). They are provided separately as part of the capstone
submission zip. See README.md "Getting the data" section for exact
folder structure to recreate locally.

## Sources

| Source | Rows (raw) | Rows (clean) | Phishing/Legit split | License | Notes |
|---|---|---|---|---|---|
| naserabdullahalam/phishing-email-dataset (Kaggle) | 82,486 | 82,077 (post-dedup + empty-text drop) | 42,891 / 39,595 (~52/48) | GNU LGPL | Aggregates Enron, Ling, CEAS, Nazario, Nigerian Fraud, SpamAssassin. Text arrives pre-cleaned by the original compiler: lowercased, punctuation stripped, HTML removed. URLs/emails survive only as separated tokens (e.g. "http www tcigp com") or fused strings (e.g. "aokfbozza4pluscom") since punctuation was stripped before the `@`/`://` characters were removed. |
| ethancratchley/email-phishing-dataset (Kaggle) | 524,846 | 31,081 (stratified sample, post-dedup: all 6,949 phishing + ~24,132 legitimate) | Raw: 6,949 / 517,897 (1.3% phishing) | Confirm on Kaggle page before final submission | Contains NO raw email text — only precomputed structural/metadata features (num_words, num_links, num_urgent_keywords, etc). Used as a separate, standalone structural-only experiment (Phase 6), not merged with the text corpus. |

## Known limitations
- naserabdullahalam text is pre-sanitized by the original compiler before publishing — true URL/domain features (intact `http://`, `@`, IP addresses, shortener domains) are NOT recoverable. Structural features for this source are token-based approximations (`num_link_mentions`, `num_fused_domain_tokens`) rather than true parsed URLs.
- No sender/subject header data available for naserabdullahalam — `sender` and `subject` columns are null throughout. All features are content-derived, not header-derived.
- ethancratchley is severely imbalanced in its raw form (1.3% phishing) — addressed via stratified sampling for the clean version, and `class_weight="balanced"` during training (Phase 6).
- Both corpora are English-language only; Enron/SpamAssassin components skew toward corporate/US email style.
- Predominantly older phishing tactics (Nazario 2005, Enron 2004) — may not reflect modern phishing sophistication (e.g. no image-based phishing, no QR-code phishing).
- Metadata-only clean sample is 31,081 rows (19.57% phishing), not the originally intended 31,949/21.8% — duplicate removal was applied after stratified sampling, shifting the ratio slightly. Documented here rather than re-run, since the shift doesn't change the experiment's conclusions.

## Preprocessing steps (Phase 2)
- Dropped 408 exact duplicate emails from the text corpus
- HTML stripped via BeautifulSoup, whitespace normalized, lowercased, control characters removed
- Dropped rows that became empty after cleaning (HTML-only emails)

## Feature engineering (Phase 3)
- Text corpus: token-based structural features (urgency keyword count, link/domain token mentions, word/stopword counts, formatting ratios) since punctuation is pre-stripped from source text
- `capital_ratio` and `exclamation_count` are non-informative (~0) due to pre-cleaned source text — kept in output for transparency but not used as meaningful predictors
- TF-IDF fit on train split only (5,000 max features) to avoid vocabulary leakage into test evaluation

## Modeling (Phase 4)
- Models: Logistic Regression, Random Forest, Naive Bayes, HistGradientBoostingClassifier
- Originally used XGBoost, replaced with scikit-learn's HistGradientBoostingClassifier for platform independence — XGBoost requires a compiled `libomp` library not installed by default on macOS; HistGB has no external compiled dependency and gave statistically equivalent results (97.8% accuracy vs XGBoost's 97.7%, 78 false negatives vs 77)
- Best model: HistGradientBoosting — 97.80% accuracy, 0.997 ROC-AUC, 78 false negatives (missed phishing) out of ~8,569 phishing test emails
- False negatives prioritized as the key evaluation metric over raw accuracy, given the asymmetric cost of missing a phishing email vs. a false alarm

### Full text corpus model comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | False Negatives |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9769 | 0.9749 | 0.9809 | 0.9779 | 0.9969 | 164 |
| Random Forest | 0.9578 | 0.9397 | 0.9823 | 0.9605 | 0.9917 | 152 |
| Naive Bayes | 0.8807 | 0.8602 | 0.9211 | 0.8896 | 0.9430 | 676 |
| HistGradientBoosting | 0.9780 | 0.9677 | 0.9909 | 0.9792 | 0.9975 | 78 |

## Explainability (Phase 5)
- SHAP TreeExplainer for HistGradientBoosting (dense input required), SHAP LinearExplainer for Logistic Regression (sparse-compatible)
- Global importance and per-email sample explanation plots saved to `reports/figures/`


## Metadata-only experiment (Phase 6)
- Clean sample: 31,081 rows (19.57% phishing)
- Best performers: Random Forest (F1=0.573, ROC-AUC=0.871) and HistGradientBoosting (F1=0.573, ROC-AUC=0.863) — both show moderate but real structural signal
- Logistic Regression and Naive Bayes performed poorly under `class_weight="balanced"` (accuracy 50% and 33% respectively) — the balancing overcorrected given only 8 weak numeric features, causing very high false-positive rates. Included for completeness rather than as a meaningful comparison point.
- **Key finding:** structural metadata alone (F1 ~0.57, best AUC ~0.87) is substantially weaker than text+structural combined (F1 ~0.98, best AUC ~0.997) — supports the conclusion that email content carries far more phishing-discriminative signal than structural counts alone for this dataset.

### Full metadata-only model comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | False Negatives |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.4996 | 0.2709 | 0.9211 | 0.4186 | 0.6811 | 96 |
| Random Forest | 0.8493 | 0.6425 | 0.5173 | 0.5731 | 0.8712 | 587 |
| Naive Bayes | 0.3305 | 0.2181 | 0.9375 | 0.3539 | 0.6337 | 76 |
| HistGradientBoosting | 0.7722 | 0.4524 | 0.7812 | 0.5730 | 0.8629 | 266 |

## Dashboard (Phase 7)
- Streamlit multipage app: Live Checker, Batch Analysis, Model Comparison, Explainability
- Batch Analysis uses only the best model (HistGradientBoosting) for speed — vectorized batch prediction reduced 2,000-row processing from 2.3 min (naive per-row loop) to 20 sec (vectorized, single-model)
- Live Checker shows all four models' individual verdicts for full transparency/comparison
- No paid APIs used — all inference runs locally within the Streamlit process

## Environment notes
- Python 3.11, conda environment `phishguard`
- scikit-learn pinned to 1.6.1 (must match the version used to train saved models, or joblib load will warn/risk breaking)
- `pandas.Styler.highlight_max()` passed to `st.dataframe()` causes a segmentation fault on this environment (pyarrow's Styler-to-Arrow conversion crashes) — avoided in favor of plain `st.dataframe()`
- macOS: HistGradientBoostingClassifier has no special OS requirements (this was the reason for dropping XGBoost, which required Homebrew's `libomp`)

## Collection date
16 July 2026