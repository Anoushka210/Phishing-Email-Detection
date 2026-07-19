"""
Phase 5: Explainability.

Generates SHAP explanations for XGBoost (TreeExplainer) and
Logistic Regression (LinearExplainer) - both exact and fast, avoiding
the slow generic KernelExplainer on a 5000+ dimension TF-IDF matrix.

Produces:
  - Global feature importance plots (top words/features driving
    predictions overall) for both models.
  - Per-email explanation plots for a few sample test emails, showing
    which words/features pushed that specific prediction toward
    phishing or legitimate.

These feed directly into the dashboard's "why this was flagged" view
in Phase 7.
"""
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")  # no display backend needed, just save to file
import matplotlib.pyplot as plt
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split

from src.config import (
    PROCESSED_DIR, MODELS_DIR, FIGURES_DIR,
    RANDOM_SEED, TEST_SIZE, STRUCTURAL_FEATURE_COLS,
)


def load_test_split_and_features():
    """Rebuild the same train/test split and feature matrix used in train.py,
    so SHAP explanations line up with the models already saved to disk."""
    df = pd.read_csv(PROCESSED_DIR / "text_corpus_features.csv")
    df["text_for_vectorization"] = df["text_for_vectorization"].fillna("")

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["label"], random_state=RANDOM_SEED
    )

    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    tfidf_test = vectorizer.transform(test_df["text_for_vectorization"])
    structural_test = csr_matrix(test_df[STRUCTURAL_FEATURE_COLS].values)
    X_test = hstack([tfidf_test, structural_test]).tocsr()

    feature_names = list(vectorizer.get_feature_names_out()) + STRUCTURAL_FEATURE_COLS

    return X_test, test_df.reset_index(drop=True), feature_names


def explain_hist_gb(X_test, feature_names, n_samples=200):
    print("[explain] loading hist_gradient_boosting model...")
    model = joblib.load(MODELS_DIR / "hist_gradient_boosting.joblib")

    explainer = shap.TreeExplainer(model)

    sample_idx = np.random.RandomState(RANDOM_SEED).choice(
        X_test.shape[0], size=min(n_samples, X_test.shape[0]), replace=False
    )
    # HistGradientBoostingClassifier (and SHAP's TreeExplainer for it)
    # requires dense input - same reason as in train.py.
    X_sample = X_test[sample_idx].toarray().astype("float32")

    print(f"[explain] computing SHAP values for {X_sample.shape[0]} sampled rows...")
    shap_values = explainer.shap_values(X_sample)

    plt.figure()
    shap.summary_plot(
        shap_values, X_sample, feature_names=feature_names,
        show=False, max_display=20
    )
    plt.tight_layout()
    out_path = FIGURES_DIR / "hist_gb_global_importance.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[saved] {out_path}")

    return explainer, feature_names


def explain_logreg(X_test, feature_names, n_samples=200):
    print("[explain] loading logistic_regression model...")
    model = joblib.load(MODELS_DIR / "logistic_regression.joblib")

    # LinearExplainer needs a background dataset - use a small random
    # sample from X_test itself as an approximation.
    background_idx = np.random.RandomState(RANDOM_SEED).choice(
        X_test.shape[0], size=min(100, X_test.shape[0]), replace=False
    )
    explainer = shap.LinearExplainer(model, X_test[background_idx])

    sample_idx = np.random.RandomState(RANDOM_SEED + 1).choice(
        X_test.shape[0], size=min(n_samples, X_test.shape[0]), replace=False
    )
    X_sample = X_test[sample_idx]

    print(f"[explain] computing SHAP values for {X_sample.shape[0]} sampled rows...")
    shap_values = explainer.shap_values(X_sample)

    plt.figure()
    shap.summary_plot(
        shap_values, X_sample, feature_names=feature_names,
        show=False, max_display=20
    )
    plt.tight_layout()
    out_path = FIGURES_DIR / "logreg_global_importance.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[saved] {out_path}")

    return explainer, feature_names


def explain_single_email(explainer, X_test, test_df, feature_names, row_idx, model_name, dense=False):
    row = X_test[row_idx]
    if dense:
        row = row.toarray().astype("float32")
    shap_values = explainer.shap_values(row)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    true_label = "phishing" if test_df.loc[row_idx, "label"] == 1 else "legitimate"
    print(f"[explain] row {row_idx} true label: {true_label}")

    row_dense = row if dense else row.toarray()

    plt.figure()
    shap.force_plot(
        explainer.expected_value if not isinstance(explainer.expected_value, list)
        else explainer.expected_value[1],
        shap_values[0] if shap_values.ndim > 1 else shap_values,
        row_dense[0],
        feature_names=feature_names,
        matplotlib=True, show=False,
    )
    out_path = FIGURES_DIR / f"{model_name}_sample_explanation.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


def run_explainability():
    X_test, test_df, feature_names = load_test_split_and_features()
    print(f"[explain] test set: {X_test.shape[0]} rows, {X_test.shape[1]} features")

    hist_gb_explainer, _ = explain_hist_gb(X_test, feature_names)
    explain_single_email(hist_gb_explainer, X_test, test_df, feature_names, row_idx=0, model_name="hist_gb", dense=True)

    logreg_explainer, _ = explain_logreg(X_test, feature_names)
    explain_single_email(logreg_explainer, X_test, test_df, feature_names, row_idx=0, model_name="logreg")

    print("\n[explain] done - see reports/figures/ for all plots")


if __name__ == "__main__":
    run_explainability()