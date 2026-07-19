"""
Phase 6: Metadata-only structural experiment.

Trains the same four model types on the ethancratchley metadata-only
dataset (pre-computed structural features, no raw text). No TF-IDF here
- this is a deliberate comparison point: "can phishing be detected from
structure alone, without reading the content?"

This dataset is heavily imbalanced even after Phase 2's stratified
sampling (~21.8% phishing), so accuracy is not the primary metric here -
precision/recall/F1 and false_negatives matter more.
"""
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

from src.config import (
    PROCESSED_DIR, MODELS_DIR, REPORTS_DIR,
    RANDOM_SEED, TEST_SIZE, METADATA_FEATURE_COLS,
)


def load_data():
    path = PROCESSED_DIR / "metadata_only_clean.csv"
    df = pd.read_csv(path)
    return df


def evaluate_model(name, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }


def train_all():
    df = load_data()
    print(f"[train-metadata] loaded {len(df)} rows")
    print(f"[train-metadata] phishing rate: {df['label'].mean():.2%}")

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["label"], random_state=RANDOM_SEED
    )
    print(f"[train-metadata] train: {len(train_df)}, test: {len(test_df)}")

    X_train = train_df[METADATA_FEATURE_COLS].values
    X_test = test_df[METADATA_FEATURE_COLS].values
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=20, class_weight="balanced",
            random_state=RANDOM_SEED, n_jobs=-1
        ),
        "naive_bayes": GaussianNB(),  # dense numeric features, not MultinomialNB
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200, max_depth=6, learning_rate=0.1,
            class_weight="balanced", random_state=RANDOM_SEED
        ),
    }

    results = []
    for name, model in models.items():
        print(f"[training-metadata] {name}...")
        model.fit(X_train, y_train)

        model_path = MODELS_DIR / f"metadata_{name}.joblib"
        joblib.dump(model, model_path)
        print(f"[saved] {model_path}")

        metrics = evaluate_model(name, model, X_test, y_test)
        results.append(metrics)
        print(
            f"[{name}] acc={metrics['accuracy']:.4f} "
            f"prec={metrics['precision']:.4f} rec={metrics['recall']:.4f} "
            f"f1={metrics['f1']:.4f} auc={metrics['roc_auc']:.4f} "
            f"(missed phishing: {metrics['false_negatives']})"
        )

    results_df = pd.DataFrame(results)
    out_path = REPORTS_DIR / "metadata_model_comparison.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n[saved] comparison table -> {out_path}")
    print("\n=== Metadata-Only Model Comparison ===")
    print(results_df.to_string(index=False))

    return results_df


if __name__ == "__main__":
    train_all()