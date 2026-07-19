"""
Phase 4: Model Training & Evaluation.

Trains Logistic Regression, Random Forest, Naive Bayes, and
HistGradientBoosting on the text corpus (TF-IDF + structural features
combined). TF-IDF is fit on the train split only, then applied to test,
to avoid vocabulary leakage. Saves trained models + vectorizer to
models/, and a metrics comparison table to reports/model_comparison.csv.

Note: HistGradientBoostingClassifier is used instead of XGBoost - it's
built into scikit-learn (no compiled external library dependency like
XGBoost's libomp requirement on macOS), which keeps the project fully
platform-independent while giving comparable gradient boosting
performance.
"""
import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

from src.config import (
    PROCESSED_DIR, MODELS_DIR, REPORTS_DIR,
    RANDOM_SEED, TEST_SIZE, TFIDF_MAX_FEATURES, STRUCTURAL_FEATURE_COLS,
)


def load_data():
    path = PROCESSED_DIR / "text_corpus_features.csv"
    df = pd.read_csv(path)
    df["text_for_vectorization"] = df["text_for_vectorization"].fillna("")
    return df


def build_features(df: pd.DataFrame, vectorizer: TfidfVectorizer, fit: bool):
    if fit:
        tfidf_matrix = vectorizer.fit_transform(df["text_for_vectorization"])
    else:
        tfidf_matrix = vectorizer.transform(df["text_for_vectorization"])

    structural_matrix = csr_matrix(df[STRUCTURAL_FEATURE_COLS].values)
    combined = hstack([tfidf_matrix, structural_matrix]).tocsr()
    return combined


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
    print(f"[train] loaded {len(df)} rows")

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["label"], random_state=RANDOM_SEED
    )
    print(f"[train] train: {len(train_df)}, test: {len(test_df)}")

    vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
    X_train = build_features(train_df, vectorizer, fit=True)
    X_test = build_features(test_df, vectorizer, fit=False)
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.joblib")
    print(f"[saved] tfidf_vectorizer.joblib (vocab size: {len(vectorizer.vocabulary_)})")

    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=20, random_state=RANDOM_SEED, n_jobs=-1
        ),
        "naive_bayes": MultinomialNB(),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200, max_depth=6, learning_rate=0.1, random_state=RANDOM_SEED
        ),
    }

    results = []
    for name, model in models.items():
        print(f"[training] {name}...")

        if name == "hist_gradient_boosting":
            # HistGradientBoostingClassifier requires dense input.
            # Cast to float32 to roughly halve memory vs float64,
            # standard practice for tree-based models.
            X_train_fit = X_train.toarray().astype("float32")
            X_test_fit = X_test.toarray().astype("float32")
        else:
            X_train_fit = X_train
            X_test_fit = X_test

        model.fit(X_train_fit, y_train)

        model_path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(model, model_path)
        print(f"[saved] {model_path}")

        metrics = evaluate_model(name, model, X_test_fit, y_test)
        results.append(metrics)
        print(
            f"[{name}] acc={metrics['accuracy']:.4f} "
            f"prec={metrics['precision']:.4f} rec={metrics['recall']:.4f} "
            f"f1={metrics['f1']:.4f} auc={metrics['roc_auc']:.4f} "
            f"(missed phishing: {metrics['false_negatives']})"
        )

    results_df = pd.DataFrame(results)
    out_path = REPORTS_DIR / "model_comparison.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n[saved] comparison table -> {out_path}")
    print("\n=== Model Comparison ===")
    print(results_df.to_string(index=False))

    return results_df


if __name__ == "__main__":
    train_all()