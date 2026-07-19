"""
Live Checker - paste an email, get instant classification across all
four text models plus a plain-English risk summary.
"""
import streamlit as st

from src.inference import load_text_models, predict_email

st.set_page_config(page_title="Live Checker - PhishGuard", layout="wide")
st.title("Live Email Checker")


@st.cache_resource
def get_models():
    return load_text_models()


vectorizer, models = get_models()

email_text = st.text_area(
    "Paste email text here",
    height=250,
    placeholder="Paste the full email body (and subject, if available)...",
)

check_btn = st.button("Check Email", type="primary")

if check_btn and email_text.strip():
    with st.spinner("Analyzing..."):
        result = predict_email(email_text, vectorizer, models)

    st.divider()

    best_model_name = "hist_gradient_boosting"
    best_result = result["model_results"][best_model_name]
    is_phishing = best_result["prediction"] == 1
    confidence = best_result["phishing_probability"] if is_phishing else 1 - best_result["phishing_probability"]

    if is_phishing:
        st.error(f"**PHISHING DETECTED** — {confidence:.1%} confidence")
    else:
        st.success(f"**Looks Legitimate** — {confidence:.1%} confidence")

    st.subheader("Model-by-model breakdown")
    cols = st.columns(4)
    labels = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "naive_bayes": "Naive Bayes",
        "hist_gradient_boosting": "HistGradientBoosting (best)",
    }
    for col, (name, label) in zip(cols, labels.items()):
        r = result["model_results"][name]
        verdict = "Phishing" if r["prediction"] == 1 else "Legitimate"
        col.metric(label, verdict, f"{r['phishing_probability']:.1%} phishing prob.")

    st.subheader("Why this verdict — key signals detected")
    feats = result["structural_features"]
    signal_cols = st.columns(4)
    signal_cols[0].metric("Urgency keyword hits", feats["num_urgent_keywords"])
    signal_cols[1].metric("Link/domain mentions", feats["num_link_mentions"])
    signal_cols[2].metric("Fused domain tokens", feats["num_fused_domain_tokens"])
    signal_cols[3].metric("Word count", feats["num_words"])

    st.caption(
        "Fewer words and more urgency/link signals are typically "
        "associated with phishing in this model's training data."
    )

elif check_btn:
    st.warning("Please paste some email text first.")