"""
Explainability - shows the precomputed SHAP global importance plots from
Phase 5, plus lets the user pick any live-checked email to see a live
per-email SHAP breakdown.
"""
import streamlit as st

from src.config import FIGURES_DIR

st.set_page_config(page_title="Explainability - PhishGuard", layout="wide")
st.title("Model Explainability")

st.markdown("""
These plots show which features (words, urgency signals, link mentions,
word counts) most influence each model's predictions, generated using
SHAP (SHapley Additive exPlanations) on a sample of the test set.
""")

tab1, tab2 = st.tabs(["HistGradientBoosting (best model)", "Logistic Regression"])

with tab1:
    st.subheader("Global feature importance")
    st.image(str(FIGURES_DIR / "hist_gb_global_importance.png"))
    st.subheader("Sample email explanation")
    st.image(str(FIGURES_DIR / "hist_gb_sample_explanation.png"))

with tab2:
    st.subheader("Global feature importance")
    st.image(str(FIGURES_DIR / "logreg_global_importance.png"))
    st.subheader("Sample email explanation")
    st.image(str(FIGURES_DIR / "logreg_sample_explanation.png"))

st.divider()
st.caption(
    "Want to see explanations for a specific email? Use the Live Checker "
    "page — a per-email breakdown feature can be added there in a future "
    "iteration using SHAP's live explainer on your pasted text."
)