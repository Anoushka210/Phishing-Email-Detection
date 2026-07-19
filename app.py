"""
Explainable Multi-Model Phishing Email Detection
Home page: project overview + navigation.
"""
import streamlit as st

st.set_page_config(
    page_title="PhishGuard",
    page_icon=":shield:",
    layout="wide",
)

st.title("PhishGuard")
st.subheader("Explainable Multi-Model Phishing Email Detection")

st.markdown("""
This dashboard demonstrates a phishing email detection system built on
four machine learning models trained on real-world email corpora
(Enron, Nazario, CEAS, SpamAssassin, and others), combining TF-IDF text
analysis with content-derived structural features.

**Navigate using the sidebar:**

- **Live Checker** - paste an email and get an instant phishing risk assessment with model explanations
- **Batch Analysis** - upload a CSV of emails and classify them all at once
- **Model Comparison** - see how Logistic Regression, Random Forest, Naive Bayes, and HistGradientBoosting compare
- **Explainability** - understand *why* the models flag an email as phishing or legitimate
""")

st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Best model accuracy", "97.8%", "HistGradientBoosting")
with col2:
    st.metric("Training emails", "82,077")
with col3:
    st.metric("False negative rate", "~0.9%", "78 missed / 8,569 phishing")

st.caption(
    "Built with scikit-learn, SHAP, and Streamlit. "
    "No paid APIs used - all models run locally."
)