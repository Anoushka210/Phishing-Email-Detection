"""
Model Comparison - side-by-side metrics for all trained models, both the
text corpus experiment and the metadata-only structural experiment.
"""
import pandas as pd
import streamlit as st

from src.config import REPORTS_DIR

st.set_page_config(page_title="Model Comparison - PhishGuard", layout="wide")
st.title("Model Comparison")

tab1, tab2 = st.tabs(["Text Corpus Models", "Metadata-Only Models"])

with tab1:
    df = pd.read_csv(REPORTS_DIR / "model_comparison.csv")
    st.dataframe(df)  # removed .style.highlight_max() - triggers a pyarrow/Styler segfault

    metric = st.selectbox("Metric to visualize", ["accuracy", "precision", "recall", "f1", "roc_auc"], key="text_metric")
    st.bar_chart(df.set_index("model")[metric])

    st.subheader("False negatives (missed phishing) — the key operational metric")
    st.bar_chart(df.set_index("model")["false_negatives"])

with tab2:
    df_meta = pd.read_csv(REPORTS_DIR / "metadata_model_comparison.csv")
    st.dataframe(df_meta)  # removed .style.highlight_max() - same fix
    
    st.info(
        "Structural features alone (no email text) perform substantially "
        "weaker than the text-based models above — best F1 here is ~0.57 "
        "vs. ~0.98 for text-based models, confirming that email content "
        "carries far more phishing-discriminative signal than structural "
        "counts alone for this dataset."
    )

    metric_meta = st.selectbox("Metric to visualize", ["accuracy", "precision", "recall", "f1", "roc_auc"], key="meta_metric")
    st.bar_chart(df_meta.set_index("model")[metric_meta])