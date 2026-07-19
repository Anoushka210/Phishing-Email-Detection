"""
Batch Analysis - upload a CSV of emails, classify all of them, download results.
Expects a CSV with a column containing email text (auto-detects common
column names, or lets the user pick).
"""
import pandas as pd
import streamlit as st

from src.inference import load_text_models, predict_batch

st.set_page_config(page_title="Batch Analysis - PhishGuard", layout="wide")
st.title("Batch Email Analysis")


@st.cache_resource
def get_models():
    return load_text_models()


vectorizer, models = get_models()

uploaded_file = st.file_uploader("Upload a CSV of emails", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(f"Loaded {len(df)} rows.")
    st.dataframe(df.head())

    candidate_cols = [c for c in df.columns if "text" in c.lower() or "email" in c.lower() or "body" in c.lower()]
    default_col = candidate_cols[0] if candidate_cols else df.columns[0]
    text_col = st.selectbox("Which column contains the email text?", df.columns, index=list(df.columns).index(default_col))

    run_btn = st.button("Run Batch Classification", type="primary")

    if run_btn:
        with st.spinner(f"Classifying {len(df)} emails..."):
            results_df = predict_batch(df[text_col], vectorizer, models)

        df["prediction"] = results_df["hist_gradient_boosting_prediction"].map(
            {1: "Phishing", 0: "Legitimate"}
        )
        df["phishing_probability"] = results_df["hist_gradient_boosting_phishing_probability"]

        st.success(f"Done. {(df['prediction'] == 'Phishing').sum()} of {len(df)} flagged as phishing.")
        st.dataframe(df)

        csv_out = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results as CSV",
            data=csv_out,
            file_name="phishguard_batch_results.csv",
            mime="text/csv",
        )
else:
    st.info("Upload a CSV file to begin. It should have a column with email text.")