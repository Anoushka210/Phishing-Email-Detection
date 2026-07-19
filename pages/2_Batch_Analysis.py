"""
Batch Analysis - upload a CSV of emails, classify all of them, download results.
Expects a CSV with a column containing email text (auto-detects common
column names, or lets the user pick).
"""
import pandas as pd
import streamlit as st

from src.inference import load_text_models, predict_email

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
        progress = st.progress(0, text="Classifying emails...")
        predictions = []
        probabilities = []

        for i, text in enumerate(df[text_col].fillna("")):
            result = predict_email(str(text), vectorizer, models)
            best = result["model_results"]["hist_gradient_boosting"]
            predictions.append("Phishing" if best["prediction"] == 1 else "Legitimate")
            probabilities.append(round(best["phishing_probability"], 4))
            progress.progress((i + 1) / len(df), text=f"Classifying emails... {i+1}/{len(df)}")

        progress.empty()

        df["prediction"] = predictions
        df["phishing_probability"] = probabilities

        st.success(f"Done. {sum(p == 'Phishing' for p in predictions)} of {len(df)} flagged as phishing.")
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