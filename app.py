import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Provider Fraud Detection", layout="wide")

# link to the notebook (HTML) - filled in after the repo is up
NOTEBOOK_URL = ""

model = joblib.load("fraud_model.joblib")
feature_columns = joblib.load("model_features.joblib")

st.title("Healthcare Provider Fraud Detection")
st.write(
    "Random Forest model trained on Medicare claims data to flag potentially "
    "fraudulent providers. Built as a triage tool, the fraud probability is meant "
    "to help an investigation team decide who to review first."
)
if NOTEBOOK_URL:
    st.markdown(f"[Full analysis notebook]({NOTEBOOK_URL})")

tab1, tab2 = st.tabs(["Unseen Data Predictions", "Score New Data"])

with tab1:
    st.subheader("Predictions on the Unseen providers")
    predictions = pd.read_csv("Unseen_Predictions.csv")

    flagged = (predictions["PredictedClass"] == "Yes").sum()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total providers", len(predictions))
    col2.metric("Flagged as potential fraud", flagged)
    col3.metric("Flagged percentage", f"{flagged / len(predictions) * 100:.1f}%")

    show_flagged_only = st.checkbox("Show only flagged providers")
    table = predictions.sort_values("Probability", ascending=False)
    if show_flagged_only:
        table = table[table["PredictedClass"] == "Yes"]

    st.dataframe(table, use_container_width=True, hide_index=True)

    st.download_button(
        "Download predictions CSV",
        predictions.to_csv(index=False),
        file_name="Unseen_Predictions.csv",
        mime="text/csv",
    )

with tab2:
    st.subheader("Score a new provider-level CSV")
    st.write(
        "Upload a CSV with one row per provider and the 16 feature columns the "
        "model was trained on. A Provider ID column is optional but recommended. "
        "Download the template below to see the exact format."
    )

    template = pd.DataFrame(columns=["Provider"] + feature_columns)
    st.download_button(
        "Download blank template CSV",
        template.to_csv(index=False),
        file_name="feature_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload provider features CSV", type="csv")

    if uploaded is not None:
        new_data = pd.read_csv(uploaded)

        missing = [c for c in feature_columns if c not in new_data.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            proba = model.predict_proba(new_data[feature_columns])[:, 1]
            predicted = pd.Series((proba > 0.5).astype(int)).map({1: "Yes", 0: "No"})

            results = pd.DataFrame({
                "Provider": new_data["Provider"] if "Provider" in new_data.columns else range(len(new_data)),
                "Probability": proba.round(4),
                "PredictedClass": predicted,
            })

            flagged_new = (results["PredictedClass"] == "Yes").sum()
            st.write(f"{flagged_new} out of {len(results)} providers flagged as potential fraud.")
            st.dataframe(results.sort_values("Probability", ascending=False),
                         use_container_width=True, hide_index=True)

            st.download_button(
                "Download these predictions",
                results.to_csv(index=False),
                file_name="new_predictions.csv",
                mime="text/csv",
            )
