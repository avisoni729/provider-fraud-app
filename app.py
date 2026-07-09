import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Provider Fraud Detection", page_icon="🩺", layout="wide")

NOTEBOOK_URL = "https://avisoni729.github.io/provider-fraud-app/Fraud_Detection_Case_Study.html"
REPO_URL = "https://github.com/avisoni729/provider-fraud-app"

model = joblib.load("fraud_model.joblib")
feature_columns = joblib.load("model_features.joblib")

st.title("Healthcare Provider Fraud Detection")
st.write(
    "A Random Forest model trained on Medicare claims data to flag potentially "
    "fraudulent providers. Built as a triage tool, the fraud probability helps an "
    "investigation team decide who to review first."
)

link1, link2, spacer = st.columns([1, 1, 3])
with link1:
    st.link_button("Full analysis notebook", NOTEBOOK_URL, use_container_width=True)
with link2:
    st.link_button("GitHub repository", REPO_URL, use_container_width=True)

st.divider()

tab1, tab2, tab3 = st.tabs(["Unseen Data Predictions", "Score New Data", "Model & Reasoning"])

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

with tab3:
    st.subheader("How the models compared")
    st.write(
        "Three models were trained on the same 16 provider-level features, with class "
        "weighting to handle the 9.4% fraud rate. All numbers below are from a held-out "
        "validation set of 1,082 providers the models never saw during training."
    )

    scorecard = pd.DataFrame(
        {
            "Accuracy": [0.864, 0.887, 0.858],
            "Recall": [0.911, 0.851, 0.901],
            "Precision": [0.400, 0.446, 0.387],
            "F1-Score": [0.556, 0.585, 0.542],
            "ROC-AUC": [0.958, 0.956, 0.955],
        },
        index=["Logistic Regression", "Random Forest (chosen)", "XGBoost"],
    )

    def highlight_chosen(row):
        if "chosen" in row.name:
            return ["background-color: #dcebfa; font-weight: bold"] * len(row)
        return [""] * len(row)

    st.dataframe(scorecard.style.apply(highlight_chosen, axis=1).format("{:.3f}"),
                 use_container_width=True)

    st.subheader("Why Random Forest")
    st.write(
        "All three models rank fraud almost equally well, the ROC-AUC scores are within "
        "0.003 of each other. So the choice wasn't about raw performance."
    )
    st.write(
        "Random Forest was kept for practical reasons. It needs no feature scaling, and "
        "it isn't thrown off by the heavy outliers in the billing features, a few providers "
        "bill enormous amounts and that's exactly the kind of thing we want the model to "
        "handle rather than be distorted by."
    )
    st.write(
        "The recall/precision balance is a separate decision from the model itself, it's "
        "set by the probability cutoff. At the current 0.5 cutoff the model catches about "
        "85% of fraud, and roughly 4-5 of every 10 flags are real. If the business decides "
        "missing fraud costs more than a wasted investigation, the cutoff gets lowered to "
        "catch more, without retraining or replacing the model."
    )
    st.write(
        "Generalization was checked directly, training recall 0.90 vs 0.851 on validation, "
        "training precision 0.45 vs 0.446. The two are close, the model behaves almost the "
        "same on data it never saw, so it's learning a pattern rather than memorizing providers."
    )

    st.subheader("What the model relies on most")
    importance = pd.Series(model.feature_importances_, index=feature_columns).sort_values()
    st.bar_chart(importance, horizontal=True, height=450)
    st.caption(
        "Billing intensity and claim volume dominate, total reimbursed, inpatient claim "
        "count, and the single largest claim. The model mostly learns from how much and "
        "how often a provider bills."
    )
