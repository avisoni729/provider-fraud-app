import streamlit as st
import pandas as pd
import joblib
import altair as alt

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

    with st.expander("Why are ~19% flagged when only ~9% of training providers were fraud?"):
        st.write(
            "**This is intentional, not drift.** The flag rate follows directly from the "
            "model's operating point. At **recall 0.851** and **precision 0.446**, with a "
            "**9.4% underlying fraud rate**, the expected flagged share works out to "
            "9.4% × 0.851 ÷ 0.446 ≈ **18%**, which is exactly where this lands. "
            "The model is deliberately run as a **high-recall triage tool**, it flags "
            "generously and lets investigators filter, rather than keeping the list short "
            "and missing real fraud."
        )

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
        "Upload a CSV with **one row per provider** and the **16 feature columns** the "
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
        "**Seven models** were tested on the same 16 provider-level features and the same "
        "held-out validation set of 1,082 providers. Class weighting handled the 9.4% fraud "
        "rate where the algorithm supports it, Naive Bayes, KNN and AdaBoost have no such "
        "option, which shows in their low recall."
    )

    scorecard = pd.DataFrame(
        {
            "Accuracy": [0.864, 0.887, 0.858, 0.940, 0.933, 0.904, 0.853],
            "Recall": [0.911, 0.851, 0.901, 0.436, 0.455, 0.743, 0.871],
            "Precision": [0.400, 0.446, 0.387, 0.846, 0.730, 0.490, 0.376],
            "F1-Score": [0.556, 0.585, 0.542, 0.575, 0.561, 0.591, 0.525],
            "ROC-AUC": [0.958, 0.956, 0.955, 0.954, 0.939, 0.937, 0.892],
        },
        index=["Logistic Regression", "Random Forest (chosen)", "XGBoost",
               "AdaBoost", "KNN", "Naive Bayes", "Decision Tree"],
    )

    def highlight_chosen(row):
        if "chosen" in row.name:
            return ["background-color: #dcebfa; font-weight: bold"] * len(row)
        return [""] * len(row)

    st.dataframe(scorecard.style.apply(highlight_chosen, axis=1).format("{:.3f}"),
                 use_container_width=True)
    st.caption(
        "**Random Forest was chosen for the best recall/robustness balance, the others are "
        "shown for comparison.** Decision Tree vs Random Forest shows what bagging buys, "
        "same tree family, one tree vs 200 averaged."
    )

    st.subheader("Why Random Forest")
    st.write(
        "All three models rank fraud almost equally well, the ROC-AUC scores are within "
        "**0.003** of each other. With ranking ability this close, raw performance couldn't "
        "separate them, **the deciding factors were robustness and practicality instead**."
    )
    st.write(
        "**Random Forest was kept for practical reasons.** It needs no feature scaling, and "
        "it isn't thrown off by the heavy outliers in the billing features, a few providers "
        "bill enormous amounts and that's exactly the kind of thing we want the model to "
        "handle rather than be distorted by."
    )
    st.info(
        "**Working assumption behind the cutoff:** both kinds of error cost money. A **missed "
        "fraud case** is an ongoing, compounding loss, the provider keeps billing fraudulently "
        "until caught. A **false alarm** costs a fixed amount of investigator time to rule out. "
        "The current **0.5 cutoff treats these two costs as roughly comparable**, which is a "
        "reasonable starting point for a triage tool."
    )
    st.write(
        "**The recall/precision balance is a separate decision from the model itself**, it's "
        "set by the probability cutoff, not the algorithm. At the current **0.5 cutoff** the "
        "model catches about **85% of fraud**, and roughly **4-5 of every 10 flags are real**. "
        "If the business later judges undetected fraud to be far more costly than a wasted "
        "investigation, the cutoff gets **lowered to favor recall**, and if investigator "
        "capacity tightens, it gets **raised to favor precision**, either way **without "
        "retraining or replacing the model**. A stronger precision headline is available "
        "at a higher cutoff, this was checked empirically during development, **the current "
        "point was chosen on purpose for triage**, not because it's the only one available."
    )
    st.write(
        "Generalization was checked directly, training recall **0.90** vs **0.851** on validation, "
        "training precision **0.45** vs **0.446**. The two are close, the model behaves almost the "
        "same on data it never saw, so it's **learning a pattern rather than memorizing providers**."
    )

    st.subheader("What the model relies on most")
    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    importance_chart = alt.Chart(importance_df).mark_bar(color="#0f6cbd").encode(
        x=alt.X("Importance:Q", axis=alt.Axis(labelFontSize=12, titleFontWeight="bold")),
        y=alt.Y("Feature:N", sort="-x", title=None,
                axis=alt.Axis(labelLimit=0, labelFontSize=13, labelFontWeight="bold")),
        tooltip=["Feature", alt.Tooltip("Importance:Q", format=".3f")],
    ).properties(height=480)
    st.altair_chart(importance_chart, use_container_width=True)

    st.write(
        "Billing intensity and claim volume dominate, **total_reimbursed**, "
        "**inpatient_claims**, and **max_reimbursed** together carry over half the "
        "importance. The model mostly learns from **how much and how often a provider bills**."
    )
