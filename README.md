# Healthcare Provider Fraud Detection

A Random Forest model that flags potentially fraudulent healthcare providers from Medicare claims data, built as a case study. The model works as a triage tool, it ranks providers by fraud probability so an investigation team can decide who to review first.

## Whats here

- `app.py` - Streamlit app. Shows predictions for the unseen provider set, and can score a new provider-level CSV against the trained model.
- `fraud_model.joblib` - the trained Random Forest.
- `model_features.joblib` - the 16 feature columns the model expects, in order.
- `Unseen_Predictions.csv` - model predictions for the 1353 unseen providers.
- `Fraud_Detection_Case_Study.html` - the full analysis notebook, readable in any browser. Covers the data merging, EDA, feature engineering, feature selection, model comparison and business recommendations.
- `requirements.txt` - pinned dependencies.

The raw claims data is not included here on purpose.

## Run locally

```
pip install -r requirements.txt
python -m streamlit run app.py
```

## Model summary

Three models were compared (Logistic Regression, Random Forest, XGBoost) with class weighting for the 9.4% fraud rate. All three land around 0.95-0.96 ROC-AUC. Random Forest was kept for practical reasons, no scaling needed and robust to the heavy outliers in the billing features. At the default 0.5 cutoff it gets Recall 0.851 and Precision 0.446 on a held-out validation set. The full reasoning is in the notebook.
