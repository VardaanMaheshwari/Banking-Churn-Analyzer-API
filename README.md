API(Render) URL -> https://banking-churn-analyzer-api.onrender.com
UI(Streamlit Cloud) URL -> https://banking-churn-analyzer-api-jeqd2zfxshqv7qemfss9zg.streamlit.app/
# Bank Customer Churn — Analysis & Modelling

End-to-end churn analysis on **10,000 customers** of a European retail bank
(France / Germany / Spain). The project takes a raw dataset through data-quality
auditing, EDA, feature engineering, and a six-model comparison, then serves the
winning model through a small FastAPI + Streamlit app.

The workflow deliberately follows the structure of a classic ML-churn project
(reusable EDA helpers → feature engineering → wide model roster → stacking
ensemble), adapted to the **bank** dataset rather than the telco one it's usually
demonstrated on.

## Dataset

`Churn_Modelling.csv` — 10,000 rows × 14 columns. Target: `Exited`
(1 = churned). Churn rate ≈ **20%**, so the classes are imbalanced and accuracy
is a misleading metric; the project optimises **ROC-AUC** and **PR-AUC**.

## Project structure

```
Bank-Churn-Analyzer-with-ML/
├── data/            Churn_Modelling.csv
├── notebooks/       Bank_Churn_Analysis.ipynb   ← the analysis, end to end
├── src/             feature_engineering.py       ← reusable EDA + FE helpers
├── api/             app.py + churn_model.joblib  ← FastAPI inference service
├── ui/              app.py                       ← Streamlit front-end
├── requirements.txt
└── LICENSE
```

## Method

1. **Inspection** — `check_data()` and `grab_col_names()` auto-profile the columns.
2. **Cleaning** — drop identifiers, remove duplicates, standardise categorical text.
3. **EDA** — churn rate by segment (country, products, activity, age) with visuals.
4. **Outliers & correlation** — IQR winsorization, correlation heatmap.
5. **Feature engineering** — bank-specific, hypothesis-driven features
   (`ZeroBalance`, `BalanceToSalary`, `ProductsPerTenure`, `EngagementScore`, age buckets).
6. **Imbalance** — **SMOTE applied *inside* the CV folds** (via an `imblearn`
   pipeline) so no synthetic data ever leaks into evaluation.
7. **Modelling** — six models compared, tuned lightly, judged on cross-validated ROC-AUC.

## Results

Cross-validated ROC-AUC (RepeatedStratifiedKFold, SMOTE in-fold):

| Model | ROC-AUC |
|---|---|
| Decision Tree | 0.807 ± 0.011 |
| Random Forest | 0.831 ± 0.006 |
| XGBoost | 0.852 ± 0.008 |
| LightGBM | 0.850 ± 0.009 |
| Stacking Ensemble | 0.853 ± 0.006 |
| **CatBoost** | **0.858 ± 0.006** |

**CatBoost** wins, with the stacking ensemble a close second — a useful finding:
the extra complexity of stacking buys almost nothing here, so the single
gradient-boosting model is the sensible deployment choice.

### Business takeaways
- **Re-engage inactive customers** — the most *actionable* driver (inactivity ≈ doubles churn risk).
- **Audit the 3–4 product segment** — its very high churn is counter-intuitive and hints at mis-sold bundles.
- **Germany-specific review** — the country gap points to a local pricing/service problem.

## Running it

```bash
pip install -r requirements.txt

# 1. Reproduce the analysis + retrain the model
jupyter notebook notebooks/Bank_Churn_Analysis.ipynb   # runs top-to-bottom, writes api/churn_model.joblib

# 2. Serve the model
cd api && uvicorn app:app --reload        # docs at http://127.0.0.1:8000/docs

# 3. Launch the UI (in a second terminal)
cd ui && streamlit run app.py
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "credit_score": 600, "geography": "Germany", "gender": "Female", "age": 52,
  "tenure": 1, "balance": 120000, "num_of_products": 3, "has_cr_card": 1,
  "is_active_member": 0, "estimated_salary": 95000
}'
```

## Tech stack

pandas · numpy · seaborn · matplotlib · missingno · scikit-learn · XGBoost ·
LightGBM · CatBoost · imbalanced-learn (SMOTE) · FastAPI · Streamlit

## License

MIT — see [LICENSE](LICENSE).
