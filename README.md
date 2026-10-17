# Loan Amount Predictor

Predicts a loan amount from applicant details using **Linear Regression** (scikit-learn), served with **Streamlit**.

- Dataset: Kaggle "Loan Approval Classification Data" (taweilo), 45,000 rows
- Target: `loan_amnt`
- Features: age, gender, education, income (log), experience, home ownership, loan purpose, interest rate, credit history, credit score, previous defaults, plus a few income interactions
- Excluded on purpose: `loan_percent_income` (derived from the target, causes leakage) and `loan_status`

## Results (20% hold-out)
| Metric | Value |
|---|---|
| R² | 0.24 |
| MAE | ~$4,300 |
| RMSE | ~$5,550 |

The dataset has a weak relationship between applicant features and loan amount, so this is a rough estimator, not a lending tool.

## Run locally
```bash
pip install -r requirements.txt
python train.py        # optional, regenerates model.pkl
streamlit run app.py
```

## Files
- `app.py` Streamlit app
- `train.py` training script
- `features.py` feature engineering shared by training and app
- `loan_data.csv` dataset
- `model.pkl` trained model
