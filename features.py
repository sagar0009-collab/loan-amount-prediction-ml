import numpy as np
import pandas as pd

RAW_COLUMNS = [
    "person_age", "person_gender", "person_education", "person_income",
    "person_emp_exp", "person_home_ownership", "loan_intent", "loan_int_rate",
    "cb_person_cred_hist_length", "credit_score", "previous_loan_defaults_on_file",
]

def add_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X[RAW_COLUMNS].copy()
    X["log_income"] = np.log1p(X["person_income"])
    X = X.drop(columns="person_income")
    X["inc_x_home_own"] = X["log_income"] * (X["person_home_ownership"] == "OWN")
    X["inc_x_default"] = X["log_income"] * (X["previous_loan_defaults_on_file"] == "Yes")
    X["inc_x_rate"] = X["log_income"] * X["loan_int_rate"]
    X["inc_x_score"] = X["log_income"] * X["credit_score"]
    X["inc_sq"] = X["log_income"] ** 2
    return X
