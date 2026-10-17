import pandas as pd, numpy as np, joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from features import add_features


def train_model(csv_path="loan_data.csv", verbose=False):
    df = pd.read_csv(csv_path)

    # Remove unrealistic outliers
    df = df[(df.person_age <= 80) & (df.person_emp_exp <= 60) & (df.person_income <= 500000)]
    df = df[df.person_age - df.person_emp_exp >= 14]

    # loan_percent_income and loan_status are excluded on purpose (target leakage)
    X = add_features(df)
    y = df["loan_amnt"]

    cat = X.select_dtypes(exclude="number").columns.tolist()
    num = X.select_dtypes(include="number").columns.tolist()
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat),
    ])
    model = Pipeline([("pre", pre), ("lr", LinearRegression())])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(Xtr, ytr)
    if verbose:
        p = model.predict(Xte)
        print("R2  :", round(r2_score(yte, p), 4))
        print("MAE :", round(mean_absolute_error(yte, p), 1))
        print("RMSE:", round(float(np.sqrt(mean_squared_error(yte, p))), 1))
    return model


if __name__ == "__main__":
    joblib.dump(train_model(verbose=True), "model.pkl")
