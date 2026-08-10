import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from preprocessing import create_preprocessor

DATA_PATH = "data/Telco_customer_churn.xlsx"
MODEL_PATH = "models/logistic_churn_model.joblib"


def load_data(path):
    return pd.read_excel(path)


def prepare_data(df):
    df = df.copy()

    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

    df["Total Charges"] = df["Total Charges"].fillna(0)

    columns_to_drop = [
        "CustomerID",
        "Churn Label",
        "Churn Score",
        "Churn Reason",
        "Count",
        "Country",
        "State",
        "City",
        "Zip Code",
        "Lat Long",
        "Latitude",
        "Longitude",
        "CLTV",
    ]

    X = df.drop(columns=["Churn Value"] + columns_to_drop)

    y = df["Churn Value"]

    return X, y


def train_model():
    df = load_data(DATA_PATH)

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    numeric_features = X_train.select_dtypes(include=["number"]).columns

    categorical_features = X_train.select_dtypes(exclude=["number"]).columns

    preprocessor = create_preprocessor(numeric_features, categorical_features)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)

    print(f"Modelo salvo em: {MODEL_PATH}")

    return (model, X_test, y_test)


if __name__ == "__main__":
    train_model()
