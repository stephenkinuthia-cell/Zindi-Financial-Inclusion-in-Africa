## import libraries 
import pandas as pd
import joblib

from preprocess import build_preprocessor, add_derived_features
from models import (
    get_logistic_model,
    get_random_forest_model,
    get_xgboost_model,
    get_lightgbm_model
)

from sklearn.metrics import recall_score

## load data
def load_data():
    df = pd.read_csv("../data/raw/Train.csv")
    df = add_derived_features(df)

    X = df.drop(columns=["bank_account"])
    y = df["bank_account"].map({"Yes": 1, "No": 0})

    return X, y

## Training and evaluating models
def train_models(X, y, preprocessor):
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()

    models = {
        "Logistic": get_logistic_model(preprocessor),
        "RandomForest": get_random_forest_model(preprocessor),
        "XGBoost": get_xgboost_model(preprocessor, scale_pos_weight),
        "LightGBM": get_lightgbm_model(preprocessor)
    }

    best_model = None
    best_recall = 0

    for name, model in models.items():
        model.fit(X, y)
        preds = model.predict(X)
        recall = recall_score(y, preds)

        print(f"{name} Recall: {recall:.4f}")

        if recall > best_recall:
            best_recall = recall
            best_model = model

    return best_model

## saving the best model
def main():
    X, y = load_data()
    preprocessor = build_preprocessor()

    best_model = train_models(X, y, preprocessor)

    joblib.dump(best_model, "artifacts/best_model.pkl")
    print("Best model saved.")


if __name__ == "__main__":
    main()
