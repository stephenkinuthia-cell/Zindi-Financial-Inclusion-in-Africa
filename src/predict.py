## import libraries
import pandas as pd
import joblib

## Load model and predict
def main():
    model = joblib.load("artifacts/best_model.pkl")

    test_df = pd.read_csv("data/raw/Test.csv")

    # If you used derived features
    from preprocess import add_derived_features
    test_df = add_derived_features(test_df)

    predictions = model.predict(test_df)

    submission = pd.DataFrame({
        "unique_id": test_df["uniqueid"] + " x " + test_df["country"],
        "bank_account": predictions
    })

    submission.to_csv("outputs/submission.csv", index=False)
    print("Submission file saved.")


if __name__ == "__main__":
    main()
