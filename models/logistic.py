from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def get_logistic_model(preprocessor):
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ])
