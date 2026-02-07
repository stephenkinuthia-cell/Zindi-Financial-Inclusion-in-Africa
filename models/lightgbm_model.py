from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

def get_lightgbm_model(preprocessor):
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=42
        ))
    ])
