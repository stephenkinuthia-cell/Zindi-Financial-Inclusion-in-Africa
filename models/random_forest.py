from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

def get_random_forest_model(preprocessor):
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced",
            random_state=42
        ))
    ])
