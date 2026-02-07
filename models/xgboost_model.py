from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

def get_xgboost_model(preprocessor, scale_pos_weight):
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42
        ))
    ])
