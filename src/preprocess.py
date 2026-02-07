## import libraries

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


## feature list
BINARY_COLS = [
    "cellphone_access",
    "gender_of_respondent",
    "location_type"
]

ORDINAL_COLS = ["education_level"]

ORDINAL_CATEGORIES = [[
    "No formal education",
    "Primary education",
    "Secondary education",
    "Vocational/Specialised training",
    "Tertiary education"
]]

ONEHOT_COLS = [
    "job_type",
    "marital_status",
    "relationship_with_head",
    "country"
]

NUMERIC_COLS = [
    "age_of_respondent",
    "household_size"
]

## processing pipeline
def build_preprocessor():
    binary_pipeline = Pipeline([
        ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore"))
    ])

    ordinal_pipeline = Pipeline([
        ("ordinal", OrdinalEncoder(categories=ORDINAL_CATEGORIES))
    ])

    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("binary", binary_pipeline, BINARY_COLS),
            ("ordinal", ordinal_pipeline, ORDINAL_COLS),
            ("onehot", OneHotEncoder(handle_unknown="ignore"), ONEHOT_COLS),
            ("numeric", numeric_pipeline, NUMERIC_COLS)
        ]
    )

    return preprocessor


## derived features
def add_derived_features(df):
    df = df.copy()
    df["is_head_or_spouse"] = df["relationship_with_head"].isin(
        ["Head of Household", "Spouse"]
    ).astype(int)

    df["high_education"] = df["education_level"].isin(
        ["Secondary education", "Vocational/Specialised training", "Tertiary education"]
    ).astype(int)

    df["urban"] = (df["location_type"] == "Urban").astype(int)

    return df
