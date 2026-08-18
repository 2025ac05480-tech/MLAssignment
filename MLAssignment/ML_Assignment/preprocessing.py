from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import NEGATIVE_LABEL, POSITIVE_LABEL, TARGET_COLUMN


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _dense_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def validate_and_encode_target(
    y: pd.Series,
    positive_label: str = POSITIVE_LABEL,
    negative_label: str = NEGATIVE_LABEL,
) -> pd.Series:
    normalized = y.astype(str).str.strip().str.lower()
    valid = {positive_label, negative_label}
    unexpected = sorted(set(normalized.unique()) - valid)
    if unexpected:
        raise ValueError(
            "Unexpected target labels found. "
            f"Expected only {sorted(valid)}, got {unexpected}."
        )
    return normalized.map({negative_label: 0, positive_label: 1}).astype(int)


def split_raw_features_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is missing from dataset.")

    X = df.drop(columns=[target_column]).copy()
    y = validate_and_encode_target(df[target_column])
    return X, y


def identify_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    return numeric_features, categorical_features


def build_preprocessor(
    X_train: pd.DataFrame,
    scale_numeric: bool,
) -> ColumnTransformer:
    numeric_features, categorical_features = identify_feature_types(X_train)

    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _dense_one_hot_encoder()),
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=numeric_steps), numeric_features),
            ("cat", Pipeline(steps=categorical_steps), categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor


def clean_uploaded_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_column_names(df)
    # Gracefully ignore accidental index columns created by CSV exports.
    unnamed_cols = [c for c in out.columns if c.lower().startswith("unnamed:")]
    if unnamed_cols:
        out = out.drop(columns=unnamed_cols)
    return out


def validate_feature_schema(
    df: pd.DataFrame,
    required_features: list[str],
) -> tuple[bool, list[str]]:
    missing = [c for c in required_features if c not in df.columns]
    return len(missing) == 0, missing


def split_optional_columns(
    df: pd.DataFrame,
    required_features: list[str],
    target_column: str = TARGET_COLUMN,
    id_candidates: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series | None, pd.Series | None]:
    id_candidates = id_candidates or []

    y_series: pd.Series | None = None
    if target_column in df.columns:
        y_series = validate_and_encode_target(df[target_column])

    id_series: pd.Series | None = None
    matched_id = next((c for c in id_candidates if c in df.columns), None)
    if matched_id is not None:
        id_series = df[matched_id].copy()

    X = df[required_features].copy()
    return X, y_series, id_series
