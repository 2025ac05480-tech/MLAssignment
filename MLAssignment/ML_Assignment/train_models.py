from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from config import (
    ARTIFACTS_DIR,
    DATASET_CANDIDATES,
    FEATURE_SCHEMA_PATH,
    METRIC_COLUMNS,
    METRICS_JSON_PATH,
    MODEL_COMPARISON_CSV_PATH,
    MODEL_REGISTRY,
    MODELS_DIR,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    RANDOM_STATE,
    SPLIT_METADATA_PATH,
    TARGET_COLUMN,
    TEST_DATA_CSV_PATH,
    TEST_LABELS_CSV_PATH,
    TEST_SIZE,
)
from preprocessing import (
    build_preprocessor,
    normalize_column_names,
    split_raw_features_target,
    validate_and_encode_target,
)


def _find_dataset_path() -> Path:
    for path in DATASET_CANDIDATES:
        if path.exists() and path.is_file():
            return path
    raise FileNotFoundError(
        "Source dataset not found. Expected one of: "
        + ", ".join(str(p) for p in DATASET_CANDIDATES)
    )


def _read_bank_dataset(path: Path) -> tuple[pd.DataFrame, str]:
    for sep in [";", ",", "\t", "|"]:
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] > 1:
                out = normalize_column_names(df)
                return out, sep
        except Exception:
            continue
    raise ValueError("Failed to parse dataset with common separators (; , tab |).")


def _build_estimator(model_name: str):
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    if model_name == "Decision Tree":
        return DecisionTreeClassifier(random_state=RANDOM_STATE)
    if model_name == "KNN":
        return KNeighborsClassifier()
    if model_name == "Naive Bayes":
        return GaussianNB()
    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    raise KeyError(f"Unknown model name: {model_name}")


def _evaluate_binary(y_true: pd.Series, y_pred, y_score) -> dict[str, float]:
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": float(roc_auc_score(y_true, y_score)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }


def _winner_row(summary_df: pd.DataFrame) -> pd.Series:
    ranked = summary_df.sort_values(
        by=["F1", "MCC", "AUC", "Accuracy"],
        ascending=False,
    )
    return ranked.iloc[0]


def _run_sanity_checks(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    summary_df: pd.DataFrame,
) -> None:
    for col in ["Accuracy", "AUC", "Precision", "Recall", "F1"]:
        if not ((summary_df[col] >= 0.0) & (summary_df[col] <= 1.0)).all():
            raise ValueError(f"Metric bounds failed for {col}.")
    if not ((summary_df["MCC"] >= -1.0) & (summary_df["MCC"] <= 1.0)).all():
        raise ValueError("Metric bounds failed for MCC.")

    for model_name, meta in MODEL_REGISTRY.items():
        model_path = MODELS_DIR / meta["artifact"]
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model artifact: {model_path}")
        loaded = joblib.load(model_path)
        sample_pred = loaded.predict(X_test.head(1))
        if len(sample_pred) != 1:
            raise ValueError(f"Prediction sanity check failed for {model_name}")

    if TARGET_COLUMN in X_test.columns:
        raise ValueError("Target leakage detected in X_test features.")
    if TARGET_COLUMN in pd.read_csv(TEST_DATA_CSV_PATH).columns:
        raise ValueError("Target leakage detected in test_data.csv.")
    if TEST_LABELS_CSV_PATH.exists():
        labels_df = pd.read_csv(TEST_LABELS_CSV_PATH)
        if labels_df.shape[0] != y_test.shape[0]:
            raise ValueError("Saved test labels row count mismatch.")


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = _find_dataset_path()
    df, separator = _read_bank_dataset(dataset_path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' not found.")

    X, y = split_raw_features_target(df, target_column=TARGET_COLUMN)

    if df.shape[0] < 500:
        raise ValueError("Dataset has fewer than 500 rows; assignment constraint not met.")
    if X.shape[1] < 12:
        raise ValueError(
            "Dataset has fewer than 12 predictor features; assignment constraint not met."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_test.to_csv(TEST_DATA_CSV_PATH, index=False)
    pd.DataFrame({TARGET_COLUMN: y_test}).to_csv(TEST_LABELS_CSV_PATH, index=False)

    feature_schema = {
        "required_raw_features": X.columns.tolist(),
        "target_column": TARGET_COLUMN,
        "optional_identifier_columns": [],
        "allowed_target_labels": [NEGATIVE_LABEL, POSITIVE_LABEL],
    }
    FEATURE_SCHEMA_PATH.write_text(json.dumps(feature_schema, indent=2), encoding="utf-8")

    split_metadata = {
        "dataset_filename": dataset_path.name,
        "dataset_path_note": "Original source dataset is kept unchanged at workspace root.",
        "separator": separator,
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "n_predictor_features": int(X.shape[1]),
        "target_column": TARGET_COLUMN,
        "target_labels": [NEGATIVE_LABEL, POSITIVE_LABEL],
        "class_distribution_raw": {
            k: int(v)
            for k, v in df[TARGET_COLUMN].astype(str).str.strip().str.lower().value_counts().items()
        },
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "stratified": True,
    }
    SPLIT_METADATA_PATH.write_text(json.dumps(split_metadata, indent=2), encoding="utf-8")

    model_results: dict[str, dict] = {}
    comparison_rows: list[dict[str, float | str]] = []

    for model_name, model_meta in MODEL_REGISTRY.items():
        estimator = _build_estimator(model_name)
        preprocessor = build_preprocessor(
            X_train=X_train,
            scale_numeric=bool(model_meta["scale_numeric"]),
        )
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", estimator),
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        if not hasattr(pipeline, "predict_proba"):
            raise ValueError(f"Model {model_name} does not support predict_proba.")
        y_score = pipeline.predict_proba(X_test)[:, 1]

        metrics = _evaluate_binary(y_test, y_pred, y_score)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()
        report = classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=[NEGATIVE_LABEL, POSITIVE_LABEL],
            output_dict=True,
            zero_division=0,
        )

        model_results[model_name] = {
            "artifact_file": model_meta["artifact"],
            "metrics": metrics,
            "confusion_matrix": cm,
            "classification_report": report,
            "n_test_rows": int(X_test.shape[0]),
        }

        row = {"Model": model_name}
        row.update(metrics)
        comparison_rows.append(row)

        joblib.dump(pipeline, MODELS_DIR / model_meta["artifact"])

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df = comparison_df[["Model", *METRIC_COLUMNS]]
    comparison_df.to_csv(MODEL_COMPARISON_CSV_PATH, index=False)

    winner = _winner_row(comparison_df)
    winner_payload = {
        "selected_model": str(winner["Model"]),
        "rule": "Highest F1; tie-breakers MCC, then AUC, then Accuracy.",
        "tie_break_order": ["F1", "MCC", "AUC", "Accuracy"],
    }

    full_metrics = {
        "dataset": {
            "name": "UCI Bank Marketing",
            "filename": dataset_path.name,
            "separator": separator,
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "predictor_feature_count": int(X.shape[1]),
            "target_column": TARGET_COLUMN,
            "target_labels": [NEGATIVE_LABEL, POSITIVE_LABEL],
        },
        "winner": winner_payload,
        "models": model_results,
    }
    METRICS_JSON_PATH.write_text(json.dumps(full_metrics, indent=2), encoding="utf-8")

    _run_sanity_checks(X_test, y_test, comparison_df)

    print("Dataset inspection summary")
    print(f"- File: {dataset_path}")
    print(f"- Shape: {df.shape}")
    print(f"- Target: {TARGET_COLUMN} -> {sorted(df[TARGET_COLUMN].astype(str).str.strip().str.lower().unique().tolist())}")
    print("\nModel comparison (held-out test set)")
    print(comparison_df.to_string(index=False))
    print("\nOverall winner")
    print(f"- {winner_payload['selected_model']}")
    print(f"- Rule: {winner_payload['rule']}")


if __name__ == "__main__":
    main()