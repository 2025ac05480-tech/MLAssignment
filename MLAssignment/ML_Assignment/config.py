from __future__ import annotations

from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COLUMN = "y"
POSITIVE_LABEL = "yes"
NEGATIVE_LABEL = "no"

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR.parent

# Keep the original downloaded dataset unchanged; this path points to it.
DATASET_CANDIDATES = [
    WORKSPACE_DIR / "bank-full.csv",
    BASE_DIR / "bank-full.csv",
    BASE_DIR / "data" / "bank-full.csv",
]

MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
METRICS_JSON_PATH = BASE_DIR / "metrics.json"
MODEL_COMPARISON_CSV_PATH = BASE_DIR / "model_comparison.csv"
TEST_DATA_CSV_PATH = BASE_DIR / "test_data.csv"
TEST_LABELS_CSV_PATH = ARTIFACTS_DIR / "test_labels.csv"
FEATURE_SCHEMA_PATH = ARTIFACTS_DIR / "feature_schema.json"
SPLIT_METADATA_PATH = ARTIFACTS_DIR / "split_metadata.json"

METRIC_COLUMNS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

MODEL_REGISTRY = {
    "Logistic Regression": {
        "artifact": "logistic_regression.joblib",
        "scale_numeric": True,
    },
    "Decision Tree": {
        "artifact": "decision_tree.joblib",
        "scale_numeric": False,
    },
    "KNN": {
        "artifact": "knn.joblib",
        "scale_numeric": True,
    },
    "Naive Bayes": {
        "artifact": "naive_bayes.joblib",
        "scale_numeric": False,
    },
    "Random Forest": {
        "artifact": "random_forest.joblib",
        "scale_numeric": False,
    },
}
