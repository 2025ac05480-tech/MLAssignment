from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
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

from config import (
    METRIC_COLUMNS,
    METRICS_JSON_PATH,
    MODEL_COMPARISON_CSV_PATH,
    MODEL_REGISTRY,
    MODELS_DIR,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    SPLIT_METADATA_PATH,
    TARGET_COLUMN,
)
from preprocessing import (
    clean_uploaded_dataframe,
    split_optional_columns,
    validate_feature_schema,
)


def _safe_read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_split_metadata() -> dict:
    return _safe_read_json(SPLIT_METADATA_PATH)


@st.cache_data
def load_metrics_payload() -> dict:
    return _safe_read_json(METRICS_JSON_PATH)


@st.cache_data
def load_feature_schema() -> dict:
    from config import FEATURE_SCHEMA_PATH

    return _safe_read_json(FEATURE_SCHEMA_PATH)


@st.cache_data
def load_model_comparison() -> pd.DataFrame:
    return pd.read_csv(MODEL_COMPARISON_CSV_PATH)


@st.cache_resource
def load_model(model_name: str):
    artifact = MODEL_REGISTRY[model_name]["artifact"]
    return joblib.load(MODELS_DIR / artifact)


def evaluate_binary(y_true: pd.Series, y_pred, y_score) -> dict[str, float]:
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": float(roc_auc_score(y_true, y_score)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }


def _render_metric_cards(metrics: dict[str, float], label: str) -> None:
    st.markdown(f"**{label}**")
    cols = st.columns(3)
    for idx, metric_name in enumerate(METRIC_COLUMNS):
        cols[idx % 3].metric(metric_name, f"{metrics[metric_name]:.4f}")


def _render_confusion_and_report(cm: list[list[int]], report: dict) -> None:
    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["No", "Yes"],
            yticklabels=["No", "Yes"],
            ax=ax,
        )
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.markdown("**Classification Report**")
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)


def _validate_artifacts_or_stop() -> None:
    needed_files = [
        METRICS_JSON_PATH,
        MODEL_COMPARISON_CSV_PATH,
        SPLIT_METADATA_PATH,
    ]
    needed_files.extend(MODELS_DIR / v["artifact"] for v in MODEL_REGISTRY.values())

    missing = [str(path.name) for path in needed_files if not path.exists()]
    if missing:
        st.error("Required model artifacts are missing.")
        st.info("Run: python train_models.py")
        st.write("Missing files:", missing)
        st.stop()


def main() -> None:
    st.set_page_config(
        page_title="UCI Bank Marketing Classification Dashboard",
        layout="wide",
    )

    st.title("UCI Bank Marketing Classification Dashboard")
    st.caption(
        "This application compares five classifiers and predicts whether a customer "
        "subscribes to a term deposit."
    )

    _validate_artifacts_or_stop()

    split_metadata = load_split_metadata()
    metrics_payload = load_metrics_payload()
    feature_schema = load_feature_schema()
    comparison_df = load_model_comparison()

    required_features = feature_schema["required_raw_features"]
    winner = metrics_payload["winner"]

    with st.sidebar:
        st.header("Project Overview")
        st.write("Project: Machine Learning Assignment 2")
        st.write("Dataset: UCI Bank Marketing")
        st.write("Models:")
        for model_name in MODEL_REGISTRY:
            st.write(f"- {model_name}")
        st.write("---")
        st.write("Evaluation split")
        st.write(f"Train rows: {split_metadata['train_rows']}")
        st.write(f"Test rows: {split_metadata['test_rows']}")
        st.write(f"Test size: {split_metadata['test_size']}")
        st.write("---")
        st.write("Usage")
        st.write("1. Upload a CSV with required feature columns.")
        st.write("2. Select a model.")
        st.write("3. Review metrics and predictions.")

    st.subheader("Model Comparison on Held-Out Test Split")
    st.dataframe(comparison_df, use_container_width=True)
    st.success(
        f"Overall winner: {winner['selected_model']} | Rule: {winner['rule']}"
    )
    st.info(
        "Different metrics capture different trade-offs. Use all six metrics "
        "before selecting a model for deployment decisions."
    )

    st.subheader("Upload Test Data")
    uploaded_file = st.file_uploader("Upload test data CSV", type=["csv"])

    uploaded_df = None
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            st.error("CSV encoding could not be read. Please save the file as UTF-8.")
            st.stop()
        except Exception:
            st.error("Unable to read uploaded CSV. Please upload a valid CSV file.")
            st.stop()

        if uploaded_df.empty:
            st.error("Uploaded CSV is empty. Please upload a file with rows.")
            st.stop()

        uploaded_df = clean_uploaded_dataframe(uploaded_df)
        st.write(f"Rows: {uploaded_df.shape[0]}, Columns: {uploaded_df.shape[1]}")
        st.dataframe(uploaded_df.head(10), use_container_width=True)

        valid, missing = validate_feature_schema(uploaded_df, required_features)
        if not valid:
            st.error("Uploaded CSV is missing required feature columns.")
            st.write("Missing columns:", missing)
            st.stop()

    st.subheader("Model Selection")
    model_name = st.selectbox(
        "Select classification model",
        list(MODEL_REGISTRY.keys()),
    )

    selected_payload = metrics_payload["models"][model_name]
    stored_metrics = selected_payload["metrics"]

    live_metrics = None
    live_cm = None
    live_report = None
    predictions_df = None

    if uploaded_df is not None:
        model = load_model(model_name)
        try:
            X_upload, y_upload, id_col = split_optional_columns(
                uploaded_df,
                required_features=required_features,
                target_column=TARGET_COLUMN,
                id_candidates=feature_schema.get("optional_identifier_columns", []),
            )
            y_pred = model.predict(X_upload)

            if not hasattr(model, "predict_proba"):
                st.warning("Selected model does not expose probabilities.")
                y_prob_pos = None
            else:
                y_prob_pos = model.predict_proba(X_upload)[:, 1]

            pred_labels = pd.Series(y_pred).map({0: NEGATIVE_LABEL, 1: POSITIVE_LABEL})
            predictions_df = pd.DataFrame({"Predicted Label": pred_labels})
            if y_prob_pos is not None:
                predictions_df["No Probability"] = 1 - y_prob_pos
                predictions_df["Yes Probability"] = y_prob_pos

            if id_col is not None:
                predictions_df.insert(0, id_col.name, id_col)

            if y_upload is not None and y_prob_pos is not None:
                live_metrics = evaluate_binary(y_upload, y_pred, y_prob_pos)
                cm = confusion_matrix(y_upload, y_pred, labels=[0, 1])
                live_cm = cm.tolist()
                live_report = classification_report(
                    y_upload,
                    y_pred,
                    labels=[0, 1],
                    target_names=[NEGATIVE_LABEL, POSITIVE_LABEL],
                    output_dict=True,
                    zero_division=0,
                )
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        except Exception:
            st.error("Prediction failed for uploaded data. Verify schema and values.")
            st.stop()

    st.subheader("Selected Model Metrics")
    if live_metrics is not None:
        _render_metric_cards(live_metrics, "Uploaded labeled-data metrics")
    else:
        _render_metric_cards(stored_metrics, "Held-out test metrics")

    st.subheader("Confusion Matrix and Classification Report")
    if live_cm is not None and live_report is not None:
        st.caption("Source: recalculated from uploaded labeled data.")
        _render_confusion_and_report(live_cm, live_report)
    else:
        st.caption("Source: stored held-out split evaluation artifacts.")
        _render_confusion_and_report(
            selected_payload["confusion_matrix"],
            selected_payload["classification_report"],
        )

    st.subheader("Predictions")
    if predictions_df is not None:
        st.dataframe(predictions_df, use_container_width=True)
        yes_count = int((predictions_df["Predicted Label"] == POSITIVE_LABEL).sum())
        no_count = int((predictions_df["Predicted Label"] == NEGATIVE_LABEL).sum())
        c1, c2 = st.columns(2)
        c1.metric("Predicted Yes", yes_count)
        c2.metric("Predicted No", no_count)

        csv_bytes = predictions_df.to_csv(index=False).encode("utf-8")
        file_name = f"predictions_{model_name.lower().replace(' ', '_')}.csv"
        st.download_button(
            label="Download predictions CSV",
            data=csv_bytes,
            file_name=file_name,
            mime="text/csv",
        )
    else:
        st.info("Upload a test CSV to generate predictions.")

    with st.expander("Model Interpretation Note"):
        st.write(
            "These predictions are educational outputs for the selected dataset and "
            "model. A probability score is not a guarantee. Metrics shown above are "
            "either from the held-out split or from uploaded labeled data, as labeled."
        )


if __name__ == "__main__":
    main()