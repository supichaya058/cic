from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


# ============================================================
# STAGE 017 — CIC-IDS2017 CHAPTER 4 VISUALIZATION
# ============================================================
#
# PURPOSE:
#   Create figures for Chapter 4.8.1
#
# FIGURES:
#   1. Accuracy / Precision / Recall / F1-score comparison
#   2. ROC Curve
#   3. Precision-Recall Curve
#
# INPUT:
#   data/models/cic_final_test_metrics_10s.csv
#   data/models/cic_final_test_predictions_10s.csv
#
# OUTPUT:
#   data/models/visualizations_ch4_cic_10s/
#
# IMPORTANT:
#   Current CIC experiment uses:
#       - Final Test = 2,512 Windows
#       - Normal = 2,210
#       - Attack = 302
#
#   This script stops if the result file is not the expected
#   2,512-window Final Test result, to prevent using an old run.
# ============================================================


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "data" / "models"

METRICS_PATH = MODEL_DIR / "cic_final_test_metrics_10s.csv"
PREDICTIONS_PATH = MODEL_DIR / "cic_final_test_predictions_10s.csv"

OUTPUT_DIR = MODEL_DIR / "visualizations_ch4_cic_10s"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. EXPECTED CURRENT EXPERIMENT
# ============================================================

EXPECTED_FINAL_TEST_ROWS = 2512
EXPECTED_NORMAL_ROWS = 2210
EXPECTED_ATTACK_ROWS = 302

MODEL_ORDER = [
    "Isolation Forest",
    "One-Class SVM",
    "LOF",
    "Statistical Threshold",
]

MODEL_SHORT_NAMES = {
    "Isolation Forest": "if",
    "One-Class SVM": "ocsvm",
    "LOF": "lof",
    "Statistical Threshold": "statistical",
}


# ============================================================
# 3. REQUIRED COLUMNS
# ============================================================

REQUIRED_METRIC_COLUMNS = [
    "model",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
]

REQUIRED_PREDICTION_COLUMNS = [
    "true_binary_label",
    "if_anomaly_score",
    "ocsvm_anomaly_score",
    "lof_anomaly_score",
    "statistical_anomaly_score",
]


# ============================================================
# 4. UTILITIES
# ============================================================

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result.columns = (
        result.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    duplicated = (
        result.columns[result.columns.duplicated()]
        .tolist()
    )

    if duplicated:
        raise ValueError(
            f"พบชื่อคอลัมน์ซ้ำ: {duplicated}"
        )

    return result


def check_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์:\n{path}"
        )


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"[SAVED] {path}")


# ============================================================
# 5. LOAD METRICS
# ============================================================

def load_metrics() -> pd.DataFrame:

    print("\n[1] LOAD FINAL TEST METRICS")
    print("-" * 72)

    check_file(METRICS_PATH)

    metrics = pd.read_csv(
        METRICS_PATH,
        encoding="utf-8-sig",
    )

    metrics = clean_columns(metrics)

    missing = [
        column
        for column in REQUIRED_METRIC_COLUMNS
        if column not in metrics.columns
    ]

    if missing:
        raise ValueError(
            "Metrics file ขาดคอลัมน์:\n"
            + "\n".join(missing)
        )

    metrics["model"] = (
        metrics["model"]
        .astype(str)
        .str.strip()
    )

    found_models = metrics["model"].tolist()

    missing_models = [
        model
        for model in MODEL_ORDER
        if model not in found_models
    ]

    if missing_models:
        raise ValueError(
            "Metrics file ขาดแบบจำลอง:\n"
            + "\n".join(missing_models)
        )

    metrics = (
        metrics[
            metrics["model"].isin(MODEL_ORDER)
        ]
        .copy()
    )

    metrics["model"] = pd.Categorical(
        metrics["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )

    metrics = (
        metrics
        .sort_values("model")
        .reset_index(drop=True)
    )

    numeric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]

    for column in numeric_columns:
        metrics[column] = pd.to_numeric(
            metrics[column],
            errors="coerce",
        )

        if metrics[column].isna().any():
            raise ValueError(
                f"คอลัมน์ {column} มีค่าไม่ถูกต้อง"
            )

    print(
        f"[PASS] Metrics rows: {len(metrics)}"
    )

    print(
        "\nMODEL METRICS"
    )

    print(
        metrics[
            [
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
            ]
        ].to_string(index=False)
    )

    return metrics


# ============================================================
# 6. LOAD PREDICTIONS
# ============================================================

def load_predictions() -> pd.DataFrame:

    print("\n[2] LOAD FINAL TEST PREDICTIONS")
    print("-" * 72)

    check_file(PREDICTIONS_PATH)

    predictions = pd.read_csv(
        PREDICTIONS_PATH,
        encoding="utf-8-sig",
    )

    predictions = clean_columns(predictions)

    missing = [
        column
        for column in REQUIRED_PREDICTION_COLUMNS
        if column not in predictions.columns
    ]

    if missing:
        raise ValueError(
            "Prediction file ขาดคอลัมน์:\n"
            + "\n".join(missing)
        )

    row_count = len(predictions)

    print(
        f"Final Test rows : {row_count}"
    )

    # --------------------------------------------------------
    # IMPORTANT CHECK
    # --------------------------------------------------------

    if row_count != EXPECTED_FINAL_TEST_ROWS:
        raise ValueError(
            "\n"
            "STOP: ผล Stage 013 ที่อ่านได้ไม่ใช่ "
            "Final Test รุ่นปัจจุบันที่คาดไว้\n\n"
            f"พบ {row_count:,} Window\n"
            f"คาดหวัง {EXPECTED_FINAL_TEST_ROWS:,} Window\n\n"
            "ห้ามนำผลชุดนี้ไปสร้างรูปสำหรับบทที่ 4"
        )

    if "true_binary_label" not in predictions.columns:
        raise ValueError(
            "ไม่พบ true_binary_label"
        )

    y_true = pd.to_numeric(
        predictions["true_binary_label"],
        errors="coerce",
    )

    if y_true.isna().any():
        raise ValueError(
            "true_binary_label มีค่าที่ไม่ใช่ตัวเลข"
        )

    y_true = y_true.astype(int)

    normal_count = int(
        np.sum(y_true == 0)
    )

    attack_count = int(
        np.sum(y_true == 1)
    )

    print(
        f"Normal          : {normal_count}"
    )

    print(
        f"Attack          : {attack_count}"
    )

    if normal_count != EXPECTED_NORMAL_ROWS:
        raise ValueError(
            "จำนวน Normal ไม่ตรงกับผล Final Test ปัจจุบัน"
        )

    if attack_count != EXPECTED_ATTACK_ROWS:
        raise ValueError(
            "จำนวน Attack ไม่ตรงกับผล Final Test ปัจจุบัน"
        )

    if not set(y_true.unique()).issubset({0, 1}):
        raise ValueError(
            "Label ต้องมีเฉพาะ 0 และ 1"
        )

    # --------------------------------------------------------
    # Score validation
    # --------------------------------------------------------

    score_columns = [
        "if_anomaly_score",
        "ocsvm_anomaly_score",
        "lof_anomaly_score",
        "statistical_anomaly_score",
    ]

    for column in score_columns:

        predictions[column] = pd.to_numeric(
            predictions[column],
            errors="coerce",
        )

        if predictions[column].isna().any():
            raise ValueError(
                f"{column} มีค่า NaN"
            )

        values = predictions[column].to_numpy(
            dtype=float
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"{column} มีค่า Infinite"
            )

    predictions["true_binary_label"] = y_true

    print(
        "[PASS] Final Test integrity check"
    )

    return predictions


# ============================================================
# 7. FIGURE 1
#    MODEL METRIC COMPARISON
# ============================================================

def create_metric_comparison(
    metrics: pd.DataFrame,
) -> None:

    print(
        "\n[3] CREATE MODEL METRIC COMPARISON"
    )
    print("-" * 72)

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]

    metric_labels = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
    ]

    x = np.arange(len(MODEL_ORDER))

    width = 0.18

    fig = plt.figure(
        figsize=(12, 7)
    )

    ax = fig.add_axes(
        [0.09, 0.13, 0.88, 0.78]
    )

    for index, column in enumerate(
        metric_columns
    ):

        values = (
            metrics[column]
            .astype(float)
            .to_numpy()
        )

        offset = (
            index - 1.5
        ) * width

        bars = ax.bar(
            x + offset,
            values,
            width,
            label=metric_labels[index],
        )

        for bar, value in zip(
            bars,
            values,
        ):

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                value + 0.015,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    ax.set_title(
        "CIC-IDS2017 Model Performance Comparison",
        fontsize=15,
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        MODEL_ORDER,
        rotation=15,
        ha="right",
    )

    ax.set_ylim(
        0,
        1.08,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.35,
    )

    ax.legend()

    output_path = (
        OUTPUT_DIR
        / "fig_4_8_1_model_performance_comparison.png"
    )

    save_figure(
        fig,
        output_path,
    )


# ============================================================
# 8. FIGURE 2
#    ROC CURVE
# ============================================================

def create_roc_curve(
    predictions: pd.DataFrame,
) -> None:

    print(
        "\n[4] CREATE ROC CURVE"
    )
    print("-" * 72)

    y_true = (
        predictions["true_binary_label"]
        .to_numpy(dtype=int)
    )

    score_map = {
        "Isolation Forest":
            "if_anomaly_score",

        "One-Class SVM":
            "ocsvm_anomaly_score",

        "LOF":
            "lof_anomaly_score",

        "Statistical Threshold":
            "statistical_anomaly_score",
    }

    fig = plt.figure(
        figsize=(9, 7)
    )

    ax = fig.add_axes(
        [0.11, 0.11, 0.84, 0.82]
    )

    for model in MODEL_ORDER:

        score_column = score_map[model]

        scores = (
            predictions[score_column]
            .to_numpy(dtype=float)
        )

        fpr, tpr, _ = roc_curve(
            y_true,
            scores,
        )

        auc_value = roc_auc_score(
            y_true,
            scores,
        )

        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            label=(
                f"{model} "
                f"(AUC = {auc_value:.4f})"
            ),
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Random Classifier",
    )

    ax.set_title(
        "ROC Curve — CIC-IDS2017 Final Test",
        fontsize=15,
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_xlim(
        0,
        1,
    )

    ax.set_ylim(
        0,
        1.02,
    )

    ax.grid(
        linestyle="--",
        alpha=0.35,
    )

    ax.legend(
        loc="lower right"
    )

    output_path = (
        OUTPUT_DIR
        / "fig_4_8_1_roc_curve.png"
    )

    save_figure(
        fig,
        output_path,
    )


# ============================================================
# 9. FIGURE 3
#    PRECISION-RECALL CURVE
# ============================================================

def create_pr_curve(
    predictions: pd.DataFrame,
) -> None:

    print(
        "\n[5] CREATE PRECISION-RECALL CURVE"
    )
    print("-" * 72)

    y_true = (
        predictions["true_binary_label"]
        .to_numpy(dtype=int)
    )

    score_map = {
        "Isolation Forest":
            "if_anomaly_score",

        "One-Class SVM":
            "ocsvm_anomaly_score",

        "LOF":
            "lof_anomaly_score",

        "Statistical Threshold":
            "statistical_anomaly_score",
    }

    prevalence = float(
        np.mean(y_true == 1)
    )

    fig = plt.figure(
        figsize=(9, 7)
    )

    ax = fig.add_axes(
        [0.11, 0.11, 0.84, 0.82]
    )

    for model in MODEL_ORDER:

        score_column = score_map[model]

        scores = (
            predictions[score_column]
            .to_numpy(dtype=float)
        )

        precision, recall, _ = (
            precision_recall_curve(
                y_true,
                scores,
            )
        )

        ap_value = average_precision_score(
            y_true,
            scores,
        )

        ax.plot(
            recall,
            precision,
            linewidth=2,
            label=(
                f"{model} "
                f"(AP = {ap_value:.4f})"
            ),
        )

    ax.axhline(
        prevalence,
        linestyle="--",
        linewidth=1.5,
        label=(
            f"Prevalence = "
            f"{prevalence:.4f}"
        ),
    )

    ax.set_title(
        "Precision-Recall Curve — CIC-IDS2017 Final Test",
        fontsize=15,
    )

    ax.set_xlabel(
        "Recall"
    )

    ax.set_ylabel(
        "Precision"
    )

    ax.set_xlim(
        0,
        1,
    )

    ax.set_ylim(
        0,
        1.02,
    )

    ax.grid(
        linestyle="--",
        alpha=0.35,
    )

    ax.legend(
        loc="lower left"
    )

    output_path = (
        OUTPUT_DIR
        / "fig_4_8_1_precision_recall_curve.png"
    )

    save_figure(
        fig,
        output_path,
    )


# ============================================================
# 10. SAVE FIGURE SUMMARY
# ============================================================

def save_summary(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:

    rows = []

    y_true = (
        predictions["true_binary_label"]
        .to_numpy(dtype=int)
    )

    score_map = {
        "Isolation Forest":
            "if_anomaly_score",

        "One-Class SVM":
            "ocsvm_anomaly_score",

        "LOF":
            "lof_anomaly_score",

        "Statistical Threshold":
            "statistical_anomaly_score",
    }

    for model in MODEL_ORDER:

        score_column = score_map[model]

        scores = (
            predictions[score_column]
            .to_numpy(dtype=float)
        )

        roc_auc = roc_auc_score(
            y_true,
            scores,
        )

        pr_auc = average_precision_score(
            y_true,
            scores,
        )

        metric_row = metrics.loc[
            metrics["model"]
            == model
        ].iloc[0]

        rows.append(
            {
                "model": model,
                "accuracy":
                    float(
                        metric_row["accuracy"]
                    ),
                "precision":
                    float(
                        metric_row["precision"]
                    ),
                "recall":
                    float(
                        metric_row["recall"]
                    ),
                "f1":
                    float(
                        metric_row["f1"]
                    ),
                "roc_auc_from_predictions":
                    roc_auc,
                "pr_auc_from_predictions":
                    pr_auc,
            }
        )

    summary_df = pd.DataFrame(rows)

    output_path = (
        OUTPUT_DIR
        / "stage017_visualization_summary.csv"
    )

    summary_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"[SAVED] {output_path}"
    )


# ============================================================
# 11. MAIN
# ============================================================

def main() -> int:

    print("=" * 72)
    print(
        "STAGE 017 — CIC-IDS2017 "
        "CHAPTER 4 VISUALIZATION"
    )
    print("=" * 72)

    print(
        f"Metrics     : {METRICS_PATH}"
    )

    print(
        f"Predictions : {PREDICTIONS_PATH}"
    )

    print(
        f"Output      : {OUTPUT_DIR}"
    )

    print(
        "Expected Final Test : "
        f"{EXPECTED_FINAL_TEST_ROWS:,} rows"
    )

    try:

        metrics = load_metrics()

        predictions = load_predictions()

        create_metric_comparison(
            metrics
        )

        create_roc_curve(
            predictions
        )

        create_pr_curve(
            predictions
        )

        save_summary(
            metrics,
            predictions,
        )

    except Exception as exc:

        print(
            "\n[ERROR]"
        )

        print(
            str(exc)
        )

        return 1

    print("\n" + "=" * 72)
    print(
        "STAGE 017 — COMPLETED"
    )
    print("=" * 72)

    print(
        "\nFigures created for Chapter 4.8.1:"
    )

    print(
        "1. fig_4_8_1_model_performance_comparison.png"
    )

    print(
        "2. fig_4_8_1_roc_curve.png"
    )

    print(
        "3. fig_4_8_1_precision_recall_curve.png"
    )

    print(
        "\n[PASS] All Chapter 4.8.1 figures generated."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())