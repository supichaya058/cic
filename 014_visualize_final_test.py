from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("TkAgg")  # เปิดหน้าต่างกราฟ
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "ไม่พบ matplotlib กรุณาติดตั้งด้วยคำสั่ง: uv add matplotlib"
    ) from exc

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ============================================================
# STAGE 014 — CIC-IDS2017 FINAL TEST VISUALIZATION
# แสดงกราฟและบันทึกภาพของ Isolation Forest, One-Class SVM,
# LOF และ Statistical Threshold
# ไม่ฝึกโมเดล ไม่ปรับพารามิเตอร์ และไม่เลือก Threshold ใหม่
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "data" / "models"
PREDICTIONS_PATH = MODEL_DIR / "cic_final_test_predictions_10s.csv"
OUTPUT_DIR = MODEL_DIR / "visualizations_10s"

ROC_PR_PLOT_PATH = OUTPUT_DIR / "cic_final_test_roc_pr_curves_10s.png"
CM_ALL_PLOT_PATH = OUTPUT_DIR / "cic_final_test_confusion_matrices_10s.png"
ROC_POINTS_PATH = OUTPUT_DIR / "cic_final_test_roc_curve_points_10s.csv"
PR_POINTS_PATH = OUTPUT_DIR / "cic_final_test_pr_curve_points_10s.csv"
SUMMARY_PATH = OUTPUT_DIR / "cic_final_test_visualization_summary_10s.csv"

MODELS = [
    {
        "name": "Isolation Forest",
        "score_column": "if_anomaly_score",
        "prediction_column": "if_prediction",
        "cm_filename": "cic_final_test_confusion_matrix_if_10s.png",
    },
    {
        "name": "One-Class SVM",
        "score_column": "ocsvm_anomaly_score",
        "prediction_column": "ocsvm_prediction",
        "cm_filename": "cic_final_test_confusion_matrix_ocsvm_10s.png",
    },
    {
        "name": "LOF",
        "score_column": "lof_anomaly_score",
        "prediction_column": "lof_prediction",
        "cm_filename": "cic_final_test_confusion_matrix_lof_10s.png",
    },
    {
        "name": "Statistical Threshold",
        "score_column": "statistical_anomaly_score",
        "prediction_column": "statistical_prediction",
        "cm_filename": "cic_final_test_confusion_matrix_statistical_10s.png",
    },
]

LABEL_NAMES = ["Normal", "Attack"]


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    """บันทึก CSV แบบ temporary file ก่อนแทนที่ไฟล์จริง"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    try:
        df.to_csv(temp_path, index=False, encoding="utf-8-sig")
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def validate_binary_column(series: pd.Series, column_name: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any():
        raise ValueError(f"คอลัมน์ {column_name} มีค่าว่างหรือไม่ใช่ตัวเลข")

    array = values.to_numpy(dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"คอลัมน์ {column_name} มี NaN หรือ Inf")
    if not np.isin(array, [0, 1]).all():
        raise ValueError(f"คอลัมน์ {column_name} ต้องมีเฉพาะ 0 และ 1")
    return array.astype(np.int8)


def load_predictions() -> tuple[pd.DataFrame, np.ndarray]:
    if not PREDICTIONS_PATH.is_file():
        raise FileNotFoundError(
            "ไม่พบไฟล์ผลการทำนายจาก STAGE 013:\n"
            f"{PREDICTIONS_PATH}\nกรุณารัน STAGE 013 ก่อน"
        )

    df = pd.read_csv(
        PREDICTIONS_PATH,
        low_memory=False,
        encoding="utf-8-sig",
    )
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"พบชื่อคอลัมน์ซ้ำ: {duplicates}")
    if df.empty:
        raise ValueError("ไฟล์ FINAL TEST predictions ไม่มีข้อมูล")
    if "true_binary_label" not in df.columns:
        raise ValueError("ไฟล์ STAGE 013 ไม่มี true_binary_label")

    y_true = validate_binary_column(df["true_binary_label"], "true_binary_label")
    if not np.any(y_true == 0) or not np.any(y_true == 1):
        raise ValueError("ข้อมูลต้องมีทั้ง Normal และ Attack")

    required = {
        column
        for model in MODELS
        for column in (model["score_column"], model["prediction_column"])
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"ไฟล์ STAGE 013 ขาดคอลัมน์สำหรับสร้างกราฟ: {missing}")

    for model in MODELS:
        score_col = model["score_column"]
        score = pd.to_numeric(df[score_col], errors="coerce")
        if score.isna().any() or not np.isfinite(score.to_numpy(dtype=float)).all():
            raise ValueError(f"{score_col} มี NaN หรือ Inf")
        df[score_col] = score.astype("float64")

        pred_col = model["prediction_column"]
        df[pred_col] = validate_binary_column(df[pred_col], pred_col)

    print(f"[PASS] STAGE 013 predictions loaded: {len(df):,} rows")
    print(f"[PASS] Normal: {int(np.sum(y_true == 0)):,}")
    print(f"[PASS] Attack: {int(np.sum(y_true == 1)):,}")
    print("[PASS] Scores and predictions found for IF, OCSVM, LOF, and Statistical Threshold")
    return df, y_true


def calculate_curve_data(
    df: pd.DataFrame,
    y_true: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    roc_rows: list[dict[str, object]] = []
    pr_rows: list[dict[str, object]] = []
    curve_metrics: dict[str, dict[str, float]] = {}

    for model in MODELS:
        name = model["name"]
        scores = df[model["score_column"]].to_numpy(dtype=np.float64)

        fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
        roc_auc = float(roc_auc_score(y_true, scores))
        for i in range(len(fpr)):
            roc_rows.append({
                "model": name,
                "fpr": float(fpr[i]),
                "tpr": float(tpr[i]),
                "threshold": float(roc_thresholds[i]),
            })

        precision, recall, pr_thresholds = precision_recall_curve(y_true, scores)
        average_precision = float(average_precision_score(y_true, scores))
        padded_thresholds = np.append(pr_thresholds, np.nan)
        for i in range(len(precision)):
            roc_threshold = padded_thresholds[i]
            pr_rows.append({
                "model": name,
                "recall": float(recall[i]),
                "precision": float(precision[i]),
                "threshold": float(roc_threshold) if np.isfinite(roc_threshold) else np.nan,
            })

        curve_metrics[name] = {
            "roc_auc": roc_auc,
            "pr_auc": average_precision,
        }
        print(
            f"[PASS] {name} curves calculated | "
            f"ROC-AUC={roc_auc:.6f} | AP={average_precision:.6f}"
        )

    return pd.DataFrame(roc_rows), pd.DataFrame(pr_rows), curve_metrics


def show_and_save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[SAVED] {path}")
    plt.show(block=True)  # ปิดหน้าต่างเพื่อแสดงกราฟถัดไป
    plt.close(fig)


def plot_roc_pr_curves(
    df: pd.DataFrame,
    y_true: np.ndarray,
    path: Path,
) -> None:
    """สร้าง ROC และ Precision-Recall ในภาพเดียวกัน รูปแบบพร้อมใช้ในรายงาน."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --------------------------------------------------------
    # (a) ROC
    # --------------------------------------------------------
    ax = axes[0]
    for model in MODELS:
        scores = df[model["score_column"]].to_numpy(dtype=np.float64)
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc_value = roc_auc_score(y_true, scores)
        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{model['name']} (AUC = {auc_value:.4f})",
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.2,
        label="Random classifier (AUC = 0.5000)",
    )
    ax.set_title("(a) ROC", fontsize=13)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(loc="lower right", fontsize=8)

    # --------------------------------------------------------
    # (b) Precision-Recall
    # --------------------------------------------------------
    ax = axes[1]
    for model in MODELS:
        scores = df[model["score_column"]].to_numpy(dtype=np.float64)
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap_value = average_precision_score(y_true, scores)
        ax.plot(
            recall,
            precision,
            linewidth=2,
            label=f"{model['name']} (AP = {ap_value:.4f})",
        )

    prevalence = float(np.mean(y_true == 1))
    ax.axhline(
        prevalence,
        linestyle="--",
        linewidth=1.2,
        label=f"Prevalence = {prevalence:.3f}",
    )
    ax.set_title("(b) Precision-Recall", fontsize=13)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(loc="lower left", fontsize=8)

    fig.suptitle("CIC-IDS2017 — Threshold-Independent Evaluation (Final Test)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[SAVED] {path}")
    plt.show(block=True)
    plt.close(fig)


def get_confusion_data(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    row_totals = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(
        cm.astype(np.float64),
        row_totals,
        out=np.zeros_like(cm, dtype=np.float64),
        where=row_totals != 0,
    )
    return cm, normalized


def annotate_confusion_matrix(ax, cm: np.ndarray, normalized: np.ndarray, title: str) -> None:
    ax.imshow(normalized, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([0, 1], labels=LABEL_NAMES)
    ax.set_yticks([0, 1], labels=LABEL_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i,
                f"{int(cm[i, j]):,}\n{normalized[i, j] * 100:.2f}%",
                ha="center", va="center", fontsize=10,
            )


def plot_all_confusion_matrices(
    df: pd.DataFrame,
    y_true: np.ndarray,
    path: Path,
) -> None:
    # Reserve a dedicated area on the far right for the colorbar
    # so it never overlaps the top-right / bottom-right confusion matrix
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, model in zip(axes_flat, MODELS):
        y_pred = df[model["prediction_column"]].to_numpy(dtype=np.int8)
        cm, normalized = get_confusion_data(y_true, y_pred)
        annotate_confusion_matrix(ax, cm, normalized, model["name"])

    scalar_image = axes_flat[0].images[0]

    # Leave clear space between the 2x2 grid and the colorbar
    fig.subplots_adjust(
        left=0.08,
        right=0.86,
        bottom=0.09,
        top=0.88,
        wspace=0.35,
        hspace=0.38,
    )

    colorbar_ax = fig.add_axes([0.89, 0.16, 0.025, 0.68])
    fig.colorbar(
        scalar_image,
        cax=colorbar_ax,
        label="Proportion within actual class",
    )

    fig.suptitle(
        "CIC-IDS2017 — Confusion Matrices (Final Test)",
        fontsize=15,
        y=0.95,
    )

    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[SAVED] {path}")
    plt.show(block=True)
    plt.close(fig)


def plot_individual_confusion_matrices(
    df: pd.DataFrame,
    y_true: np.ndarray,
) -> list[Path]:
    saved: list[Path] = []
    for model in MODELS:
        y_pred = df[model["prediction_column"]].to_numpy(dtype=np.int8)
        cm, normalized = get_confusion_data(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(7, 6))
        image = ax.imshow(normalized, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        ax.set_title(
            f"CIC-IDS2017 — {model['name']}\nConfusion Matrix (Final Test)",
            fontsize=13,
        )
        ax.set_xticks([0, 1], labels=LABEL_NAMES)
        ax.set_yticks([0, 1], labels=LABEL_NAMES)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")

        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i,
                    f"{int(cm[i, j]):,}\n{normalized[i, j] * 100:.2f}%",
                    ha="center", va="center", fontsize=12,
                )

        fig.colorbar(image, ax=ax, label="Proportion within actual class")
        fig.tight_layout()
        output_path = OUTPUT_DIR / model["cm_filename"]
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[SAVED] {output_path}")
        plt.show(block=True)
        plt.close(fig)
        saved.append(output_path)

    return saved


def save_visualization_summary(
    df: pd.DataFrame,
    y_true: np.ndarray,
    curve_metrics: dict[str, dict[str, float]],
) -> None:
    rows = []
    for model in MODELS:
        name = model["name"]
        y_pred = df[model["prediction_column"]].to_numpy(dtype=np.int8)
        tn, fp, fn, tp = [int(x) for x in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()]
        normal_total = tn + fp
        normal_far = float(fp / normal_total) if normal_total else 0.0
        rows.append({
            "stage": "014",
            "dataset": "CIC-IDS2017",
            "evaluation_split": "FINAL_TEST",
            "model": name,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "normal_far": normal_far,
            "roc_auc": curve_metrics[name]["roc_auc"],
            "pr_auc_average_precision": curve_metrics[name]["pr_auc"],
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "final_test_rows": int(len(df)),
            "normal_rows": int(np.sum(y_true == 0)),
            "attack_rows": int(np.sum(y_true == 1)),
            "source_file": str(PREDICTIONS_PATH),
            "retraining": False,
            "retuning": False,
        })
    summary_df = pd.DataFrame(rows)
    summary_df["best_model_by_f1"] = False
    if not summary_df.empty:
        best_idx = summary_df["f1"].astype(float).idxmax()
        summary_df.loc[best_idx, "best_model_by_f1"] = True
        best_name = str(summary_df.loc[best_idx, "model"])
        best_f1 = float(summary_df.loc[best_idx, "f1"])
        print(f"[PASS] Best Model by Final Test F1: {best_name} | F1={best_f1:.6f}")
    atomic_csv(summary_df, SUMMARY_PATH)
    print(f"[SAVED] {SUMMARY_PATH}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("STAGE 014 — CIC-IDS2017 FINAL TEST VISUALIZATION")
    print("Models: Isolation Forest, One-Class SVM, LOF, Statistical Threshold")
    print(f"INPUT  : {PREDICTIONS_PATH}")
    print(f"OUTPUT : {OUTPUT_DIR}")
    print("DATA   : FINAL_TEST predictions from STAGE 013")
    print("RETRAIN: NO | RETUNE: NO | THRESHOLD SELECTION: NO")
    print("=" * 78)

    print("\n[1] Load STAGE 013 predictions")
    df, y_true = load_predictions()

    print("\n[2] Calculate ROC and PR curve data")
    roc_df, pr_df, curve_metrics = calculate_curve_data(df, y_true)
    atomic_csv(roc_df, ROC_POINTS_PATH)
    atomic_csv(pr_df, PR_POINTS_PATH)
    print(f"[SAVED] {ROC_POINTS_PATH}")
    print(f"[SAVED] {PR_POINTS_PATH}")

    print("\n[3] Create ROC + Precision-Recall Curves")
    plot_roc_pr_curves(df, y_true, ROC_PR_PLOT_PATH)

    print("\n[4] Create Confusion Matrices (IF, OCSVM, LOF, Statistical Threshold)")
    plot_all_confusion_matrices(df, y_true, CM_ALL_PLOT_PATH)
    plot_individual_confusion_matrices(df, y_true)

    print("\n[5] Save visualization summary")
    save_visualization_summary(df, y_true, curve_metrics)

    print("\n" + "=" * 78)
    print("STAGE 014 — VISUALIZATION SUMMARY")
    print(f"Rows       : {len(df):,}")
    print(f"Normal     : {int(np.sum(y_true == 0)):,}")
    print(f"Attack     : {int(np.sum(y_true == 1)):,}")
    print(f"Methods    : {len(MODELS)}")
    print("ROC + PR   : CREATED (side-by-side)")
    print("Confusion  : CREATED (IF, OCSVM, LOF, Statistical Threshold)")
    print("Best model : Selected for reporting by highest Final Test F1")
    print("Retraining : NO")
    print("Retuning   : NO")
    print("STAGE 014 — SUCCESS")
    print("=" * 78)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n" + "=" * 78)
        print("STAGE 014 — FAILED")
        print("=" * 78)
        print(f"Error: {exc}")
        sys.exit(1)
