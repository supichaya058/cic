
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# STAGE 011 — CIC-IDS2017 STATISTICAL THRESHOLD BASELINE
# ============================================================
#
# INPUT:
#   data/time_windows/cic_train_scaled_10s.csv
#   data/time_windows/cic_validation_scaled_10s.csv
#
# Stage 008 Feature Selection is not used.
# All 13 Behavioral Features from Stage 007 are used.
#
# OUTPUT:
#   data/time_windows/statistical_threshold_baseline/
#       cic_statistical_threshold_tuning_results_10s.csv
#       cic_statistical_threshold_model_10s.json
#       cic_statistical_threshold_locked_10s.csv
#       cic_statistical_threshold_parameters_10s.csv
#       cic_statistical_threshold_validation_scores_10s.csv
#       cic_statistical_threshold_summary_10s.csv
#
# TRAIN:
#   Normal-only
#   Fit statistical reference only.
#
# VALIDATION:
#   Normal + Attack
#   Tune and select threshold.
#
# FINAL TEST:
#   NOT USED
#
# FEATURES:
#   All 13 Behavioral Features from Stage 007.
#
# SCORE:
#   mean(((x - train_median) / reference_scale) ** 2)
#   using all 13 Behavioral Features.
#
# THRESHOLD:
#   Percentile of VALIDATION Normal scores.
#
# SELECTION:
#   Normal-only FAR <= 15%
#   Highest Validation F1
#
# PREDICTION:
#   score > threshold => Attack
#
# No ML model is trained in this stage.
# ============================================================


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "time_windows"

OUTPUT_DIR = (
    DATA_DIR / "statistical_threshold_baseline"
)

TRAIN_PATH = (
    DATA_DIR / "cic_train_scaled_10s.csv"
)

VALIDATION_PATH = (
    DATA_DIR / "cic_validation_scaled_10s.csv"
)

TUNING_RESULTS_PATH = (
    OUTPUT_DIR
    / "cic_statistical_threshold_tuning_results_10s.csv"
)

MODEL_PATH = (
    OUTPUT_DIR
    / "cic_statistical_threshold_model_10s.json"
)

LOCKED_PATH = (
    OUTPUT_DIR
    / "cic_statistical_threshold_locked_10s.csv"
)

PARAMETERS_PATH = (
    OUTPUT_DIR
    / "cic_statistical_threshold_parameters_10s.csv"
)

VALIDATION_SCORES_PATH = (
    OUTPUT_DIR
    / "cic_statistical_threshold_validation_scores_10s.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "cic_statistical_threshold_summary_10s.csv"
)

WINDOW_SECONDS = 10

THRESHOLD_PERCENTILES = [
    80, 85, 88, 90, 92, 95, 97, 98, 99
]

MAX_NORMAL_FAR = 0.15
EPSILON = 1e-12

FEATURE_COLUMNS = [
    "request_rate",
    "total_requests",
    "throughput",
    "flow_count",
    "error_rate",
    "4xx_count",
    "5xx_count",
    "avg_latency",
    "max_latency",
    "latency_std_dev",
    "unique_ip_count",
    "new_ip_ratio",
    "active_sessions",
]

NORMAL_LABELS = {
    "normal",
    "benign",
    "legitimate",
    "clean",
    "background",
    "false",
    "no",
    "0",
    "0.0",
}

ATTACK_LABELS = {
    "attack",
    "malicious",
    "true",
    "yes",
    "1",
    "1.0",
}

EXPECTED_ATTACK_TYPES = {
    "dosslowloris",
    "dosslowhttptest",
    "doshulk",
    "dosgoldeneye",
    "ddosloic",
}


# ============================================================
# 2. FILE UTILITIES
# ============================================================

def clean_columns(
    df: pd.DataFrame,
    description: str,
) -> pd.DataFrame:

    result = df.copy()

    result.columns = (
        result.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    duplicates = result.columns[
        result.columns.duplicated()
    ].tolist()

    if duplicates:
        raise ValueError(
            f"{description} มีชื่อคอลัมน์ซ้ำ: {duplicates}"
        )

    return result


def atomic_csv(
    df: pd.DataFrame,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_name(
        path.name + ".tmp"
    )

    try:
        df.to_csv(
            temp_path,
            index=False,
            encoding="utf-8-sig",
        )

        os.replace(
            temp_path,
            path,
        )

    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_json(
    data: dict[str, Any],
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_name(
        path.name + ".tmp"
    )

    try:
        with temp_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                indent=4,
                ensure_ascii=False,
                allow_nan=False,
            )

        os.replace(
            temp_path,
            path,
        )

    finally:
        if temp_path.exists():
            temp_path.unlink()


# ============================================================
# 3. LOAD MODEL FEATURES
# ============================================================

def load_features() -> list[str]:
    """Use all 13 Behavioral Features from Stage 007."""

    features = list(FEATURE_COLUMNS)

    if len(features) != 13:
        raise RuntimeError(
            f"จำนวน Behavioral Features ต้องเป็น 13 แต่พบ {len(features)}"
        )

    if len(set(features)) != len(features):
        raise RuntimeError("พบชื่อ Behavioral Feature ซ้ำกัน")

    print(f"[PASS] Using all Behavioral Features: {len(features)}")

    for feature in features:
        print(f"  - {feature}")

    return features


# ============================================================
# 4. LABEL HANDLING
# ============================================================

def convert_labels(
    series: pd.Series,
    description: str,
) -> np.ndarray:

    labels = []

    for value in series:

        if pd.isna(value):
            raise ValueError(
                f"{description} มี Label ว่าง"
            )

        if isinstance(value, (bool, np.bool_)):
            labels.append(int(value))
            continue

        if isinstance(
            value,
            (int, float, np.integer, np.floating),
        ):
            number = float(value)

            if number == 0.0:
                labels.append(0)
            elif number == 1.0:
                labels.append(1)
            else:
                raise ValueError(
                    f"{description} พบ Binary Label "
                    f"ที่ไม่รองรับ: {value}"
                )

            continue

        token = str(value).strip().lower()

        if token in NORMAL_LABELS:
            labels.append(0)

        elif token in ATTACK_LABELS:
            labels.append(1)

        else:
            raise ValueError(
                f"{description} พบ Label ที่ไม่รู้จัก: {value}"
            )

    return np.asarray(
        labels,
        dtype=np.int8,
    )


def normalize_attack_type(value) -> str:

    if pd.isna(value):
        return ""

    return "".join(
        char.lower()
        for char in str(value)
        if char.isalnum()
    )


def get_attack_type(
    row: pd.Series,
) -> str | None:

    for column in ("attack_type", "traffic_type"):

        if column not in row.index:
            continue

        token = normalize_attack_type(
            row[column]
        )

        if not token:
            continue

        for expected in EXPECTED_ATTACK_TYPES:
            if expected in token:
                return expected

    return None


def make_window_keys(
    df: pd.DataFrame,
) -> set[tuple[str, int]]:

    window_ids = pd.to_numeric(
        df["window_id"],
        errors="coerce",
    )

    if (
        window_ids.isna().any()
        or not np.isfinite(window_ids).all()
        or (window_ids < 1).any()
        or (window_ids % 1 != 0).any()
    ):
        raise ValueError(
            "พบ window_id ที่ไม่ใช่จำนวนเต็มบวก"
        )

    return set(
        zip(
            df["source_file"].astype(str).str.strip(),
            window_ids.astype("int64"),
        )
    )


# ============================================================
# 5. LOAD AND VALIDATE DATA
# ============================================================

def load_split(
    path: Path,
    split_name: str,
    features: list[str],
) -> pd.DataFrame:

    if not path.is_file():
        raise FileNotFoundError(
            f"ไม่พบไฟล์ {split_name}:\n{path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False,
        encoding="utf-8-sig",
    )

    df = clean_columns(
        df,
        split_name,
    )

    if df.empty:
        raise ValueError(
            f"{split_name} ไม่มีข้อมูล"
        )

    required = {
        "source_file",
        "window_id",
        "binary_label",
        "split",
        *features,
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{split_name} ขาดคอลัมน์: {missing}"
        )

    actual_splits = (
        df["split"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if set(actual_splits.unique()) != {split_name}:
        raise ValueError(
            f"{split_name} มีค่า Split ไม่ถูกต้อง: "
            f"{sorted(actual_splits.unique())}"
        )

    metadata_columns = [
        "source_file",
        "window_id",
        "binary_label",
    ]

    if df[metadata_columns].isna().any().any():
        raise ValueError(
            f"{split_name} มี Metadata ว่าง"
        )

    df["source_file"] = (
        df["source_file"]
        .astype(str)
        .str.strip()
    )

    if df["source_file"].eq("").any():
        raise ValueError(
            f"{split_name} พบ source_file ว่าง"
        )

    window_id = pd.to_numeric(
        df["window_id"],
        errors="coerce",
    )

    invalid_window = (
        window_id.isna()
        | ~np.isfinite(window_id)
        | (window_id < 1)
        | (window_id % 1 != 0)
    )

    if invalid_window.any():
        raise ValueError(
            f"{split_name} มี window_id ไม่ถูกต้อง"
        )

    df["window_id"] = window_id.astype("int64")

    if df.duplicated(
        subset=["source_file", "window_id"]
    ).any():
        raise ValueError(
            f"{split_name} มี Window ซ้ำ"
        )

    # Validate labels.
    convert_labels(
        df["binary_label"],
        split_name,
    )

    # Validate feature values.
    for feature in features:

        values = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

        invalid = (
            values.isna()
            | ~np.isfinite(values)
        )

        if invalid.any():
            raise ValueError(
                f"{split_name}: {feature} มี NaN หรือ Inf "
                f"{int(invalid.sum()):,} แถว"
            )

        df[feature] = values.astype("float64")

    print(
        f"[PASS] {split_name}: "
        f"{len(df):,} rows, {len(features)} features"
    )

    return df


def validate_data(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:

    y_train = convert_labels(
        train_df["binary_label"],
        "TRAIN",
    )

    y_validation = convert_labels(
        validation_df["binary_label"],
        "VALIDATION",
    )

    if not np.all(y_train == 0):
        raise ValueError(
            "TRAIN ต้องมีเฉพาะ Normal/BENIGN"
        )

    if not np.any(y_validation == 0):
        raise ValueError(
            "VALIDATION ไม่มี Normal/BENIGN"
        )

    if not np.any(y_validation == 1):
        raise ValueError(
            "VALIDATION ไม่มี Attack"
        )

    # TRAIN source should be different from VALIDATION source.
    train_sources = set(
        train_df["source_file"].astype(str).str.strip()
    )

    validation_sources = set(
        validation_df["source_file"].astype(str).str.strip()
    )

    if train_sources.intersection(validation_sources):
        raise ValueError(
            "พบ Source File ซ้ำระหว่าง TRAIN และ VALIDATION"
        )

    # Exact windows must not overlap.
    if make_window_keys(train_df).intersection(
        make_window_keys(validation_df)
    ):
        raise ValueError(
            "พบ Window ซ้ำระหว่าง TRAIN และ VALIDATION"
        )

    # Verify that the selected attack types exist in Validation.
    if (
        "attack_type" not in validation_df.columns
        and "traffic_type" not in validation_df.columns
    ):
        raise ValueError(
            "ไม่พบ attack_type หรือ traffic_type "
            "สำหรับตรวจสอบประเภทการโจมตี"
        )

    validation_attacks = validation_df.loc[
        y_validation == 1
    ]

    observed = set()

    for _, row in validation_attacks.iterrows():
        attack_type = get_attack_type(row)

        if attack_type:
            observed.add(attack_type)

    missing_attacks = sorted(
        EXPECTED_ATTACK_TYPES - observed
    )

    if missing_attacks:
        raise ValueError(
            f"VALIDATION ขาด Attack Type: {missing_attacks}"
        )

    print("[PASS] TRAIN contains Normal only")
    print("[PASS] VALIDATION contains Normal and Attack")
    print("[PASS] TRAIN and VALIDATION source separation")
    print("[PASS] No duplicated windows")
    print("[PASS] All five attack types present in VALIDATION")

    print("\nVALIDATION LABEL DISTRIBUTION")
    print("-" * 76)
    print(f"Normal : {int(np.sum(y_validation == 0)):,}")
    print(f"Attack : {int(np.sum(y_validation == 1)):,}")

    return y_train, y_validation


# ============================================================
# 6. PREPARE FEATURE MATRICES
# ============================================================

def make_feature_matrix(
    df: pd.DataFrame,
    features: list[str],
    description: str,
) -> np.ndarray:

    X = df[features].to_numpy(
        dtype=np.float64
    )

    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError(
            f"{description} ไม่มี Feature Matrix"
        )

    if not np.isfinite(X).all():
        raise ValueError(
            f"{description} มี NaN หรือ Inf"
        )

    return X


# ============================================================
# 7. FIT STATISTICAL REFERENCE USING TRAIN ONLY
# ============================================================

def fit_statistical_reference(
    X_train: np.ndarray,
    features: list[str],
) -> pd.DataFrame:
    """
    Calculate robust reference statistics from TRAIN only.

    Robust scale priority:
        1. 1.4826 * MAD
        2. IQR / 1.349
        3. Standard deviation

    The input data has already been scaled in Stage 007.
    No scaler is fitted again here.
    """

    rows = []

    for index, feature in enumerate(features):

        values = X_train[:, index]

        median = float(np.median(values))

        mad = float(
            np.median(
                np.abs(values - median)
            )
        )

        scale = 1.4826 * mad
        scale_method = "MAD"

        if scale <= EPSILON:
            q25, q75 = np.percentile(
                values,
                [25, 75],
            )

            scale = float(
                (q75 - q25) / 1.349
            )
            scale_method = "IQR"

        if scale <= EPSILON:
            scale = float(
                np.std(values, ddof=0)
            )
            scale_method = "STD"

        if (
            not np.isfinite(median)
            or not np.isfinite(mad)
            or not np.isfinite(scale)
            or scale <= EPSILON
        ):
            raise ValueError(
                f"Feature {feature} มี Reference Scale "
                "เป็นศูนย์หรือไม่สามารถคำนวณได้"
            )

        rows.append({
            "feature": feature,
            "train_median": median,
            "train_mad": mad,
            "reference_scale": scale,
            "scale_method": scale_method,
            "train_mean": float(np.mean(values)),
            "train_std": float(np.std(values, ddof=0)),
            "train_rows": int(len(X_train)),
            "fit_split": "TRAIN",
        })

    reference = pd.DataFrame(rows)

    print("\n[PASS] Statistical reference fitted using TRAIN only")
    print(reference.to_string(index=False))

    return reference


# ============================================================
# 8. CALCULATE STATISTICAL ANOMALY SCORE
# ============================================================

def calculate_statistical_score(
    X: np.ndarray,
    reference: pd.DataFrame,
    features: list[str],
) -> np.ndarray:
    """
    z_j = (x_j - train_median_j) / reference_scale_j
    score = mean(z_j ** 2)

    Higher score = greater deviation from TRAIN.
    """

    indexed = reference.set_index("feature")

    medians = np.asarray(
        [
            indexed.loc[feature, "train_median"]
            for feature in features
        ],
        dtype=np.float64,
    )

    scales = np.asarray(
        [
            indexed.loc[feature, "reference_scale"]
            for feature in features
        ],
        dtype=np.float64,
    )

    if (
        not np.isfinite(medians).all()
        or not np.isfinite(scales).all()
        or (scales <= 0).any()
    ):
        raise ValueError(
            "Statistical Reference ไม่ถูกต้อง"
        )

    z_scores = (X - medians) / scales
    scores = np.mean(np.square(z_scores), axis=1)

    scores = np.asarray(
        scores,
        dtype=np.float64,
    ).reshape(-1)

    if not np.isfinite(scores).all():
        raise ValueError(
            "Statistical anomaly scores มี NaN หรือ Inf"
        )

    return scores


# ============================================================
# 9. EVALUATION METRICS
# ============================================================

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | int]:

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    normal_total = int(
        np.sum(y_true == 0)
    )

    accuracy = float(
        accuracy_score(y_true, y_pred)
    )

    precision = float(
        precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    recall = float(
        recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    f1 = float(
        f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    normal_far = (
        float(fp / normal_total)
        if normal_total > 0
        else np.nan
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "far": normal_far,
        "normal_far": normal_far,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_attack": int(fp + tp),
    }


# ============================================================
# 10. THRESHOLD TUNING ON VALIDATION
# ============================================================

def tune_threshold(
    validation_scores: np.ndarray,
    y_validation: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """
    Tune Statistical Threshold using VALIDATION only.

    Threshold percentiles are calculated from VALIDATION
    Normal scores.

    Selection:
        1. Normal-only FAR <= 15%
        2. Highest F1
        3. Highest Precision
        4. Lowest Normal-only FAR
        5. Higher percentile
    """

    normal_scores = validation_scores[
        y_validation == 0
    ]

    if len(normal_scores) == 0:
        raise ValueError(
            "ไม่มี Normal scores สำหรับสร้าง Threshold"
        )

    roc_auc = float(
        roc_auc_score(
            y_validation,
            validation_scores,
        )
    )

    pr_auc = float(
        average_precision_score(
            y_validation,
            validation_scores,
        )
    )

    rows = []

    for percentile in THRESHOLD_PERCENTILES:

        threshold = float(
            np.percentile(
                normal_scores,
                percentile,
            )
        )

        predictions = (
            validation_scores > threshold
        ).astype(np.int8)

        metrics = calculate_metrics(
            y_validation,
            predictions,
        )

        eligible = bool(
            metrics["normal_far"]
            <= MAX_NORMAL_FAR + EPSILON
        )

        rows.append({
            "stage": "011",
            "dataset": "CIC-IDS2017",
            "method": "Statistical Threshold",
            "window_seconds": WINDOW_SECONDS,
            "percentile": int(percentile),
            "threshold": threshold,
            **metrics,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "max_normal_far": MAX_NORMAL_FAR,
            "eligible": eligible,
            "selected": False,
            "threshold_source": "VALIDATION Normal scores",
            "fit_data": "TRAIN only",
            "evaluation_data": "VALIDATION only",
            "final_test_used": False,
        })

    tuning_df = pd.DataFrame(rows)

    eligible_df = tuning_df.loc[
        tuning_df["eligible"]
    ].copy()

    if eligible_df.empty:
        return tuning_df, None

    eligible_df = eligible_df.sort_values(
        by=[
            "f1",
            "precision",
            "normal_far",
            "percentile",
        ],
        ascending=[
            False,
            False,
            True,
            False,
        ],
        kind="mergesort",
    )

    best = eligible_df.iloc[0].to_dict()

    selected_mask = (
        tuning_df["percentile"].eq(
            int(best["percentile"])
        )
    )

    tuning_df.loc[
        selected_mask,
        "selected",
    ] = True

    return tuning_df, best


# ============================================================
# 11. SAVE LOCKED STATISTICAL MODEL
# ============================================================

def save_locked_model(
    best: dict[str, Any],
    reference: pd.DataFrame,
    features: list[str],
    train_rows: int,
) -> None:
    """
    Save the locked Statistical Threshold.

    Schema is compatible with Stage 012:
        stage
        method
        dataset
        window
        features
        threshold_percentile
        threshold
        reference[feature].median
        reference[feature].scale
        validation_metrics
    """

    locked = {
        "stage": "011",
        "method": "Statistical Threshold",
        "dataset": "CIC-IDS2017",
        "window": "10 seconds",
        "features": list(features),
        "feature_selection": "NOT PERFORMED",
        "score_formula": (
            "mean(((x - train_median) / reference_scale) ** 2)"
        ),
        "score_direction": "higher_is_more_anomalous",
        "prediction_rule": "score > threshold",
        "threshold_percentile": int(
            best["percentile"]
        ),
        "threshold": float(best["threshold"]),
        "max_normal_far": MAX_NORMAL_FAR,
        "validation_metrics": {
            "accuracy": float(best["accuracy"]),
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "f1": float(best["f1"]),
            "normal_far": float(best["normal_far"]),
            "roc_auc": float(best["roc_auc"]),
            "pr_auc": float(best["pr_auc"]),
            "tn": int(best["tn"]),
            "fp": int(best["fp"]),
            "fn": int(best["fn"]),
            "tp": int(best["tp"]),
        },
        "reference": {
            str(row["feature"]): {
                "median": float(row["train_median"]),
                "scale": float(row["reference_scale"]),
                "scale_method": str(row["scale_method"]),
            }
            for _, row in reference.iterrows()
        },
        "fit_data": "TRAIN only",
        "threshold_tuning_data": "VALIDATION only",
        "train_rows": int(train_rows),
        "final_test_loaded": False,
    }

    atomic_json(
        locked,
        MODEL_PATH,
    )

    # Human-readable locked record.
    locked_row = {
        "stage": "011",
        "model": "Statistical Threshold",
        "dataset": "CIC-IDS2017",
        "window": "10 seconds",
        "features": "|".join(features),
        "score_formula": locked["score_formula"],
        "threshold_percentile": locked["threshold_percentile"],
        "threshold": locked["threshold"],
        "max_normal_far": MAX_NORMAL_FAR,
        **locked["validation_metrics"],
        "fit_data": "TRAIN only",
        "threshold_tuning_data": "VALIDATION only",
        "final_test_loaded": False,
    }

    atomic_csv(
        pd.DataFrame([locked_row]),
        LOCKED_PATH,
    )

    # Save TRAIN reference parameters.
    atomic_csv(
        reference,
        PARAMETERS_PATH,
    )

    print(f"[SAVED] {MODEL_PATH}")
    print(f"[SAVED] {LOCKED_PATH}")
    print(f"[SAVED] {PARAMETERS_PATH}")


# ============================================================
# 12. SAVE VALIDATION PREDICTIONS
# ============================================================

def save_validation_predictions(
    validation_df: pd.DataFrame,
    validation_scores: np.ndarray,
    threshold: float,
) -> None:

    predictions = validation_df.copy()

    predictions["statistical_anomaly_score"] = (
        validation_scores
    )

    predictions["statistical_prediction"] = (
        validation_scores > threshold
    ).astype(np.int8)

    predictions["statistical_prediction_label"] = np.where(
        predictions["statistical_prediction"].to_numpy() == 1,
        "Attack",
        "Normal",
    )

    atomic_csv(
        predictions,
        VALIDATION_SCORES_PATH,
    )

    print(f"[SAVED] {VALIDATION_SCORES_PATH}")


# ============================================================
# 13. SAVE TUNING SUMMARY
# ============================================================

def save_summary(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    best: dict[str, Any],
    features: list[str],
) -> None:

    summary = [
        {"item": "stage", "value": "011"},
        {"item": "dataset", "value": "CIC-IDS2017"},
        {"item": "window", "value": "10 seconds"},
        {"item": "train_rows", "value": int(len(train_df))},
        {"item": "validation_rows", "value": int(len(validation_df))},
        {
            "item": "train_normal_rows",
            "value": int(np.sum(y_train == 0)),
        },
        {
            "item": "train_attack_rows",
            "value": int(np.sum(y_train == 1)),
        },
        {
            "item": "validation_normal_rows",
            "value": int(np.sum(y_validation == 0)),
        },
        {
            "item": "validation_attack_rows",
            "value": int(np.sum(y_validation == 1)),
        },
        {
            "item": "features_used",
            "value": "|".join(features),
        },
        {
            "item": "score_formula",
            "value": (
                "mean(((x - train_median) / reference_scale) ** 2)"
            ),
        },
        {
            "item": "threshold_percentile",
            "value": int(best["percentile"]),
        },
        {
            "item": "locked_threshold",
            "value": float(best["threshold"]),
        },
        {
            "item": "max_normal_far",
            "value": MAX_NORMAL_FAR,
        },
        {
            "item": "validation_accuracy",
            "value": float(best["accuracy"]),
        },
        {
            "item": "validation_precision",
            "value": float(best["precision"]),
        },
        {
            "item": "validation_recall",
            "value": float(best["recall"]),
        },
        {
            "item": "validation_f1",
            "value": float(best["f1"]),
        },
        {
            "item": "validation_normal_far",
            "value": float(best["normal_far"]),
        },
        {
            "item": "validation_roc_auc",
            "value": float(best["roc_auc"]),
        },
        {
            "item": "validation_pr_auc",
            "value": float(best["pr_auc"]),
        },
        {
            "item": "fit_data",
            "value": "TRAIN only",
        },
        {
            "item": "threshold_tuning_data",
            "value": "VALIDATION only",
        },
        {
            "item": "final_test_loaded",
            "value": False,
        },
        {
            "item": "data_leakage",
            "value": "FINAL_TEST not used",
        },
    ]

    atomic_csv(
        pd.DataFrame(summary),
        SUMMARY_PATH,
    )

    print(f"[SAVED] {SUMMARY_PATH}")


# ============================================================
# 14. MAIN
# ============================================================

def main() -> None:

    print("=" * 78)
    print("STAGE 011 — CIC-IDS2017 STATISTICAL THRESHOLD")
    print("=" * 78)

    print(f"TRAIN      : {TRAIN_PATH}")
    print(f"VALIDATION : {VALIDATION_PATH}")
    print(f"FEATURES   : {len(FEATURE_COLUMNS)} Behavioral Features")
    print(f"OUTPUT     : {OUTPUT_DIR}")
    print("FINAL TEST : NOT LOADED")
    print(f"Threshold percentiles: {THRESHOLD_PERCENTILES}")
    print(f"Normal-only FAR limit: {MAX_NORMAL_FAR:.0%}")
    print("=" * 78)

    # 1. Load all 13 Behavioral Features from Stage 007.
    print("\n[1] Load 13 Behavioral Features")

    features = load_features()

    # 2. Load TRAIN and VALIDATION only.
    print("\n[2] Load TRAIN and VALIDATION")

    train_df = load_split(
        TRAIN_PATH,
        "TRAIN",
        features,
    )

    validation_df = load_split(
        VALIDATION_PATH,
        "VALIDATION",
        features,
    )

    # 3. Validate split and labels.
    print("\n[3] Validate split and labels")

    y_train, y_validation = validate_data(
        train_df,
        validation_df,
    )

    # 4. Create feature matrices.
    print("\n[4] Prepare feature matrices")

    X_train = make_feature_matrix(
        train_df,
        features,
        "TRAIN",
    )

    X_validation = make_feature_matrix(
        validation_df,
        features,
        "VALIDATION",
    )

    print(f"TRAIN matrix      : {X_train.shape}")
    print(f"VALIDATION matrix : {X_validation.shape}")
    print("[INFO] Reuse Stage 007 scaled features; no second scaler")

    # 5. Fit reference from TRAIN only.
    print("\n[5] Fit statistical reference using TRAIN only")

    reference = fit_statistical_reference(
        X_train,
        features,
    )

    # 6. Calculate validation anomaly scores.
    print("\n[6] Calculate VALIDATION statistical scores")

    validation_scores = calculate_statistical_score(
        X_validation,
        reference,
        features,
    )

    print(
        f"Score min  : {validation_scores.min():.10f}"
    )
    print(
        f"Score max  : {validation_scores.max():.10f}"
    )
    print(
        f"Score mean : {validation_scores.mean():.10f}"
    )

    # 7. Tune threshold using Validation only.
    print("\n[7] Tune threshold using VALIDATION")

    tuning_df, best = tune_threshold(
        validation_scores,
        y_validation,
    )

    print("\nTHRESHOLD SEARCH RESULTS")
    print("-" * 78)

    print(
        tuning_df[
            [
                "percentile",
                "threshold",
                "precision",
                "recall",
                "f1",
                "normal_far",
                "eligible",
                "selected",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    if best is None:
        raise RuntimeError(
            "ไม่พบ Threshold ที่ผ่านเกณฑ์ "
            f"Normal FAR <= {MAX_NORMAL_FAR:.0%}"
        )

    # 8. Save tuning results.
    print("\n[8] Save threshold tuning results")

    atomic_csv(
        tuning_df,
        TUNING_RESULTS_PATH,
    )

    print(f"[SAVED] {TUNING_RESULTS_PATH}")

    # 9. Lock the selected model and threshold.
    print("\n[9] Lock selected statistical threshold")

    save_locked_model(
        best,
        reference,
        features,
        len(train_df),
    )

    # 10. Save validation predictions.
    print("\n[10] Save VALIDATION predictions")

    save_validation_predictions(
        validation_df,
        validation_scores,
        float(best["threshold"]),
    )

    # 11. Save summary.
    print("\n[11] Save stage summary")

    save_summary(
        train_df,
        validation_df,
        y_train,
        y_validation,
        best,
        features,
    )

    # 12. Final report.
    print("\n" + "=" * 78)
    print("STAGE 011 — SELECTED STATISTICAL THRESHOLD")
    print("=" * 78)

    print("Status          : SELECTED")
    print("Method          : Statistical Threshold")
    print(f"Percentile      : P{int(best['percentile'])}")
    print(
        f"Threshold       : {float(best['threshold']):.12f}"
    )
    print(f"Accuracy        : {best['accuracy']:.6f}")
    print(f"Precision       : {best['precision']:.6f}")
    print(f"Recall          : {best['recall']:.6f}")
    print(f"F1              : {best['f1']:.6f}")
    print(f"Normal FAR      : {best['normal_far']:.6f}")
    print(f"ROC-AUC         : {best['roc_auc']:.6f}")
    print(f"PR-AUC          : {best['pr_auc']:.6f}")

    print(
        "Confusion Matrix: "
        f"TN={best['tn']}, FP={best['fp']}, "
        f"FN={best['fn']}, TP={best['tp']}"
    )

    print("\n" + "=" * 78)
    print("STAGE 011 SUMMARY")
    print("=" * 78)

    print(f"TRAIN rows       : {len(train_df):,}")
    print(f"VALIDATION rows  : {len(validation_df):,}")
    print(f"Features         : {len(features)}")
    print(
        f"Threshold trials : {len(THRESHOLD_PERCENTILES)}"
    )
    print("Reference fit    : TRAIN only")
    print("Threshold tuning : VALIDATION only")
    print("Final Test       : NOT USED")

    print("\nSAVED FILES")
    print("-" * 78)

    for path in [
        TUNING_RESULTS_PATH,
        MODEL_PATH,
        LOCKED_PATH,
        PARAMETERS_PATH,
        VALIDATION_SCORES_PATH,
        SUMMARY_PATH,
    ]:
        print(path)

    print("=" * 78)
    print("STAGE 011 COMPLETED")
    print("=" * 78)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n" + "=" * 78)
        print("STAGE 011 FAILED")
        print("=" * 78)
        print(f"Error: {exc}")
        sys.exit(1)