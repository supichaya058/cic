from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


# ============================================================
# STAGE 012 — CIC-IDS2017 FINAL MODEL TRAINING
# ============================================================
#
# PURPOSE:
#   Refit the selected final ML models using TRAIN only.
#
# INPUT FROM STAGE 007:
#   data/time_windows/cic_train_scaled_10s.csv
#   data/time_windows/cic_validation_scaled_10s.csv
#   - all 13 Behavioral Features
#
# INPUT FROM STAGE 010:
#   data/time_windows/cic_selected_model_configs_10s.csv
#   - selected model configurations
#   - locked detection thresholds
#
# INPUT FROM STAGE 011:
#   data/time_windows/statistical_threshold_baseline/
#       cic_statistical_threshold_model_10s.json
#   - locked statistical reference
#   - locked statistical threshold
#
# OUTPUT:
#   data/models/cic_final_isolation_forest_10s.joblib
#   data/models/cic_final_ocsvm_10s.joblib
#   data/models/cic_final_lof_10s.joblib
#   data/models/cic_final_anomaly_models_10s.joblib
#   data/models/cic_final_model_configs_locked_10s.csv
#   data/models/cic_final_validation_scores_10s.csv
#   data/models/cic_final_validation_metrics_10s.csv
#   data/models/cic_final_model_training_summary_10s.csv
#   data/models/cic_final_model_manifest_10s.json
#
# RULES:
#   1. Use all 13 Behavioral Features from Stage 007.
#   2. Load all ML configurations and thresholds from Stage 010.
#   3. Load Statistical Threshold and reference from Stage 011.
#   4. Fit ML models on TRAIN only.
#   5. Use VALIDATION only for locked-parameter audit.
#   6. Do NOT retune, reselect, or reload Final Test here.
#   7. Reuse the already-scaled data from Stage 007.
#
# SCORE CONVENTION:
#   Higher score = more anomalous.
# ============================================================


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "time_windows"
MODEL_DIR = BASE_DIR / "data" / "models"

TRAIN_PATH = DATA_DIR / "cic_train_scaled_10s.csv"
VALIDATION_PATH = DATA_DIR / "cic_validation_scaled_10s.csv"

STAGE010_CONFIG_PATH = DATA_DIR / "cic_selected_model_configs_10s.csv"

STATISTICAL_MODEL_PATH = (
    DATA_DIR
    / "statistical_threshold_baseline"
    / "cic_statistical_threshold_model_10s.json"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SECONDS = 10
MAX_NORMAL_FAR = 0.15
EPSILON = 1e-12

# All 13 Behavioral Features from Stage 007 are used directly.
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

MODEL_NAMES = [
    "Isolation Forest",
    "One-Class SVM",
    "LOF",
]

MODEL_SHORT_NAMES = {
    "Isolation Forest": "if",
    "One-Class SVM": "ocsvm",
    "LOF": "lof",
}

BENIGN_VALUES = {
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

ATTACK_VALUES = {
    "attack",
    "malicious",
    "true",
    "yes",
    "1",
    "1.0",
}


# ============================================================
# 2. SAFE FILE WRITERS
# ============================================================

def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_name(path.name + ".tmp")

    try:
        df.to_csv(
            temp_path,
            index=False,
            encoding="utf-8-sig",
        )
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_name(path.name + ".tmp")

    try:
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                indent=4,
                ensure_ascii=False,
                allow_nan=False,
            )

        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_joblib(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_name(path.name + ".tmp")

    try:
        joblib.dump(obj, temp_path)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


# ============================================================
# 3. COLUMN CLEANING
# ============================================================

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result.columns = (
        result.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    duplicated = result.columns[
        result.columns.duplicated()
    ].tolist()

    if duplicated:
        raise ValueError(
            f"พบชื่อคอลัมน์ซ้ำ: {duplicated}"
        )

    return result


# ============================================================
# 4. LOAD MODEL FEATURES
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
# 5. LOAD STAGE 010 CONFIGURATIONS AND THRESHOLDS
# ============================================================

def load_stage010() -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    if not STAGE010_CONFIG_PATH.is_file():
        raise FileNotFoundError(
            "ไม่พบผลล็อกจาก Stage 010:\n"
            f"{STAGE010_CONFIG_PATH}"
        )

    df = pd.read_csv(
        STAGE010_CONFIG_PATH,
        encoding="utf-8-sig",
    )
    df = clean_columns(df)

    required = {
        "model",
        "selection_status",
        "configuration",
        "threshold_percentile",
        "threshold",
        "f1",
        "normal_far",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Stage 010 config file ขาดคอลัมน์: "
            f"{sorted(missing)}"
        )

    df["model"] = (
        df["model"]
        .astype(str)
        .str.strip()
    )

    configs: dict[str, dict[str, Any]] = {}
    thresholds: dict[str, dict[str, Any]] = {}

    allowed_percentiles = {
        80, 85, 88, 90, 92, 95, 97, 98, 99
    }

    for model_name in MODEL_NAMES:
        rows = df.loc[
            df["model"] == model_name
        ]

        if len(rows) != 1:
            raise ValueError(
                f"Stage 010 ต้องมี {model_name} "
                f"1 แถว แต่พบ {len(rows)} แถว"
            )

        row = rows.iloc[0]

        status = str(
            row["selection_status"]
        ).strip().upper()

        if status != "SELECTED":
            raise ValueError(
                f"{model_name} จาก Stage 010 ยังไม่ใช่ SELECTED: "
                f"{status}"
            )

        try:
            config = json.loads(
                str(row["configuration"])
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Configuration ของ {model_name} "
                f"ไม่ใช่ JSON ที่ถูกต้อง"
            ) from exc

        if not isinstance(config, dict) or not config:
            raise ValueError(
                f"Configuration ของ {model_name} "
                "ต้องเป็น JSON object ที่ไม่ว่าง"
            )

        try:
            percentile_float = float(
                row["threshold_percentile"]
            )
            threshold = float(row["threshold"])
            validation_f1 = float(row["f1"])
            validation_far = float(row["normal_far"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"ค่าที่ล็อกของ {model_name} จาก Stage 010 ไม่ถูกต้อง"
            ) from exc

        if (
            not np.isfinite(percentile_float)
            or not percentile_float.is_integer()
        ):
            raise ValueError(
                f"Threshold percentile ของ {model_name} ไม่ถูกต้อง"
            )

        percentile = int(percentile_float)

        if percentile not in allowed_percentiles:
            raise ValueError(
                f"Stage 010 มี Percentile ที่ไม่อนุญาต: "
                f"P{percentile}"
            )

        if not np.isfinite(threshold):
            raise ValueError(
                f"Threshold ของ {model_name} เป็น NaN/Inf"
            )

        if not np.isfinite(validation_f1):
            raise ValueError(
                f"F1 ของ {model_name} จาก Stage 010 ไม่ถูกต้อง"
            )

        if (
            not np.isfinite(validation_far)
            or validation_far > MAX_NORMAL_FAR + EPSILON
        ):
            raise ValueError(
                f"Normal FAR ของ {model_name} จาก Stage 010 "
                f"เกิน {MAX_NORMAL_FAR:.0%}: "
                f"{validation_far:.6f}"
            )

        configs[model_name] = config

        thresholds[model_name] = {
            "percentile": percentile,
            "threshold": threshold,
            "stage010_validation_f1": validation_f1,
            "stage010_validation_normal_far": validation_far,
        }

    # Explicitly verify the parameters selected in the current 010 run.
    expected_configs = {
        "Isolation Forest": {
            "n_estimators": 1000,
            "contamination": 0.10,
            "max_samples": "auto",
            "max_features": 1.0,
            "random_state": 42,
            "n_jobs": -1,
        },
        "One-Class SVM": {
            "kernel": "rbf",
            "nu": 0.01,
            "gamma": "scale",
        },
        "LOF": {
            "n_neighbors": 10,
            "contamination": 0.05,
            "novelty": True,
            "metric": "minkowski",
            "p": 2,
            "n_jobs": -1,
        },
    }

    for model_name in MODEL_NAMES:
        expected = expected_configs[model_name]
        actual = configs[model_name]

        for key, value in expected.items():
            if actual.get(key) != value:
                raise ValueError(
                    f"{model_name} จาก Stage 010 มีค่า {key} "
                    f"ไม่ตรงกับ configuration ที่กำหนด\n"
                    f"พบ: {actual.get(key)!r}\n"
                    f"ต้องการ: {value!r}"
                )

    print("[PASS] Stage 010 selected configurations loaded")
    print("[PASS] Stage 010 locked thresholds loaded")

    for model_name in MODEL_NAMES:
        info = thresholds[model_name]
        print(f"\nMODEL: {model_name}")
        print(f"Configuration: {configs[model_name]}")
        print(
            f"Threshold: P{info['percentile']} | "
            f"value={info['threshold']:.12f}"
        )

    return configs, thresholds


# ============================================================
# 6. LOAD STAGE 011 STATISTICAL LOCK
# ============================================================

def load_statistical_lock(
    features: list[str],
) -> dict[str, Any]:
    if not STATISTICAL_MODEL_PATH.is_file():
        raise FileNotFoundError(
            "ไม่พบ Statistical Threshold จาก Stage 011:\n"
            f"{STATISTICAL_MODEL_PATH}"
        )

    try:
        with STATISTICAL_MODEL_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(
            f"อ่าน Stage 011 ไม่สำเร็จ: {exc}"
        ) from exc

    if data.get("stage") != "011":
        raise ValueError(
            "ไฟล์ Statistical Threshold ไม่ใช่ Stage 011"
        )

    if data.get("method") != "Statistical Threshold":
        raise ValueError(
            "Method ใน Stage 011 ไม่ถูกต้อง"
        )

    if data.get("dataset") != "CIC-IDS2017":
        raise ValueError(
            "Dataset ใน Stage 011 ไม่ใช่ CIC-IDS2017"
        )

    if data.get("window") != "10 seconds":
        raise ValueError(
            "Window ใน Stage 011 ไม่ใช่ 10 seconds"
        )

    if data.get("features") != features:
        raise ValueError(
            "Features ใน Stage 011 ไม่ตรงกับ Behavioral Features ของ Stage 007\n"
            f"พบ      : {data.get('features')}\n"
            f"ต้องการ : {features}"
        )

    try:
        percentile = int(
            data["threshold_percentile"]
        )
        threshold = float(
            data["threshold"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Stage 011 ไม่มี Percentile/Threshold ที่ถูกต้อง"
        ) from exc

    if percentile not in {
        80, 85, 88, 90, 92, 95, 97, 98, 99
    }:
        raise ValueError(
            f"Stage 011 Percentile ไม่ถูกต้อง: P{percentile}"
        )

    if not np.isfinite(threshold):
        raise ValueError(
            "Stage 011 Threshold เป็น NaN/Inf"
        )

    reference = data.get("reference")

    if not isinstance(reference, dict):
        raise ValueError(
            "Stage 011 ไม่มี Statistical Reference"
        )

    for feature in features:
        if feature not in reference:
            raise ValueError(
                f"Stage 011 Reference ขาด Feature: {feature}"
            )

        item = reference[feature]

        if not isinstance(item, dict):
            raise ValueError(
                f"Reference ของ {feature} ไม่ถูกต้อง"
            )

        try:
            median = float(item["median"])
            scale = float(item["scale"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Reference ของ {feature} ไม่มี median/scale"
            ) from exc

        if (
            not np.isfinite(median)
            or not np.isfinite(scale)
            or scale <= 0
        ):
            raise ValueError(
                f"Reference ของ {feature} มีค่าไม่ถูกต้อง"
            )

    validation_metrics = data.get(
        "validation_metrics"
    )

    if not isinstance(validation_metrics, dict):
        raise ValueError(
            "Stage 011 ไม่มี validation_metrics"
        )

    print(
        "\n[PASS] Stage 011 statistical lock loaded"
    )
    print(
        f"Statistical Threshold: "
        f"P{percentile} | value={threshold:.12f}"
    )

    return data


# ============================================================
# 7. LOAD DATA SPLIT
# ============================================================

def normalize_binary_label(value: Any) -> int | None:
    if pd.isna(value):
        return None

    if isinstance(value, (bool, np.bool_)):
        return int(value)

    if isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        number = float(value)

        if number == 0.0:
            return 0

        if number == 1.0:
            return 1

        return None

    token = (
        str(value)
        .strip()
        .lower()
    )

    if token in BENIGN_VALUES:
        return 0

    if token in ATTACK_VALUES:
        return 1

    return None


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
    df = clean_columns(df)

    required = {
        "source_file",
        "window_id",
        "binary_label",
        "split",
        *features,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{split_name} ขาดคอลัมน์: "
            f"{sorted(missing)}"
        )

    if df.empty:
        raise ValueError(
            f"{split_name} ไม่มีข้อมูล"
        )

    actual_split = (
        df["split"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if not actual_split.eq(split_name).all():
        raise ValueError(
            f"{split_name} มีค่า Split ไม่ถูกต้อง"
        )

    if df[
        ["source_file", "window_id", "binary_label"]
    ].isna().any().any():
        raise ValueError(
            f"{split_name} มี Metadata ว่าง"
        )

    if df.duplicated(
        subset=["source_file", "window_id"]
    ).any():
        raise ValueError(
            f"{split_name} มี Window ซ้ำ"
        )

    for feature in features:
        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

        values = df[feature].to_numpy(
            dtype=np.float64
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"{split_name}: {feature} มี NaN/Inf"
            )

    labels = df["binary_label"].apply(
        normalize_binary_label
    )

    if labels.isna().any():
        raise ValueError(
            f"{split_name} มี Binary Label ที่ไม่รู้จัก"
        )

    df["_y"] = labels.astype(
        np.int8
    )

    print(
        f"[PASS] {split_name}: "
        f"{len(df):,} rows, "
        f"{len(features)} features"
    )

    return df


# ============================================================
# 8. VALIDATE SPLIT INTEGRITY
# ============================================================

def validate_splits(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:

    y_train = train_df["_y"].to_numpy(
        dtype=np.int8
    )

    y_validation = validation_df["_y"].to_numpy(
        dtype=np.int8
    )

    if not np.all(y_train == 0):
        raise ValueError(
            "TRAIN ต้องเป็น Normal-only"
        )

    if not np.any(y_validation == 0):
        raise ValueError(
            "VALIDATION ไม่มี Normal"
        )

    if not np.any(y_validation == 1):
        raise ValueError(
            "VALIDATION ไม่มี Attack"
        )

    train_sources = set(
        train_df["source_file"]
        .astype(str)
        .str.strip()
    )

    validation_sources = set(
        validation_df["source_file"]
        .astype(str)
        .str.strip()
    )

    if train_sources.intersection(
        validation_sources
    ):
        raise ValueError(
            "TRAIN และ VALIDATION ใช้ Source File ซ้ำกัน"
        )

    train_keys = set(
        zip(
            train_df["source_file"]
            .astype(str)
            .str.strip(),
            train_df["window_id"]
            .astype(int),
        )
    )

    validation_keys = set(
        zip(
            validation_df["source_file"]
            .astype(str)
            .str.strip(),
            validation_df["window_id"]
            .astype(int),
        )
    )

    if train_keys.intersection(
        validation_keys
    ):
        raise ValueError(
            "TRAIN และ VALIDATION มี Window ซ้ำกัน"
        )

    print("[PASS] TRAIN is Normal-only")
    print(
        "[PASS] VALIDATION contains Normal and Attack"
    )
    print(
        "[PASS] TRAIN and VALIDATION source separation"
    )
    print(
        "[PASS] No duplicated windows between splits"
    )

    print(
        "\nVALIDATION LABEL DISTRIBUTION"
    )
    print("-" * 78)
    print(
        f"Normal : {int(np.sum(y_validation == 0)):,}"
    )
    print(
        f"Attack : {int(np.sum(y_validation == 1)):,}"
    )

    return y_train, y_validation


# ============================================================
# 9. FEATURE MATRIX
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
            f"{description} มี NaN/Inf"
        )

    return X


# ============================================================
# 10. BUILD FINAL ML MODELS
# ============================================================

def build_models(
    model_configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:

    return {
        "Isolation Forest": IsolationForest(
            **model_configs["Isolation Forest"]
        ),
        "One-Class SVM": OneClassSVM(
            **model_configs["One-Class SVM"]
        ),
        "LOF": LocalOutlierFactor(
            **model_configs["LOF"]
        ),
    }


# ============================================================
# 11. FIT FINAL ML MODELS USING TRAIN ONLY
# ============================================================

def fit_models(
    models: dict[str, Any],
    X_train: np.ndarray,
) -> dict[str, Any]:

    fitted = {}

    for name, model in models.items():
        print(f"\n[FIT] {name}")

        with warnings.catch_warnings(
            record=True
        ) as caught:
            warnings.simplefilter("always")
            model.fit(X_train)

        fitted[name] = model

        print("[PASS] Model fitted on TRAIN only")

        if caught:
            print(
                f"[WARNING] {name}: "
                f"{len(caught)} warning(s)"
            )

            for item in caught[:5]:
                print(f"  - {item.message}")

    return fitted


# ============================================================
# 12. ANOMALY SCORES
# ============================================================

def calculate_model_scores(
    model_name: str,
    model: Any,
    X: np.ndarray,
) -> np.ndarray:

    if model_name == "Isolation Forest":
        scores = -model.score_samples(X)

    elif model_name == "One-Class SVM":
        scores = -model.decision_function(X)

    elif model_name == "LOF":
        scores = -model.score_samples(X)

    else:
        raise ValueError(
            f"ไม่รู้จัก Model: {model_name}"
        )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    ).reshape(-1)

    if len(scores) != len(X):
        raise ValueError(
            f"{model_name}: จำนวน Score "
            "ไม่ตรงกับจำนวนข้อมูล"
        )

    if not np.isfinite(scores).all():
        raise ValueError(
            f"{model_name}: Score มี NaN/Inf"
        )

    return scores


def calculate_statistical_scores(
    X: np.ndarray,
    statistical_lock: dict[str, Any],
    features: list[str],
) -> np.ndarray:

    reference = statistical_lock["reference"]

    medians = np.asarray(
        [
            float(
                reference[feature]["median"]
            )
            for feature in features
        ],
        dtype=np.float64,
    )

    scales = np.asarray(
        [
            float(
                reference[feature]["scale"]
            )
            for feature in features
        ],
        dtype=np.float64,
    )

    z_scores = (
        X - medians
    ) / scales

    scores = np.mean(
        np.square(z_scores),
        axis=1,
    )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    ).reshape(-1)

    if len(scores) != len(X):
        raise ValueError(
            "Statistical Score จำนวนไม่ตรงกับข้อมูล"
        )

    if not np.isfinite(scores).all():
        raise ValueError(
            "Statistical Score มี NaN/Inf"
        )

    return scores


# ============================================================
# 13. METRICS
# ============================================================

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float | int]:

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    normal_total = int(
        np.sum(y_true == 0)
    )

    normal_far = (
        float(fp / normal_total)
        if normal_total > 0
        else np.nan
    )

    if len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(
            y_true,
            scores,
        )

        pr_auc = average_precision_score(
            y_true,
            scores,
        )
    else:
        roc_auc = np.nan
        pr_auc = np.nan

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "normal_far": normal_far,
        "far": normal_far,
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_attack": int(fp + tp),
    }


# ============================================================
# 14. LOCKED VALIDATION AUDIT
# ============================================================

def audit_validation(
    validation_df: pd.DataFrame,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    models: dict[str, Any],
    model_configs: dict[str, dict[str, Any]],
    locked_thresholds: dict[str, dict[str, Any]],
    statistical_lock: dict[str, Any],
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:

    score_df = validation_df.drop(
        columns=["_y"],
        errors="ignore",
    ).copy()

    metrics_rows = []

    # --------------------------------------------------------
    # ML models
    # --------------------------------------------------------

    for model_name in MODEL_NAMES:
        short_name = MODEL_SHORT_NAMES[
            model_name
        ]

        scores = calculate_model_scores(
            model_name,
            models[model_name],
            X_validation,
        )

        threshold_info = locked_thresholds[
            model_name
        ]

        threshold = float(
            threshold_info["threshold"]
        )

        predictions = (
            scores > threshold
        ).astype(np.int8)

        score_df[
            f"{short_name}_anomaly_score"
        ] = scores

        score_df[
            f"{short_name}_prediction"
        ] = predictions

        score_df[
            f"{short_name}_prediction_label"
        ] = np.where(
            predictions == 1,
            "Attack",
            "Normal",
        )

        metrics = calculate_metrics(
            y_validation,
            predictions,
            scores,
        )

        row = {
            "stage": "012",
            "dataset": "CIC-IDS2017",
            "window_seconds": WINDOW_SECONDS,
            "model": model_name,
            "features": "|".join(features),
            "configuration": json.dumps(
                model_configs[model_name],
                ensure_ascii=False,
                sort_keys=True,
            ),
            "threshold_percentile": int(
                threshold_info["percentile"]
            ),
            "locked_threshold": threshold,
            "stage010_validation_f1": float(
                threshold_info[
                    "stage010_validation_f1"
                ]
            ),
            "stage010_validation_normal_far": float(
                threshold_info[
                    "stage010_validation_normal_far"
                ]
            ),
            "prediction_rule": (
                "score > locked threshold"
            ),
            "fit_data": "TRAIN only",
            "evaluation_data": (
                "VALIDATION only"
            ),
            "final_test_used": False,
            **metrics,
        }

        metrics_rows.append(row)

        print(
            f"[PASS] {model_name} locked validation "
            f"| F1={metrics['f1']:.6f} "
            f"| FAR={metrics['normal_far']:.6f}"
        )

        # Refit with identical TRAIN/configuration
        # should reproduce Stage 010's validation result.
        f1_difference = abs(
            metrics["f1"]
            - threshold_info[
                "stage010_validation_f1"
            ]
        )

        far_difference = abs(
            metrics["normal_far"]
            - threshold_info[
                "stage010_validation_normal_far"
            ]
        )

        if f1_difference <= 1e-9:
            print(
                f"[PASS] {model_name} F1 reproduces Stage 010"
            )
        else:
            print(
                f"[WARNING] {model_name} F1 differs from "
                f"Stage 010 by {f1_difference:.12g}"
            )

        if far_difference <= 1e-9:
            print(
                f"[PASS] {model_name} FAR reproduces Stage 010"
            )
        else:
            print(
                f"[WARNING] {model_name} FAR differs from "
                f"Stage 010 by {far_difference:.12g}"
            )

    # --------------------------------------------------------
    # Statistical Threshold
    # --------------------------------------------------------

    statistical_scores = (
        calculate_statistical_scores(
            X_validation,
            statistical_lock,
            features,
        )
    )

    statistical_threshold = float(
        statistical_lock["threshold"]
    )

    statistical_predictions = (
        statistical_scores
        > statistical_threshold
    ).astype(np.int8)

    score_df[
        "statistical_anomaly_score"
    ] = statistical_scores

    score_df[
        "statistical_prediction"
    ] = statistical_predictions

    score_df[
        "statistical_prediction_label"
    ] = np.where(
        statistical_predictions == 1,
        "Attack",
        "Normal",
    )

    statistical_metrics = calculate_metrics(
        y_validation,
        statistical_predictions,
        statistical_scores,
    )

    stage011_validation = (
        statistical_lock[
            "validation_metrics"
        ]
    )

    metrics_rows.append({
        "stage": "012",
        "dataset": "CIC-IDS2017",
        "window_seconds": WINDOW_SECONDS,
        "model": "Statistical Threshold",
        "features": "|".join(features),
        "configuration": (
            "Locked statistical reference "
            "from Stage 011"
        ),
        "threshold_percentile": int(
            statistical_lock[
                "threshold_percentile"
            ]
        ),
        "locked_threshold": statistical_threshold,
        "stage010_validation_f1": np.nan,
        "stage010_validation_normal_far": np.nan,
        "stage011_validation_f1": float(
            stage011_validation["f1"]
        ),
        "stage011_validation_normal_far": float(
            stage011_validation["normal_far"]
        ),
        "prediction_rule": (
            "score > locked threshold"
        ),
        "fit_data": (
            "TRAIN reference from Stage 011"
        ),
        "evaluation_data": (
            "VALIDATION only"
        ),
        "final_test_used": False,
        **statistical_metrics,
    })

    print(
        "[PASS] Statistical Threshold locked validation "
        f"| F1={statistical_metrics['f1']:.6f} "
        f"| FAR={statistical_metrics['normal_far']:.6f}"
    )

    f1_difference = abs(
        statistical_metrics["f1"]
        - float(stage011_validation["f1"])
    )

    far_difference = abs(
        statistical_metrics["normal_far"]
        - float(
            stage011_validation[
                "normal_far"
            ]
        )
    )

    if f1_difference <= 1e-9:
        print(
            "[PASS] Statistical Threshold F1 "
            "reproduces Stage 011"
        )
    else:
        print(
            "[WARNING] Statistical Threshold F1 "
            f"differs from Stage 011 by "
            f"{f1_difference:.12g}"
        )

    if far_difference <= 1e-9:
        print(
            "[PASS] Statistical Threshold FAR "
            "reproduces Stage 011"
        )
    else:
        print(
            "[WARNING] Statistical Threshold FAR "
            f"differs from Stage 011 by "
            f"{far_difference:.12g}"
        )

    metrics_df = pd.DataFrame(
        metrics_rows
    )

    return score_df, metrics_df


# ============================================================
# 15. SAVE FINAL ARTIFACTS
# ============================================================

def save_final_artifacts(
    models: dict[str, Any],
    model_configs: dict[str, dict[str, Any]],
    locked_thresholds: dict[str, dict[str, Any]],
    statistical_lock: dict[str, Any],
    validation_scores: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    features: list[str],
    train_rows: int,
) -> list[Path]:

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved: list[Path] = []

    # --------------------------------------------------------
    # Individual ML models
    # --------------------------------------------------------

    individual_paths = {
        "Isolation Forest":
            MODEL_DIR
            / "cic_final_isolation_forest_10s.joblib",

        "One-Class SVM":
            MODEL_DIR
            / "cic_final_ocsvm_10s.joblib",

        "LOF":
            MODEL_DIR
            / "cic_final_lof_10s.joblib",
    }

    for name, path in individual_paths.items():
        atomic_joblib(
            models[name],
            path,
        )
        saved.append(path)
        print(f"[SAVED] {path}")

    # --------------------------------------------------------
    # Unified bundle
    # --------------------------------------------------------

    bundle = {
        "stage": "012",
        "dataset": "CIC-IDS2017",
        "window": "10 seconds",
        "features": list(features),
        "feature_selection": "NOT PERFORMED",
        "train_rows": int(train_rows),
        "models": models,
        "model_configs": model_configs,
        "locked_thresholds": locked_thresholds,
        "statistical_threshold": statistical_lock,
        "score_direction": (
            "higher_is_more_anomalous"
        ),
        "prediction_rule": (
            "score > locked threshold"
        ),
        "fit_data": "TRAIN only",
        "validation_data": "VALIDATION only",
        "final_test_loaded": False,
        "retuning_performed": False,
        "feature_reselection_performed": False,
        "source_files": {
            "stage007_train_scaled": str(TRAIN_PATH),
            "stage007_validation_scaled": str(VALIDATION_PATH),
            "stage010": str(STAGE010_CONFIG_PATH),
            "stage011": str(STATISTICAL_MODEL_PATH),
        },
    }

    bundle_path = (
        MODEL_DIR
        / "cic_final_anomaly_models_10s.joblib"
    )

    atomic_joblib(
        bundle,
        bundle_path,
    )
    saved.append(bundle_path)
    print(f"[SAVED] {bundle_path}")

    # --------------------------------------------------------
    # Locked configuration record
    # --------------------------------------------------------

    config_rows = []

    for name in MODEL_NAMES:
        info = locked_thresholds[name]

        config_rows.append({
            "stage": "012",
            "dataset": "CIC-IDS2017",
            "model": name,
            "features": "|".join(features),
            "configuration": json.dumps(
                model_configs[name],
                ensure_ascii=False,
                sort_keys=True,
            ),
            "threshold_percentile": int(
                info["percentile"]
            ),
            "locked_threshold": float(
                info["threshold"]
            ),
            "stage010_validation_f1": float(
                info[
                    "stage010_validation_f1"
                ]
            ),
            "stage010_validation_normal_far": float(
                info[
                    "stage010_validation_normal_far"
                ]
            ),
            "fit_data": "TRAIN only",
            "threshold_source": "STAGE 010",
            "retuned": False,
        })

    config_rows.append({
        "stage": "012",
        "dataset": "CIC-IDS2017",
        "model": "Statistical Threshold",
        "features": "|".join(features),
        "configuration": json.dumps(
            {
                "method":
                    "Statistical Threshold",
                "score_formula":
                    statistical_lock[
                        "score_formula"
                    ],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "threshold_percentile": int(
            statistical_lock[
                "threshold_percentile"
            ]
        ),
        "locked_threshold": float(
            statistical_lock["threshold"]
        ),
        "stage010_validation_f1": np.nan,
        "stage010_validation_normal_far": np.nan,
        "stage011_validation_f1": float(
            statistical_lock[
                "validation_metrics"
            ]["f1"]
        ),
        "stage011_validation_normal_far": float(
            statistical_lock[
                "validation_metrics"
            ]["normal_far"]
        ),
        "fit_data": (
            "TRAIN reference from STAGE 011"
        ),
        "threshold_source": "STAGE 011",
        "retuned": False,
    })

    config_path = (
        MODEL_DIR
        / "cic_final_model_configs_locked_10s.csv"
    )

    atomic_csv(
        pd.DataFrame(config_rows),
        config_path,
    )
    saved.append(config_path)
    print(f"[SAVED] {config_path}")

    # --------------------------------------------------------
    # Validation scores
    # --------------------------------------------------------

    score_path = (
        MODEL_DIR
        / "cic_final_validation_scores_10s.csv"
    )

    atomic_csv(
        validation_scores,
        score_path,
    )
    saved.append(score_path)
    print(f"[SAVED] {score_path}")

    # --------------------------------------------------------
    # Validation metrics
    # --------------------------------------------------------

    metrics_path = (
        MODEL_DIR
        / "cic_final_validation_metrics_10s.csv"
    )

    atomic_csv(
        validation_metrics,
        metrics_path,
    )
    saved.append(metrics_path)
    print(f"[SAVED] {metrics_path}")

    # --------------------------------------------------------
    # Training summary
    # --------------------------------------------------------

    summary_rows = []

    for name in [
        *MODEL_NAMES,
        "Statistical Threshold",
    ]:
        row = validation_metrics.loc[
            validation_metrics["model"] == name
        ].iloc[0]

        if name in MODEL_NAMES:
            threshold_info = (
                locked_thresholds[name]
            )

            configuration = json.dumps(
                model_configs[name],
                ensure_ascii=False,
                sort_keys=True,
            )

            threshold_source = "STAGE 010"
            status = "SUCCESS"
            percentile = int(
                threshold_info["percentile"]
            )
            locked_threshold = float(
                threshold_info["threshold"]
            )
        else:
            configuration = (
                "Locked statistical reference"
            )
            threshold_source = "STAGE 011"
            status = "LOADED_FROM_STAGE_011"
            percentile = int(
                statistical_lock[
                    "threshold_percentile"
                ]
            )
            locked_threshold = float(
                statistical_lock["threshold"]
            )

        summary_rows.append({
            "stage": "012",
            "dataset": "CIC-IDS2017",
            "model": name,
            "status": status,
            "train_rows": int(train_rows),
            "features": "|".join(features),
            "configuration": configuration,
            "threshold_percentile": percentile,
            "locked_threshold": locked_threshold,
            "validation_accuracy": float(
                row["accuracy"]
            ),
            "validation_precision": float(
                row["precision"]
            ),
            "validation_recall": float(
                row["recall"]
            ),
            "validation_f1": float(
                row["f1"]
            ),
            "validation_normal_far": float(
                row["normal_far"]
            ),
            "threshold_source": threshold_source,
            "fit_data": "TRAIN only",
            "final_test_used": False,
        })

    summary_path = (
        MODEL_DIR
        / "cic_final_model_training_summary_10s.csv"
    )

    atomic_csv(
        pd.DataFrame(summary_rows),
        summary_path,
    )
    saved.append(summary_path)
    print(f"[SAVED] {summary_path}")

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = {
        "stage": "012",
        "dataset": "CIC-IDS2017",
        "window": "10 seconds",
        "features": list(features),
        "train_rows": int(train_rows),
        "validation_rows": int(
            len(validation_scores)
        ),
        "models": list(MODEL_NAMES),
        "model_configs": model_configs,
        "locked_thresholds": {
            name: {
                "percentile": int(
                    locked_thresholds[name][
                        "percentile"
                    ]
                ),
                "threshold": float(
                    locked_thresholds[name][
                        "threshold"
                    ]
                ),
            }
            for name in MODEL_NAMES
        },
        "statistical_threshold": {
            "percentile": int(
                statistical_lock[
                    "threshold_percentile"
                ]
            ),
            "threshold": float(
                statistical_lock["threshold"]
            ),
        },
        "feature_selection_source": "NOT PERFORMED; all 13 Behavioral Features from Stage 007 used",
        "parameter_source": (
            str(STAGE010_CONFIG_PATH)
        ),
        "statistical_threshold_source": (
            str(STATISTICAL_MODEL_PATH)
        ),
        "fit_data": "TRAIN only",
        "validation_data": "VALIDATION only",
        "final_test_loaded": False,
        "retuning_performed": False,
        "feature_reselection_performed": False,
    }

    manifest_path = (
        MODEL_DIR
        / "cic_final_model_manifest_10s.json"
    )

    atomic_json(
        manifest,
        manifest_path,
    )
    saved.append(manifest_path)
    print(f"[SAVED] {manifest_path}")

    return saved


# ============================================================
# 16. MAIN
# ============================================================

def main() -> None:

    print("=" * 78)
    print("STAGE 012 — CIC-IDS2017 FINAL MODEL TRAINING")
    print("=" * 78)

    print(f"TRAIN      : {TRAIN_PATH}")
    print(f"VALIDATION : {VALIDATION_PATH}")
    print(f"FEATURES   : {len(FEATURE_COLUMNS)} Behavioral Features")
    print(f"STAGE 010  : {STAGE010_CONFIG_PATH}")
    print(f"STAGE 011  : {STATISTICAL_MODEL_PATH}")
    print(f"OUTPUT     : {MODEL_DIR}")
    print("FINAL TEST : NOT LOADED")
    print("Retuning   : NOT PERFORMED")
    print("Feature re-selection : NOT PERFORMED")
    print("=" * 78)

    # --------------------------------------------------------
    # 1. Load all 13 Behavioral Features from Stage 007
    # --------------------------------------------------------

    print(
        "\n[1] Load selected features "
        "from STAGE 008"
    )

    features = load_features()

    # --------------------------------------------------------
    # 2. Load Stage 010
    # --------------------------------------------------------

    print(
        "\n[2] Load locked configurations "
        "from STAGE 010"
    )

    model_configs, locked_thresholds = (
        load_stage010()
    )

    # --------------------------------------------------------
    # 3. Load Stage 011
    # --------------------------------------------------------

    print(
        "\n[3] Load locked Statistical Threshold "
        "from STAGE 011"
    )

    statistical_lock = load_statistical_lock(
        features
    )

    # --------------------------------------------------------
    # 4. Load TRAIN / VALIDATION
    # --------------------------------------------------------

    print(
        "\n[4] Load TRAIN and VALIDATION"
    )

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

    # --------------------------------------------------------
    # 5. Validate split integrity
    # --------------------------------------------------------

    print(
        "\n[5] Validate split integrity"
    )

    y_train, y_validation = validate_splits(
        train_df,
        validation_df,
    )

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

    print(
        f"TRAIN matrix      : {X_train.shape}"
    )

    print(
        f"VALIDATION matrix : {X_validation.shape}"
    )

    print(
        "[PASS] Using all 13 Behavioral Features "
        "without feature selection"
    )

    # --------------------------------------------------------
    # 6. Build models
    # --------------------------------------------------------

    print(
        "\n[6] Build FINAL ML models "
        "from STAGE 010"
    )

    models = build_models(
        model_configs
    )

    for name in MODEL_NAMES:
        print(f"\n{name}")
        print(
            f"Configuration: "
            f"{model_configs[name]}"
        )

    # --------------------------------------------------------
    # 7. Fit TRAIN only
    # --------------------------------------------------------

    print(
        "\n[7] Fit FINAL ML models "
        "using TRAIN only"
    )

    fitted_models = fit_models(
        models,
        X_train,
    )

    # --------------------------------------------------------
    # 8. Validation audit
    # --------------------------------------------------------

    print(
        "\n[8] Audit VALIDATION using "
        "LOCKED thresholds"
    )

    validation_scores, validation_metrics = (
        audit_validation(
            validation_df=validation_df,
            X_validation=X_validation,
            y_validation=y_validation,
            models=fitted_models,
            model_configs=model_configs,
            locked_thresholds=locked_thresholds,
            statistical_lock=statistical_lock,
            features=features,
        )
    )

    # --------------------------------------------------------
    # 9. Save artifacts
    # --------------------------------------------------------

    print(
        "\n[9] Save FINAL model artifacts"
    )

    saved_files = save_final_artifacts(
        models=fitted_models,
        model_configs=model_configs,
        locked_thresholds=locked_thresholds,
        statistical_lock=statistical_lock,
        validation_scores=validation_scores,
        validation_metrics=validation_metrics,
        features=features,
        train_rows=len(train_df),
    )

    # --------------------------------------------------------
    # 10. Final report
    # --------------------------------------------------------

    print("\n" + "=" * 78)
    print("STAGE 012 — SUCCESS")
    print("=" * 78)

    print(
        f"TRAIN rows       : {len(train_df):,}"
    )
    print(
        f"VALIDATION rows  : {len(validation_df):,}"
    )
    print(
        f"Features         : {len(features)}"
    )
    print(
        "Features used    : "
        + ", ".join(features)
    )
    print(
        "Fitted models    : "
        "Isolation Forest, One-Class SVM, LOF"
    )
    print(
        "Configurations   : LOCKED FROM STAGE 010"
    )
    print(
        "Thresholds       : LOCKED FROM STAGE 010"
    )
    print(
        "Statistical      : LOCKED FROM STAGE 011"
    )
    print(
        "Final Test       : NOT LOADED"
    )
    print(
        "Retuning         : NOT PERFORMED"
    )
    print(
        "Feature selection: NOT PERFORMED"
    )

    print(
        "\nLOCKED THRESHOLDS"
    )
    print("-" * 78)

    for name in MODEL_NAMES:
        info = locked_thresholds[name]

        print(
            f"{name:<20} "
            f"P{info['percentile']:<2d} | "
            f"{info['threshold']:.12f}"
        )

    print(
        f"{'Statistical Threshold':<20} "
        f"P{statistical_lock['threshold_percentile']:<2d} | "
        f"{statistical_lock['threshold']:.12f}"
    )

    print(
        "\nVALIDATION AUDIT"
    )
    print("-" * 78)

    display_columns = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "normal_far",
        "roc_auc",
        "pr_auc",
    ]

    print(
        validation_metrics[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print(
        "\nSAVED FILES"
    )
    print("-" * 78)

    for path in saved_files:
        print(path)

    print("=" * 78)
    print("STAGE 012 COMPLETED")
    print("=" * 78)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n" + "=" * 78)
        print("STAGE 012 FAILED")
        print("=" * 78)
        print(f"Error: {exc}")
        sys.exit(1)
