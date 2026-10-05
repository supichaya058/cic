from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# STAGE 013 — CIC-IDS2017 FINAL TEST EVALUATION
# ============================================================
# INPUT:
#   data/time_windows/cic_train_scaled_10s.csv
#   data/time_windows/cic_validation_scaled_10s.csv
#   data/time_windows/cic_final_test_scaled_10s.csv
#   data/models/cic_final_anomaly_models_10s.joblib
#
# All 13 Behavioral Features from Stage 007 are used.
#
# EVALUATION:
#   Isolation Forest, One-Class SVM, LOF,
#   Statistical Threshold
#
# FINAL TEST only:
#   No retraining, no tuning, no threshold reselection.
#   Use locked models and thresholds from STAGE 012.
#   Use Statistical Reference locked in STAGE 011.
# ============================================================


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "time_windows"
MODEL_DIR = BASE_DIR / "data" / "models"

TRAIN_PATH = DATA_DIR / "cic_train_scaled_10s.csv"
VALIDATION_PATH = DATA_DIR / "cic_validation_scaled_10s.csv"
FINAL_TEST_PATH = DATA_DIR / "cic_final_test_scaled_10s.csv"
MODEL_BUNDLE_PATH = MODEL_DIR / "cic_final_anomaly_models_10s.joblib"

PREDICTIONS_PATH = MODEL_DIR / "cic_final_test_predictions_10s.csv"
METRICS_PATH = MODEL_DIR / "cic_final_test_metrics_10s.csv"
CONFUSION_PATH = MODEL_DIR / "cic_final_test_confusion_matrices_10s.csv"
ATTACK_TYPE_PATH = MODEL_DIR / "cic_final_test_attack_type_metrics_10s.csv"
SUMMARY_PATH = MODEL_DIR / "cic_final_test_evaluation_summary_10s.csv"

MODEL_NAMES = [
    "Isolation Forest",
    "One-Class SVM",
    "LOF",
    "Statistical Threshold",
]

ML_MODEL_NAMES = MODEL_NAMES[:3]

MODEL_SHORT_NAMES = {
    "Isolation Forest": "if",
    "One-Class SVM": "ocsvm",
    "LOF": "lof",
    "Statistical Threshold": "statistical",
}

# All 13 Behavioral Features from Stage 007.
# No Feature Selection is performed in this pipeline.
EXPECTED_FEATURES = [
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

BENIGN_VALUES = {
    "normal", "benign", "legitimate", "clean", "background",
    "false", "no", "0", "0.0",
}

ATTACK_VALUES = {
    "attack", "malicious", "true", "yes", "1", "1.0",
}

GENERIC_ATTACK_VALUES = (
    BENIGN_VALUES
    | ATTACK_VALUES
    | {"", "nan", "none", "null", "unknown", "unknown attack"}
)


# ============================================================
# 2. FILE UTILITIES
# ============================================================

def clean_columns(df: pd.DataFrame, description: str) -> pd.DataFrame:
    result = df.copy()
    result.columns = (
        result.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )
    duplicated = result.columns[result.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(f"{description} มีชื่อคอลัมน์ซ้ำ: {duplicated}")
    return result


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    try:
        df.to_csv(temp_path, index=False, encoding="utf-8-sig")
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def source_key(series: pd.Series) -> pd.Series:
    """Normalize paths to source basenames for window identity checks."""
    return (
        series.astype(str)
        .str.replace("\\", "/", regex=False)
        .str.strip()
        .str.split("/")
        .str[-1]
        .str.lower()
    )


def window_keys(df: pd.DataFrame) -> set[tuple[str, int]]:
    return set(
        zip(
            source_key(df["source_file"]).tolist(),
            df["window_id"].astype("int64").tolist(),
        )
    )


# ============================================================
# 3. LOAD MODEL FEATURES
# ============================================================

def load_features() -> list[str]:
    """Use all 13 Behavioral Features from Stage 007."""

    features = list(EXPECTED_FEATURES)

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
# 4. LOAD AND VALIDATE MODEL BUNDLE FROM STAGE 012
# ============================================================

def load_model_bundle(features: list[str]) -> dict[str, Any]:
    if not MODEL_BUNDLE_PATH.is_file():
        raise FileNotFoundError(
            "ไม่พบ Final Model Bundle จาก STAGE 012:\n"
            f"{MODEL_BUNDLE_PATH}\n"
            "กรุณารัน STAGE 012 ก่อน"
        )

    try:
        bundle = joblib.load(MODEL_BUNDLE_PATH)
    except Exception as exc:
        raise RuntimeError(f"โหลด Model Bundle ไม่สำเร็จ: {exc}") from exc

    if not isinstance(bundle, dict):
        raise ValueError("Final Model Bundle มีรูปแบบไม่ถูกต้อง")
    if bundle.get("stage") != "012":
        raise ValueError("Model Bundle ไม่ใช่ผลลัพธ์จาก STAGE 012")
    if bundle.get("dataset") != "CIC-IDS2017":
        raise ValueError("Model Bundle ไม่ใช่ CIC-IDS2017")
    if bundle.get("window") != "10 seconds":
        raise ValueError("Model Bundle ไม่ได้ใช้ Time Window 10 วินาที")
    if bundle.get("features") != features:
        raise ValueError(
            "Features ใน STAGE 012 ไม่ตรงกับ Feature List ปัจจุบัน\n"
            f"Bundle: {bundle.get('features')}\n"
            f"Current: {features}"
        )

    if len(bundle.get("features", [])) != 13:
        raise ValueError(
            "STAGE 012 Bundle ต้องใช้ 13 Behavioral Features"
        )

    print("[PASS] STAGE 012 uses all 13 Behavioral Features")

    models = bundle.get("models")
    configs = bundle.get("model_configs")
    thresholds = bundle.get("locked_thresholds")
    stat = bundle.get("statistical_threshold")

    if not isinstance(models, dict):
        raise ValueError("ไม่พบ Models ใน STAGE 012 Bundle")
    if not isinstance(configs, dict):
        raise ValueError("ไม่พบ Model Configurations ใน STAGE 012")
    if not isinstance(thresholds, dict):
        raise ValueError("ไม่พบ Locked Thresholds ใน STAGE 012")
    if not isinstance(stat, dict):
        raise ValueError("ไม่พบ Statistical Threshold ใน STAGE 012")

    for name in ML_MODEL_NAMES:
        if name not in models:
            raise ValueError(f"Model Bundle ขาดโมเดล: {name}")
        if name not in configs or not isinstance(configs[name], dict):
            raise ValueError(f"Model Bundle ขาด Configuration ที่ถูกต้อง: {name}")
        if name not in thresholds or not isinstance(thresholds[name], dict):
            raise ValueError(f"Model Bundle ขาด Locked Threshold: {name}")

        info = thresholds[name]
        try:
            percentile = int(info["percentile"])
            threshold = float(info["threshold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Locked Threshold ของ {name} ไม่ครบ") from exc

        if percentile not in {80, 85, 88, 90, 92, 95, 97, 98, 99}:
            raise ValueError(f"Threshold Percentile ของ {name} ไม่ถูกต้อง")
        if not np.isfinite(threshold):
            raise ValueError(f"Threshold ของ {name} เป็น NaN หรือ Inf")

    if stat.get("stage") != "011":
        raise ValueError("Statistical Threshold ไม่ใช่ผลจาก STAGE 011")
    if stat.get("method") != "Statistical Threshold":
        raise ValueError("Statistical Threshold Method ไม่ถูกต้อง")
    if stat.get("dataset") != "CIC-IDS2017":
        raise ValueError("Statistical Threshold ไม่ใช่ CIC-IDS2017")
    if stat.get("window") != "10 seconds":
        raise ValueError("Statistical Threshold ไม่ได้ใช้ Time Window 10 วินาที")
    if stat.get("features") != features:
        raise ValueError("Statistical Threshold ใช้ Features ไม่ตรงกัน")

    try:
        stat_threshold = float(stat["threshold"])
        stat_percentile = int(stat["threshold_percentile"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Statistical Threshold ไม่มีค่า Threshold ที่ถูกต้อง") from exc

    if (
        not np.isfinite(stat_threshold)
        or stat_percentile not in {80, 85, 88, 90, 92, 95, 97, 98, 99}
    ):
        raise ValueError("Statistical Threshold หรือ Percentile ไม่ถูกต้อง")

    reference = stat.get("reference")
    if not isinstance(reference, dict):
        raise ValueError("ไม่พบ TRAIN Statistical Reference")

    for feature in features:
        if feature not in reference or not isinstance(reference[feature], dict):
            raise ValueError(f"Statistical Reference ขาด Feature: {feature}")
        try:
            median = float(reference[feature]["median"])
            scale = float(reference[feature]["scale"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Reference ของ {feature} ไม่ครบ") from exc
        if not np.isfinite(median) or not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Reference ของ {feature} ไม่ถูกต้อง")

    print(f"[PASS] Stage 012 Bundle loaded: {MODEL_BUNDLE_PATH}")
    print("[PASS] Three ML models found")
    print("[PASS] Locked thresholds found")
    print("[PASS] Stage 011 Statistical Threshold found")
    return bundle


# ============================================================
# 5. LABEL HANDLING
# ============================================================

def map_binary_label(value: Any) -> int | None:
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if number == 0.0:
            return 0
        if number == 1.0:
            return 1
        return None

    token = str(value).strip().lower()
    if token in BENIGN_VALUES:
        return 0
    if token in ATTACK_VALUES:
        return 1
    return None


def extract_binary_labels(df: pd.DataFrame) -> np.ndarray:
    if "binary_label" not in df.columns:
        raise ValueError("FINAL TEST ไม่มีคอลัมน์ binary_label")

    mapped = df["binary_label"].map(map_binary_label)
    if mapped.isna().any():
        invalid = (
            df.loc[mapped.isna(), "binary_label"]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(f"พบ binary_label ที่ไม่รู้จักใน FINAL TEST: {invalid}")

    y = mapped.to_numpy(dtype=np.int8)
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("binary_label มีค่าอื่นนอกเหนือจาก 0 และ 1")
    if not np.any(y == 0):
        raise ValueError("FINAL TEST ไม่มี Normal/BENIGN")
    if not np.any(y == 1):
        raise ValueError("FINAL TEST ไม่มี Attack")
    return y


# ============================================================
# 6. LOAD AND VALIDATE FINAL TEST
# ============================================================

def load_final_test(
    features: list[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if not FINAL_TEST_PATH.is_file():
        raise FileNotFoundError(
            "ไม่พบ FINAL TEST file:\n"
            f"{FINAL_TEST_PATH}"
        )

    df = clean_columns(
        pd.read_csv(
            FINAL_TEST_PATH,
            low_memory=False,
            encoding="utf-8-sig",
        ),
        "FINAL TEST",
    )

    if df.empty:
        raise ValueError("FINAL TEST ไม่มีข้อมูล")

    required = {
        "source_file", "window_id", "binary_label", "split", *features
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"FINAL TEST ขาดคอลัมน์: {sorted(missing)}")

    split_values = df["split"].astype(str).str.strip().str.upper()
    if set(split_values.unique()) != {"FINAL_TEST"}:
        raise ValueError(
            "ไฟล์ที่โหลดไม่ใช่ FINAL_TEST ทั้งหมด\n"
            f"พบ Split: {sorted(split_values.unique())}"
        )

    if df[["source_file", "window_id", "binary_label"]].isna().any().any():
        raise ValueError("FINAL TEST มี Metadata ว่าง")

    df["source_file"] = df["source_file"].astype(str).str.strip()
    if df["source_file"].eq("").any() or df["source_file"].str.lower().eq("nan").any():
        raise ValueError("FINAL TEST มี source_file ว่าง")

    window_id = pd.to_numeric(df["window_id"], errors="coerce")
    if (
        window_id.isna().any()
        or not np.isfinite(window_id.to_numpy()).all()
        or (window_id < 1).any()
        or (window_id % 1 != 0).any()
    ):
        raise ValueError("FINAL TEST มี window_id ที่ไม่ถูกต้อง")
    df["window_id"] = window_id.astype("int64")

    if df.duplicated(subset=["source_file", "window_id"]).any():
        raise ValueError("FINAL TEST มี Window ซ้ำ")

    for feature in features:
        values = pd.to_numeric(df[feature], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"FINAL TEST: {feature} มี NaN หรือ Inf")
        df[feature] = values.astype("float64")

    y_test = extract_binary_labels(df)
    X_test = df[features].to_numpy(dtype=np.float64)
    if not np.isfinite(X_test).all():
        raise ValueError("FINAL TEST Feature Matrix มี NaN หรือ Inf")

    print(f"[PASS] FINAL TEST loaded: {len(df):,} rows")
    print(f"[PASS] Normal: {int(np.sum(y_test == 0)):,}")
    print(f"[PASS] Attack: {int(np.sum(y_test == 1)):,}")
    print(f"[PASS] Feature matrix: {X_test.shape}")
    return df, X_test, y_test


# ============================================================
# 7. CHECK TRAIN / VALIDATION / FINAL TEST WINDOW OVERLAP
# ============================================================

def identity_keys(path: Path, expected_split: str) -> set[tuple[str, int]]:
    """Load identifiers only; sharing a source file is allowed."""
    if not path.is_file():
        raise FileNotFoundError(f"ไม่พบไฟล์สำหรับตรวจสอบ Split: {path}")

    df = clean_columns(
        pd.read_csv(
            path,
            usecols=lambda c: c in {"source_file", "window_id", "split"},
            low_memory=False,
            encoding="utf-8-sig",
        ),
        expected_split,
    )
    required = {"source_file", "window_id", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{expected_split} ขาดคอลัมน์สำหรับตรวจสอบ: {sorted(missing)}")

    splits = df["split"].astype(str).str.strip().str.upper()
    if set(splits.unique()) != {expected_split}:
        raise ValueError(f"Split ในไฟล์ {expected_split} ไม่ถูกต้อง")

    source = source_key(df["source_file"])
    window = pd.to_numeric(df["window_id"], errors="coerce")
    if (
        source.eq("").any()
        or source.eq("nan").any()
        or window.isna().any()
        or not np.isfinite(window.to_numpy()).all()
        or (window < 1).any()
        or (window % 1 != 0).any()
    ):
        raise ValueError(f"{expected_split} มี source_file หรือ window_id ไม่ถูกต้อง")

    return set(zip(source.tolist(), window.astype("int64").tolist()))


def validate_no_window_overlap(test_df: pd.DataFrame) -> None:
    """Verify exact window identity is absent from TRAIN and VALIDATION."""
    train_keys = identity_keys(TRAIN_PATH, "TRAIN")
    validation_keys = identity_keys(VALIDATION_PATH, "VALIDATION")
    test_keys = window_keys(test_df)

    train_overlap = test_keys.intersection(train_keys)
    validation_overlap = test_keys.intersection(validation_keys)

    if train_overlap:
        raise ValueError(
            "พบ Window ซ้ำระหว่าง TRAIN และ FINAL_TEST\n"
            f"ตัวอย่าง: {sorted(train_overlap)[:10]}"
        )
    if validation_overlap:
        raise ValueError(
            "พบ Window ซ้ำระหว่าง VALIDATION และ FINAL_TEST\n"
            f"ตัวอย่าง: {sorted(validation_overlap)[:10]}"
        )

    print("[PASS] No TRAIN/FINAL_TEST exact window overlap")
    print("[PASS] No VALIDATION/FINAL_TEST exact window overlap")


# ============================================================
# 8. ATTACK TYPE IDENTIFICATION
# ============================================================

def identify_attack_types(
    df: pd.DataFrame,
    y_true: np.ndarray,
) -> tuple[pd.Series, str | None]:
    """Find a detailed attack category column, if available."""
    candidates = [
        "attack_type", "attack_name", "attack_category",
        "traffic_type", "attack_label", "label", "Label", "class",
    ]

    best_column = None
    best_score = 0
    best_priority = len(candidates) + 1
    attack_mask = y_true == 1

    for priority, column in enumerate(candidates):
        if column not in df.columns:
            continue
        values = df.loc[attack_mask, column].dropna().astype(str).str.strip()
        informative = {
            value for value in values
            if value.lower() not in GENERIC_ATTACK_VALUES
        }
        score = len(informative)
        if score > best_score or (
            score == best_score and score > 0 and priority < best_priority
        ):
            best_column = column
            best_score = score
            best_priority = priority

    if best_column is None or best_score == 0:
        print("[INFO] ไม่พบประเภทการโจมตีแบบละเอียด จะจัดกลุ่มเป็น Attack")
        return pd.Series(
            np.where(y_true == 0, "Normal", "Attack"),
            index=df.index,
            dtype="object",
        ), None

    types = []
    for i, value in enumerate(df[best_column].astype("string").str.strip()):
        if y_true[i] == 0:
            types.append("Normal")
        elif pd.isna(value) or str(value).lower() in GENERIC_ATTACK_VALUES:
            types.append("Unknown Attack")
        else:
            types.append(str(value))

    result = pd.Series(types, index=df.index, dtype="object")
    print(f"[PASS] Attack type column: {best_column}")
    print("Attack types found:")
    for attack_type in sorted(result.loc[attack_mask].unique()):
        count = int(((y_true == 1) & result.eq(attack_type).to_numpy()).sum())
        print(f"  - {attack_type}: {count:,}")
    return result, best_column


# ============================================================
# 9. ANOMALY SCORE FUNCTIONS
# ============================================================

def calculate_model_scores(
    name: str,
    model: Any,
    X: np.ndarray,
) -> np.ndarray:
    """Normalize score direction: higher values mean more anomalous."""
    if name == "Isolation Forest":
        scores = -model.score_samples(X)
    elif name == "One-Class SVM":
        scores = -model.decision_function(X)
    elif name == "LOF":
        scores = -model.score_samples(X)
    else:
        raise ValueError(f"ไม่รู้จัก ML model: {name}")

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(scores) != len(X) or not np.isfinite(scores).all():
        raise ValueError(f"{name} ให้ Anomaly Score ไม่ถูกต้อง")
    return scores


def calculate_statistical_scores(
    X: np.ndarray,
    statistical_lock: dict[str, Any],
    features: list[str],
) -> np.ndarray:
    """Use Stage 011 TRAIN reference without refitting or retuning."""
    reference = statistical_lock["reference"]
    medians = np.asarray(
        [float(reference[f]["median"]) for f in features],
        dtype=np.float64,
    )
    scales = np.asarray(
        [float(reference[f]["scale"]) for f in features],
        dtype=np.float64,
    )
    scores = np.mean(np.square((X - medians) / scales), axis=1)
    if not np.isfinite(scores).all():
        raise ValueError("Statistical Threshold มี Score เป็น NaN หรือ Inf")
    return scores


# ============================================================
# 10. EVALUATION METRICS
# ============================================================

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    normal_total = int(tn + fp)
    attack_total = int(tp + fn)
    far = float(fp / normal_total) if normal_total else np.nan
    fnr = float(fn / attack_total) if attack_total else np.nan
    specificity = float(tn / normal_total) if normal_total else np.nan

    if len(np.unique(y_true)) == 2:
        roc_auc = float(roc_auc_score(y_true, scores))
        pr_auc = float(average_precision_score(y_true, scores))
    else:
        roc_auc = np.nan
        pr_auc = np.nan

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "far": far,
        "normal_far": far,
        "fnr": fnr,
        "specificity": specificity,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "normal_support": normal_total,
        "attack_support": attack_total,
        "predicted_normal": int(tn + fn),
        "predicted_attack": int(fp + tp),
    }


# ============================================================
# 11. EVALUATE ALL LOCKED METHODS ON FINAL TEST
# ============================================================

def evaluate_models(
    test_df: pd.DataFrame,
    X_test: np.ndarray,
    y_test: np.ndarray,
    attack_types: pd.Series,
    bundle: dict[str, Any],
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate models using locked parameters and thresholds only."""
    models = bundle["models"]
    model_configs = bundle["model_configs"]
    locked_thresholds = bundle["locked_thresholds"]
    statistical_lock = bundle["statistical_threshold"]

    predictions_df = test_df.copy()
    predictions_df["true_binary_label"] = y_test
    predictions_df["true_binary_label_text"] = np.where(
        y_test == 1, "Attack", "Normal"
    )
    predictions_df["true_attack_type"] = attack_types.to_numpy()

    metrics_rows = []
    confusion_rows = []
    attack_rows = []
    attack_names = sorted(attack_types.loc[y_test == 1].unique().tolist())

    # ML models from Stage 012.
    for name in ML_MODEL_NAMES:
        print(f"\n[EVALUATE] {name}")
        scores = calculate_model_scores(name, models[name], X_test)
        threshold_info = locked_thresholds[name]
        threshold = float(threshold_info["threshold"])
        percentile = int(threshold_info["percentile"])
        predictions = (scores > threshold).astype(np.int8)
        short = MODEL_SHORT_NAMES[name]

        predictions_df[f"{short}_anomaly_score"] = scores
        predictions_df[f"{short}_prediction"] = predictions
        predictions_df[f"{short}_prediction_label"] = np.where(
            predictions == 1, "Attack", "Normal"
        )

        metrics = calculate_metrics(y_test, predictions, scores)
        metrics_rows.append({
            "stage": "013",
            "model": name,
            "configuration": json.dumps(
                model_configs[name], ensure_ascii=False, sort_keys=True
            ),
            "threshold_percentile": percentile,
            "locked_threshold": threshold,
            "score_direction": "higher_is_more_anomalous",
            "prediction_rule": "score > threshold",
            **metrics,
        })
        confusion_rows.append({
            "stage": "013",
            "model": name,
            "threshold_percentile": percentile,
            "threshold": threshold,
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tp": metrics["tp"],
        })

        print(
            f"[PASS] {name} | F1={metrics['f1']:.6f} | "
            f"FAR={metrics['normal_far']:.6f} | "
            f"ROC-AUC={metrics['roc_auc']:.6f} | "
            f"PR-AUC={metrics['pr_auc']:.6f}"
        )

        for attack_type in attack_names:
            mask = (
                (y_test == 1)
                & attack_types.eq(attack_type).to_numpy()
            )
            support = int(mask.sum())
            detected = int(np.sum(predictions[mask] == 1))
            missed = int(np.sum(predictions[mask] == 0))
            attack_rows.append({
                "stage": "013",
                "model": name,
                "attack_type": str(attack_type),
                "support": support,
                "detected": detected,
                "missed": missed,
                "detection_rate": float(detected / support) if support else np.nan,
                "mean_anomaly_score": float(np.mean(scores[mask])) if support else np.nan,
                "threshold": threshold,
            })

    # Statistical Threshold from Stage 011, stored in Stage 012 bundle.
    print("\n[EVALUATE] Statistical Threshold")
    stat_threshold = float(statistical_lock["threshold"])
    stat_percentile = int(statistical_lock["threshold_percentile"])
    stat_scores = calculate_statistical_scores(
        X_test, statistical_lock, features
    )
    stat_predictions = (stat_scores > stat_threshold).astype(np.int8)

    predictions_df["statistical_anomaly_score"] = stat_scores
    predictions_df["statistical_prediction"] = stat_predictions
    predictions_df["statistical_prediction_label"] = np.where(
        stat_predictions == 1, "Attack", "Normal"
    )

    stat_metrics = calculate_metrics(y_test, stat_predictions, stat_scores)
    metrics_rows.append({
        "stage": "013",
        "model": "Statistical Threshold",
        "configuration": "Locked statistical reference from STAGE 011",
        "threshold_percentile": stat_percentile,
        "locked_threshold": stat_threshold,
        "score_direction": "higher_is_more_anomalous",
        "prediction_rule": "score > threshold",
        **stat_metrics,
    })
    confusion_rows.append({
        "stage": "013",
        "model": "Statistical Threshold",
        "threshold_percentile": stat_percentile,
        "threshold": stat_threshold,
        "tn": stat_metrics["tn"],
        "fp": stat_metrics["fp"],
        "fn": stat_metrics["fn"],
        "tp": stat_metrics["tp"],
    })

    print(
        f"[PASS] Statistical Threshold | F1={stat_metrics['f1']:.6f} | "
        f"FAR={stat_metrics['normal_far']:.6f} | "
        f"ROC-AUC={stat_metrics['roc_auc']:.6f} | "
        f"PR-AUC={stat_metrics['pr_auc']:.6f}"
    )

    for attack_type in attack_names:
        mask = (
            (y_test == 1)
            & attack_types.eq(attack_type).to_numpy()
        )
        support = int(mask.sum())
        detected = int(np.sum(stat_predictions[mask] == 1))
        missed = int(np.sum(stat_predictions[mask] == 0))
        attack_rows.append({
            "stage": "013",
            "model": "Statistical Threshold",
            "attack_type": str(attack_type),
            "support": support,
            "detected": detected,
            "missed": missed,
            "detection_rate": float(detected / support) if support else np.nan,
            "mean_anomaly_score": float(np.mean(stat_scores[mask])) if support else np.nan,
            "threshold": stat_threshold,
        })

    return (
        predictions_df,
        pd.DataFrame(metrics_rows),
        pd.DataFrame(confusion_rows),
        pd.DataFrame(attack_rows),
    )


# ============================================================
# 12. SAVE EVALUATION SUMMARY
# ============================================================

def save_summary(
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    features: list[str],
    attack_type_column: str | None,
    metrics_df: pd.DataFrame,
    overlap_checked: bool,
) -> Path:
    summary_rows = [
        {"item": "stage", "value": "013"},
        {"item": "dataset", "value": "CIC-IDS2017"},
        {"item": "window", "value": "10 seconds"},
        {"item": "final_test_rows", "value": int(len(test_df))},
        {"item": "final_test_normal_rows", "value": int(np.sum(y_test == 0))},
        {"item": "final_test_attack_rows", "value": int(np.sum(y_test == 1))},
        {"item": "features_used", "value": "|".join(features)},
        {"item": "feature_selection", "value": "NOT PERFORMED"},
        {"item": "models_evaluated", "value": "|".join(MODEL_NAMES)},
        {"item": "attack_type_column", "value": attack_type_column or "Not available"},
        {"item": "exact_window_overlap_check", "value": "PASS" if overlap_checked else "NOT CHECKED"},
        {"item": "models_retrained", "value": False},
        {"item": "thresholds_retuned", "value": False},
        {"item": "final_test_used_for_tuning", "value": False},
        {"item": "roc_auc_method", "value": "roc_auc_score"},
        {"item": "pr_auc_method", "value": "average_precision_score"},
    ]

    metric_columns = [
        "accuracy", "precision", "recall", "f1", "far", "normal_far",
        "fnr", "specificity", "balanced_accuracy", "roc_auc", "pr_auc",
        "tn", "fp", "fn", "tp",
    ]
    for _, row in metrics_df.iterrows():
        model_key = str(row["model"]).lower().replace(" ", "_")
        for column in metric_columns:
            summary_rows.append({
                "item": f"{model_key}_{column}",
                "value": row[column],
            })

    atomic_csv(pd.DataFrame(summary_rows), SUMMARY_PATH)
    return SUMMARY_PATH


# ============================================================
# 13. MAIN
# ============================================================

def main() -> None:
    print("=" * 78)
    print("STAGE 013 — CIC-IDS2017 FINAL TEST EVALUATION")
    print("=" * 78)
    print(f"TRAIN      : {TRAIN_PATH}")
    print(f"VALIDATION : {VALIDATION_PATH}")
    print(f"FINAL TEST : {FINAL_TEST_PATH}")
    print(f"FEATURES   : {len(EXPECTED_FEATURES)} Behavioral Features")
    print(f"MODEL      : {MODEL_BUNDLE_PATH}")
    print(f"OUTPUT     : {MODEL_DIR}")
    print("RETRAIN    : NO")
    print("RETUNING   : NO")
    print("FINAL TEST : Evaluation only")
    print("=" * 78)

    print("\n[1] Load 13 Behavioral Features")
    features = load_features()

    print("\n[2] Load locked models from STAGE 012")
    bundle = load_model_bundle(features)

    print("\n[3] Load FINAL TEST")
    test_df, X_test, y_test = load_final_test(features)

    print("\n[4] Check split overlap")
    validate_no_window_overlap(test_df)

    print("\n[5] Identify attack types")
    attack_types, attack_type_column = identify_attack_types(
        test_df, y_test
    )

    print("\n[6] Evaluate all locked models on FINAL TEST")
    predictions_df, metrics_df, confusion_df, attack_df = evaluate_models(
        test_df=test_df,
        X_test=X_test,
        y_test=y_test,
        attack_types=attack_types,
        bundle=bundle,
        features=features,
    )

    print("\n[7] Save final evaluation results")
    atomic_csv(predictions_df, PREDICTIONS_PATH)
    print(f"[SAVED] {PREDICTIONS_PATH}")

    atomic_csv(metrics_df, METRICS_PATH)
    print(f"[SAVED] {METRICS_PATH}")

    atomic_csv(confusion_df, CONFUSION_PATH)
    print(f"[SAVED] {CONFUSION_PATH}")

    if not attack_df.empty:
        atomic_csv(attack_df, ATTACK_TYPE_PATH)
        print(f"[SAVED] {ATTACK_TYPE_PATH}")
    else:
        print("[INFO] ไม่มีข้อมูลสำหรับ Attack Type Analysis")

    summary_path = save_summary(
        test_df=test_df,
        y_test=y_test,
        features=features,
        attack_type_column=attack_type_column,
        metrics_df=metrics_df,
        overlap_checked=True,
    )
    print(f"[SAVED] {summary_path}")

    print("\n" + "=" * 78)
    print("STAGE 013 — FINAL TEST RESULTS")
    print("=" * 78)
    print(f"FINAL TEST rows : {len(test_df):,}")
    print(f"Normal          : {int(np.sum(y_test == 0)):,}")
    print(f"Attack          : {int(np.sum(y_test == 1)):,}")
    print(f"Features        : {len(features)} (all 13 Behavioral Features)")
    print(f"Models          : {len(MODEL_NAMES)}")
    print("Retrain         : NO")
    print("Threshold tuning: NO")
    print("Overlap check   : PASS")

    print("\nFINAL TEST METRICS")
    print("-" * 78)
    display_columns = [
        "model", "accuracy", "precision", "recall", "f1",
        "normal_far", "roc_auc", "pr_auc",
    ]
    print(
        metrics_df[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nCONFUSION MATRICES")
    print("-" * 78)
    print(
        confusion_df[["model", "tn", "fp", "fn", "tp"]]
        .to_string(index=False)
    )

    if not attack_df.empty:
        print("\nATTACK TYPE DETECTION")
        print("-" * 78)
        print(
            attack_df[
                [
                    "model", "attack_type", "support", "detected",
                    "missed", "detection_rate",
                ]
            ].to_string(
                index=False,
                float_format=lambda value: f"{value:.6f}",
            )
        )

    print("\nSAVED FILES")
    print("-" * 78)
    print(f"Predictions : {PREDICTIONS_PATH}")
    print(f"Metrics     : {METRICS_PATH}")
    print(f"Confusion   : {CONFUSION_PATH}")
    if not attack_df.empty:
        print(f"Attack types: {ATTACK_TYPE_PATH}")
    print(f"Summary     : {SUMMARY_PATH}")
    print("\nFINAL TEST was used only for final evaluation.")
    print("No model was retrained or retuned.")
    print("\n" + "=" * 78)
    print("STAGE 013 — SUCCESS")
    print("=" * 78)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n" + "=" * 78)
        print("STAGE 013 — FAILED")
        print("=" * 78)
        print(f"Error: {exc}")
        sys.exit(1)