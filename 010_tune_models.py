
from __future__ import annotations

from itertools import product
from pathlib import Path
import json
import os
import sys
import warnings

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
# STAGE 010 — CIC-IDS2017 MODEL CONFIGURATION MATCHING
# ============================================================
# Use the configurations selected for the self-created dataset.
# All 13 Behavioral Features from Stage 007 are used directly.
# Hyperparameters are fixed; only the decision-threshold percentile
# is calibrated using VALIDATION. FINAL_TEST is never loaded here.
#
# Isolation Forest : 1000, 0.10, auto, 1.0
# One-Class SVM    : rbf, nu=0.01, gamma=scale
# LOF              : n_neighbors=10, contamination=0.05
# Thresholds       : [80, 85, 88, 90, 92, 95, 97, 98, 99]
# Selection        : Normal-only FAR <= 15%, then highest Validation F1
# Preprocessing    : Reuse Stage 007 scaled features
# ============================================================


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "time_windows"
MODEL_DIR = BASE_DIR / "data" / "models"

TRAIN_FILE = DATA_DIR / "cic_train_scaled_10s.csv"
VALIDATION_FILE = DATA_DIR / "cic_validation_scaled_10s.csv"

BASELINE_BUNDLE_FILE = MODEL_DIR / "cic_baseline_models_10s.joblib"
BASELINE_METRICS_FILE = MODEL_DIR / "cic_baseline_metrics_10s.csv"
BASELINE_SCORES_FILE = MODEL_DIR / "cic_baseline_validation_scores_10s.csv"

RESULTS_FILE = DATA_DIR / "cic_tuning_results_10s.csv"
SELECTED_CONFIG_FILE = DATA_DIR / "cic_selected_model_configs_10s.csv"
SUMMARY_FILE = DATA_DIR / "cic_tuning_summary_10s.csv"

WINDOW_SECONDS = 10
MAX_NORMAL_FAR = 0.15
EPSILON = 1e-12
THRESHOLD_PERCENTILES = [80, 85, 88, 90, 92, 95, 97, 98, 99]

# The exact selected hyperparameters from the self-created dataset.
IF_N_ESTIMATORS = [1000]
IF_CONTAMINATION = [0.10]
IF_MAX_SAMPLES = ["auto"]
IF_MAX_FEATURES = [1.0]

OCSVM_NU = [0.01]
OCSVM_GAMMA = ["scale"]

LOF_NEIGHBORS = [10]
LOF_CONTAMINATION = [0.05]

EXPECTED_CONFIG_COUNTS = {
    "isolation_forest": 1,
    "one_class_svm": 1,
    "lof": 1,
}

MODEL_NAMES = {
    "isolation_forest": "Isolation Forest",
    "one_class_svm": "One-Class SVM",
    "lof": "LOF",
}

SEARCH_PARAMETERS = {
    "isolation_forest": [
        "n_estimators", "contamination", "max_samples", "max_features"
    ],
    "one_class_svm": ["kernel", "nu", "gamma"],
    "lof": ["n_neighbors", "contamination"],
}

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
    "normal", "benign", "legitimate", "clean", "background",
    "0", "0.0", "false", "no",
}
ATTACK_LABELS = {
    "attack", "malicious", "1", "1.0", "true", "yes",
}

SEARCH_SPACE_DETAILS = {
    "isolation_forest": {
        "search_space_definition": (
            "n_estimators=1000; contamination=0.10; "
            "max_samples='auto'; max_features=1.0 (fixed; matched "
            "to the self-created dataset)"
        ),
        "contamination_handling": (
            "Fixed at 0.10 to match the self-created dataset. "
            "Decision threshold is calibrated from VALIDATION "
            "Normal scores using score_samples()."
        ),
        "grid_comparison_note": (
            "Fixed configuration matched to self-created dataset: "
            "n_estimators=1000, contamination=0.10, "
            "max_samples='auto', max_features=1.0."
        ),
    },
    "one_class_svm": {
        "search_space_definition": (
            "kernel='rbf'; nu=0.01; gamma='scale' (fixed; matched "
            "to the self-created dataset)"
        ),
        "contamination_handling": "Not applicable to One-Class SVM.",
        "grid_comparison_note": (
            "Fixed configuration matched to self-created dataset: "
            "kernel='rbf', nu=0.01, gamma='scale'."
        ),
    },
    "lof": {
        "search_space_definition": (
            "n_neighbors=10; contamination=0.05; novelty=True, "
            "metric='minkowski', p=2 (fixed; matched to the "
            "self-created dataset)"
        ),
        "contamination_handling": (
            "Fixed at 0.05 to match the self-created dataset. "
            "Decision threshold is calibrated from VALIDATION "
            "Normal scores using score_samples()."
        ),
        "grid_comparison_note": (
            "Fixed configuration matched to self-created dataset: "
            "n_neighbors=10, contamination=0.05, novelty=True."
        ),
    },
}


# ============================================================
# 2. FILE AND COLUMN UTILITIES
# ============================================================

def clean_columns(df: pd.DataFrame, description: str) -> pd.DataFrame:
    result = df.copy()
    result.columns = (
        result.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )
    duplicates = result.columns[result.columns.duplicated()].tolist()
    if duplicates:
        raise ValueError(f"{description} มีชื่อคอลัมน์ซ้ำ: {duplicates}")
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


def config_signature(config: dict, model_key: str) -> str:
    active = {key: config.get(key) for key in SEARCH_PARAMETERS[model_key]}
    return json.dumps(
        active, sort_keys=True, separators=(",", ":"), default=str
    )


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
# 4. LOAD BASELINE FROM STAGE 009
# ============================================================

def load_baseline(features: list[str]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    required_files = [
        BASELINE_BUNDLE_FILE,
        BASELINE_METRICS_FILE,
        BASELINE_SCORES_FILE,
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "ไม่พบไฟล์จาก Stage 009:\n"
            + "\n".join(missing)
            + "\nกรุณารัน Stage 009 ให้สำเร็จก่อน"
        )

    bundle = joblib.load(BASELINE_BUNDLE_FILE)
    expected_models = set(MODEL_NAMES.values())
    if not isinstance(bundle, dict) or str(bundle.get("stage")) != "009":
        raise ValueError("Baseline bundle ไม่ถูกต้องหรือไม่ใช่ Stage 009")
    if bundle.get("features") != features:
        raise ValueError(
            "Feature ใน Baseline ไม่ตรงกับชุด Behavioral Features ของ Stage 007\n"
            f"Baseline: {bundle.get('features')}\nCurrent: {features}"
        )

    configs = bundle.get("model_configs")
    models = bundle.get("models")
    if not isinstance(configs, dict) or set(configs) != expected_models:
        raise ValueError("Baseline model_configs ไม่ครบทั้ง 3 โมเดล")
    if not isinstance(models, dict) or not expected_models.issubset(set(models)):
        raise ValueError("Baseline bundle มีโมเดลไม่ครบทั้ง 3 โมเดล")

    metrics = clean_columns(
        pd.read_csv(BASELINE_METRICS_FILE, encoding="utf-8-sig"),
        "Baseline metrics",
    )
    required_metrics = {"model", "f1", "normal_far"}
    missing_metrics = sorted(required_metrics - set(metrics.columns))
    if missing_metrics:
        raise ValueError(f"Baseline metrics ขาดคอลัมน์: {missing_metrics}")
    metrics["model"] = metrics["model"].astype(str).str.strip()
    if metrics["model"].duplicated().any() or set(metrics["model"]) != expected_models:
        raise ValueError("Baseline metrics มีชื่อโมเดลซ้ำหรือไม่ครบ")

    scores = clean_columns(
        pd.read_csv(BASELINE_SCORES_FILE, low_memory=False, encoding="utf-8-sig"),
        "Baseline validation scores",
    )
    required_scores = {"source_file", "window_id", "binary_label"}
    missing_scores = sorted(required_scores - set(scores.columns))
    if missing_scores:
        raise ValueError(f"Baseline validation scores ขาดคอลัมน์: {missing_scores}")

    print("[PASS] Stage 009 baseline bundle")
    print("[PASS] Baseline configurations and metrics")
    print("[PASS] Baseline validation scores")
    return bundle, metrics, scores


# ============================================================
# 5. LABEL HANDLING AND WINDOW IDENTITY
# ============================================================

def convert_labels(series: pd.Series, description: str) -> np.ndarray:
    normalized = series.astype(str).str.strip().str.lower()
    unknown = sorted(set(normalized.unique()) - NORMAL_LABELS - ATTACK_LABELS)
    if unknown:
        raise ValueError(f"{description} มี Label ที่ไม่รู้จัก: {unknown}")
    return normalized.map(lambda value: 0 if value in NORMAL_LABELS else 1).to_numpy(dtype=np.int8)


def window_keys(df: pd.DataFrame) -> set[tuple[str, int]]:
    window_ids = pd.to_numeric(df["window_id"], errors="coerce")
    if (
        window_ids.isna().any()
        or not np.isfinite(window_ids).all()
        or (window_ids % 1 != 0).any()
    ):
        raise ValueError("พบ window_id ที่ไม่ใช่จำนวนเต็ม")
    return set(zip(df["source_file"].astype(str).str.strip(), window_ids.astype("int64")))


# ============================================================
# 6. LOAD TRAIN AND VALIDATION
# ============================================================

def load_split(path: Path, split_name: str, features: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"ไม่พบไฟล์ {split_name}:\n{path}")

    df = clean_columns(
        pd.read_csv(path, low_memory=False, encoding="utf-8-sig"),
        split_name,
    )
    if df.empty:
        raise ValueError(f"{split_name} ไม่มีข้อมูล")

    required = ["source_file", "window_id", "binary_label", "split", *features]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{split_name} ขาดคอลัมน์: {missing}")

    actual_splits = df["split"].astype(str).str.strip().str.upper()
    if not actual_splits.eq(split_name).all():
        raise ValueError(f"พบแถวที่ไม่ใช่ {split_name} ในไฟล์")
    if df[["source_file", "window_id", "binary_label"]].isna().any().any():
        raise ValueError(f"{split_name} มี Metadata ว่าง")

    df["source_file"] = df["source_file"].astype(str).str.strip()
    window_id = pd.to_numeric(df["window_id"], errors="coerce")
    if (
        window_id.isna().any()
        or not np.isfinite(window_id).all()
        or (window_id < 1).any()
        or (window_id % 1 != 0).any()
    ):
        raise ValueError(f"{split_name}: window_id ต้องเป็นจำนวนเต็มบวก")
    df["window_id"] = window_id.astype("int64")

    if df.duplicated(["source_file", "window_id"]).any():
        raise ValueError(f"{split_name} มี Window ซ้ำ")
    convert_labels(df["binary_label"], split_name)

    for feature in features:
        values = pd.to_numeric(df[feature], errors="coerce")
        invalid = values.isna() | ~np.isfinite(values)
        if invalid.any():
            raise ValueError(
                f"{split_name}: {feature} มี NaN หรือ Inf "
                f"{int(invalid.sum()):,} แถว"
            )
        df[feature] = values.astype("float64")

    print(f"[PASS] {split_name}: {len(df):,} rows, {len(features)} features")
    return df


# ============================================================
# 7. VALIDATE DATA AND BASELINE ALIGNMENT
# ============================================================

def validate_data(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    baseline_bundle: dict,
    baseline_scores: pd.DataFrame,
) -> np.ndarray:
    y_train = convert_labels(train["binary_label"], "TRAIN")
    y_validation = convert_labels(validation["binary_label"], "VALIDATION")

    if not np.all(y_train == 0):
        raise ValueError("TRAIN ต้องเป็น Normal-only")
    if not np.any(y_validation == 0) or not np.any(y_validation == 1):
        raise ValueError("VALIDATION ต้องมีทั้ง Normal และ Attack")

    train_sources = set(train["source_file"].astype(str).str.strip())
    validation_sources = set(validation["source_file"].astype(str).str.strip())
    if train_sources.intersection(validation_sources):
        raise ValueError("พบ Source File ซ้ำระหว่าง TRAIN และ VALIDATION")
    if window_keys(train).intersection(window_keys(validation)):
        raise ValueError("พบ Window ซ้ำระหว่าง TRAIN และ VALIDATION")

    if len(baseline_scores) != len(validation) or window_keys(baseline_scores) != window_keys(validation):
        raise ValueError(
            "Validation ใน Stage 009 ไม่ตรงกับข้อมูลปัจจุบัน "
            "กรุณารัน Stage 009 ใหม่"
        )

    baseline_y = convert_labels(baseline_scores["binary_label"], "Stage 009 VALIDATION")
    current_labels = {
        (str(source).strip(), int(window_id)): int(label)
        for source, window_id, label in zip(
            validation["source_file"], validation["window_id"], y_validation
        )
    }
    baseline_window_ids = pd.to_numeric(baseline_scores["window_id"], errors="coerce")
    baseline_labels = {
        (str(source).strip(), int(window_id)): int(label)
        for source, window_id, label in zip(
            baseline_scores["source_file"], baseline_window_ids, baseline_y
        )
    }
    if current_labels != baseline_labels:
        raise ValueError("Labels ใน Stage 009 ไม่ตรงกับ VALIDATION ปัจจุบัน")

    train_rows = baseline_bundle.get("train_rows")
    if train_rows is not None:
        try:
            train_rows = int(train_rows)
        except (TypeError, ValueError) as exc:
            raise ValueError("train_rows ใน Stage 009 bundle ไม่ถูกต้อง") from exc
        if train_rows != len(train):
            raise ValueError(
                "จำนวน TRAIN ใน Stage 009 ไม่ตรงกับข้อมูลปัจจุบัน "
                "กรุณารัน Stage 009 ใหม่"
            )
        print("[PASS] Stage 009 training row count matches")
    else:
        print("[INFO] Stage 009 ไม่มี train_rows metadata; ใช้ TRAIN ปัจจุบันในการ Fit candidates")

    print("[PASS] TRAIN is Normal-only")
    print("[PASS] VALIDATION contains both classes")
    print("[PASS] TRAIN and VALIDATION source separation")
    print("[PASS] No duplicated windows between splits")
    print("[PASS] Stage 009 validation matches current data")
    print("\nVALIDATION LABEL DISTRIBUTION")
    print("-" * 76)
    print(f"Normal : {int(np.sum(y_validation == 0)):,}")
    print(f"Attack : {int(np.sum(y_validation == 1)):,}")
    return y_validation


# ============================================================
# 8. PREPARE FEATURE MATRICES
# ============================================================

def prepare_matrix(df: pd.DataFrame, features: list[str], description: str) -> np.ndarray:
    X = df[features].to_numpy(dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError(f"{description} ไม่มีข้อมูลสำหรับประมวลผล")
    if not np.isfinite(X).all():
        raise ValueError(f"{description} มี NaN หรือ Inf")
    return X


# ============================================================
# 9. FIXED MATCHED CONFIGURATIONS
# ============================================================

def build_candidate_configs() -> dict[str, list[dict]]:
    if_configs = [
        {
            "n_estimators": n_estimators,
            "contamination": contamination,
            "max_samples": max_samples,
            "max_features": max_features,
            "random_state": 42,
            "n_jobs": -1,
        }
        for n_estimators, contamination, max_samples, max_features in product(
            IF_N_ESTIMATORS, IF_CONTAMINATION, IF_MAX_SAMPLES, IF_MAX_FEATURES
        )
    ]
    ocsvm_configs = [
        {"kernel": "rbf", "nu": nu, "gamma": gamma}
        for nu, gamma in product(OCSVM_NU, OCSVM_GAMMA)
    ]
    lof_configs = [
        {
            "n_neighbors": neighbors,
            "contamination": contamination,
            "novelty": True,
            "metric": "minkowski",
            "p": 2,
            "n_jobs": -1,
        }
        for neighbors, contamination in product(LOF_NEIGHBORS, LOF_CONTAMINATION)
    ]

    candidates = {
        "isolation_forest": if_configs,
        "one_class_svm": ocsvm_configs,
        "lof": lof_configs,
    }
    for key, configs in candidates.items():
        expected = EXPECTED_CONFIG_COUNTS[key]
        if len(configs) != expected:
            raise RuntimeError(
                f"จำนวน candidate ของ {key} ไม่ถูกต้อง: ได้ {len(configs)} "
                f"แต่คาดหวัง {expected}"
            )
        signatures = {config_signature(config, key) for config in configs}
        if len(signatures) != len(configs):
            raise RuntimeError(f"พบ Configuration ซ้ำใน {MODEL_NAMES[key]}")

    total = sum(len(configs) for configs in candidates.values())
    if total != 3:
        raise RuntimeError(f"จำนวน Configuration รวมไม่ถูกต้อง: {total}")

    print("[PASS] IF configurations: 1 (fixed)")
    print("[PASS] OCSVM configurations: 1 (fixed)")
    print("[PASS] LOF configurations: 1 (fixed)")
    print("[PASS] Total configurations: 3")
    return candidates


# ============================================================
# 10. MODEL FACTORY AND ANOMALY SCORES
# ============================================================

def create_model(model_key: str, config: dict):
    if model_key == "isolation_forest":
        return IsolationForest(**config)
    if model_key == "one_class_svm":
        return OneClassSVM(**config)
    if model_key == "lof":
        return LocalOutlierFactor(**config)
    raise ValueError(f"ไม่รู้จัก Model: {model_key}")


def calculate_anomaly_scores(model_key: str, model, X: np.ndarray) -> np.ndarray:
    if model_key == "isolation_forest":
        scores = -model.score_samples(X)
    elif model_key == "one_class_svm":
        scores = -model.decision_function(X)
    elif model_key == "lof":
        scores = -model.score_samples(X)
    else:
        raise ValueError(f"ไม่รู้จัก Model: {model_key}")

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(scores) != len(X) or not np.isfinite(scores).all():
        raise ValueError(f"{model_key}: anomaly scores ไม่ถูกต้อง")
    return scores


# ============================================================
# 11. THRESHOLD METRICS
# ============================================================

def evaluate_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_pred = (scores > threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    normal_count = int(np.sum(y_true == 0))
    normal_far = float(fp / normal_count) if normal_count else np.nan
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "normal_far": normal_far,
        "far": normal_far,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_attack": int(fp + tp),
    }


# ============================================================
# 12. EVALUATE ONE FIXED CONFIGURATION AND CALIBRATE THRESHOLD
# ============================================================

def tune_one_configuration(
    model_key: str,
    config: dict,
    config_number: int,
    baseline_config: dict | None,
    baseline_reference: dict,
    X_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
) -> list[dict]:
    model_name = MODEL_NAMES[model_key]
    baseline_match = (
        isinstance(baseline_config, dict)
        and config_signature(config, model_key) == config_signature(baseline_config, model_key)
    )
    config_json = json.dumps(config, ensure_ascii=False, sort_keys=True, default=str)

    base_row = {
        "stage": "010",
        "dataset": "CIC-IDS2017",
        "window_seconds": WINDOW_SECONDS,
        "model": model_name,
        "model_key": model_key,
        "config_number": config_number,
        "search_space_configurations": EXPECTED_CONFIG_COUNTS[model_key],
        "search_space_definition": SEARCH_SPACE_DETAILS[model_key]["search_space_definition"],
        "contamination_handling": SEARCH_SPACE_DETAILS[model_key]["contamination_handling"],
        "grid_comparison_note": SEARCH_SPACE_DETAILS[model_key]["grid_comparison_note"],
        "configuration": config_json,
        "is_baseline_configuration": baseline_match,
        "normal_far_limit": MAX_NORMAL_FAR,
        "prediction_rule": "score > threshold",
        "threshold_source": "VALIDATION Normal scores",
        "preprocessing": "Reuse Stage 007 scaled features",
        "feature_selection": "NOT PERFORMED",
        "fit_data": "TRAIN only",
        "evaluation_data": "VALIDATION only",
        "final_test_used": False,
        "baseline_native_f1": baseline_reference["f1"],
        "baseline_native_normal_far": baseline_reference["normal_far"],
    }

    try:
        model = create_model(model_key, config)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X_train)
        scores = calculate_anomaly_scores(model_key, model, X_validation)
        normal_scores = scores[y_validation == 0]
        if normal_scores.size == 0:
            raise ValueError("ไม่มี Normal scores ใน VALIDATION")

        roc_auc = float(roc_auc_score(y_validation, scores))
        pr_auc = float(average_precision_score(y_validation, scores))
        warning_messages = [str(item.message) for item in caught]
        rows = []

        for percentile in THRESHOLD_PERCENTILES:
            threshold = float(np.percentile(normal_scores, percentile))
            metrics = evaluate_threshold(y_validation, scores, threshold)
            eligible = bool(metrics["normal_far"] <= MAX_NORMAL_FAR + EPSILON)
            rows.append({
                **base_row,
                "threshold_percentile": int(percentile),
                "threshold": threshold,
                **metrics,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "eligible": eligible,
                "selected": False,
                "config_status": "SUCCESS",
                "warning_count": len(warning_messages),
                "warnings": " | ".join(warning_messages),
                "error": "",
            })
        return rows

    except Exception as exc:
        return [{
            **base_row,
            "threshold_percentile": np.nan,
            "threshold": np.nan,
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "normal_far": np.nan,
            "far": np.nan,
            "tn": np.nan,
            "fp": np.nan,
            "fn": np.nan,
            "tp": np.nan,
            "predicted_attack": np.nan,
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "eligible": False,
            "selected": False,
            "config_status": "FAILED",
            "warning_count": 0,
            "warnings": "",
            "error": repr(exc),
        }]


# ============================================================
# 13. SELECT THRESHOLD PER MODEL
# ============================================================

def select_best_configurations(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = results.copy()
    selected_rows = []

    for model_key, model_name in MODEL_NAMES.items():
        eligible = results.loc[
            (results["model_key"] == model_key)
            & (results["config_status"] == "SUCCESS")
            & (results["eligible"] == True)
        ].copy()

        if eligible.empty:
            selected_rows.append({
                "stage": "010",
                "dataset": "CIC-IDS2017",
                "model": model_name,
                "model_key": model_key,
                "search_space_configurations": EXPECTED_CONFIG_COUNTS[model_key],
                "search_space_definition": SEARCH_SPACE_DETAILS[model_key]["search_space_definition"],
                "contamination_handling": SEARCH_SPACE_DETAILS[model_key]["contamination_handling"],
                "grid_comparison_note": SEARCH_SPACE_DETAILS[model_key]["grid_comparison_note"],
                "selection_status": "NO_FEASIBLE_CONFIGURATION",
                "configuration": "",
                "config_number": np.nan,
                "is_baseline_configuration": False,
                "threshold_percentile": np.nan,
                "threshold": np.nan,
                "accuracy": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "normal_far": np.nan,
                "roc_auc": np.nan,
                "pr_auc": np.nan,
                "tn": np.nan,
                "fp": np.nan,
                "fn": np.nan,
                "tp": np.nan,
                "normal_far_limit": MAX_NORMAL_FAR,
                "selection_rule": "Normal-only FAR <= 15%, then highest Validation F1",
                "prediction_rule": "score > threshold",
                "preprocessing": "Reuse Stage 007 scaled features",
                "final_test_used": False,
            })
            continue

        # Tie-break: highest F1, highest precision, lowest FAR,
        # higher percentile, then earlier (only) configuration.
        best = eligible.sort_values(
            by=["f1", "precision", "normal_far", "threshold_percentile", "config_number"],
            ascending=[False, False, True, False, True],
            kind="mergesort",
        ).iloc[0]

        selected_rows.append({
            "stage": "010",
            "dataset": "CIC-IDS2017",
            "model": model_name,
            "model_key": model_key,
            "search_space_configurations": EXPECTED_CONFIG_COUNTS[model_key],
            "search_space_definition": SEARCH_SPACE_DETAILS[model_key]["search_space_definition"],
            "contamination_handling": SEARCH_SPACE_DETAILS[model_key]["contamination_handling"],
            "grid_comparison_note": SEARCH_SPACE_DETAILS[model_key]["grid_comparison_note"],
            "selection_status": "SELECTED",
            "configuration": best["configuration"],
            "config_number": int(best["config_number"]),
            "is_baseline_configuration": bool(best["is_baseline_configuration"]),
            "threshold_percentile": int(best["threshold_percentile"]),
            "threshold": float(best["threshold"]),
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
            "normal_far_limit": MAX_NORMAL_FAR,
            "selection_rule": "Normal-only FAR <= 15%, then highest Validation F1",
            "prediction_rule": "score > threshold",
            "preprocessing": "Reuse Stage 007 scaled features",
            "final_test_used": False,
        })

        mask = (
            (results["model_key"] == model_key)
            & (results["config_number"] == best["config_number"])
            & (results["threshold_percentile"] == best["threshold_percentile"])
            & (results["config_status"] == "SUCCESS")
        )
        results.loc[mask, "selected"] = True

    return results, pd.DataFrame(selected_rows)


# ============================================================
# 14. BUILD TUNING SUMMARY
# ============================================================

def build_tuning_summary(
    results: pd.DataFrame,
    selected: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    candidates: dict[str, list[dict]],
) -> pd.DataFrame:
    rows = []
    for model_key, model_name in MODEL_NAMES.items():
        model_results = results.loc[results["model_key"] == model_key]
        successful = model_results.loc[model_results["config_status"] == "SUCCESS"]
        eligible = successful.loc[successful["eligible"] == True]
        chosen_rows = selected.loc[selected["model_key"] == model_key]
        if len(chosen_rows) != 1:
            raise RuntimeError(f"ไม่พบผลการเลือกเพียงหนึ่งรายการสำหรับ {model_name}")
        chosen = chosen_rows.iloc[0]

        baseline_rows = baseline_metrics.loc[baseline_metrics["model"] == model_name]
        if len(baseline_rows) != 1:
            raise RuntimeError(f"ไม่พบ Baseline metrics ของ {model_name}")
        baseline = baseline_rows.iloc[0]

        successful_count = int(successful["config_number"].nunique())
        rows.append({
            "stage": "010",
            "dataset": "CIC-IDS2017",
            "model": model_name,
            "model_key": model_key,
            "search_space_configurations": EXPECTED_CONFIG_COUNTS[model_key],
            "search_space_definition": SEARCH_SPACE_DETAILS[model_key]["search_space_definition"],
            "contamination_handling": SEARCH_SPACE_DETAILS[model_key]["contamination_handling"],
            "grid_comparison_note": SEARCH_SPACE_DETAILS[model_key]["grid_comparison_note"],
            "baseline_f1_native": float(baseline["f1"]),
            "baseline_normal_far_native": float(baseline["normal_far"]),
            "configurations_attempted": len(candidates[model_key]),
            "configurations_successful": successful_count,
            "configurations_failed": len(candidates[model_key]) - successful_count,
            "threshold_candidates_per_config": len(THRESHOLD_PERCENTILES),
            "eligible_trials": len(eligible),
            "selection_status": chosen["selection_status"],
            "selected_configuration": chosen["configuration"],
            "selected_threshold_percentile": chosen["threshold_percentile"],
            "selected_threshold": chosen["threshold"],
            "validation_f1": chosen["f1"],
            "validation_precision": chosen["precision"],
            "validation_recall": chosen["recall"],
            "validation_normal_far": chosen["normal_far"],
            "validation_roc_auc": chosen["roc_auc"],
            "validation_pr_auc": chosen["pr_auc"],
            "normal_far_limit": MAX_NORMAL_FAR,
            "preprocessing": "Reuse Stage 007 scaled features",
            "fit_data": "TRAIN only",
            "selection_data": "VALIDATION only",
            "final_test_used": False,
        })
    return pd.DataFrame(rows)


# ============================================================
# 15. SAVE RESULTS
# ============================================================

def save_results(results: pd.DataFrame, selected: pd.DataFrame, summary: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for df, path in [
        (results, RESULTS_FILE),
        (selected, SELECTED_CONFIG_FILE),
        (summary, SUMMARY_FILE),
    ]:
        atomic_csv(df, path)
        if not path.is_file():
            raise RuntimeError(f"ไม่พบไฟล์หลังบันทึก: {path}")
        print(f"[SAVED] {path}")


# ============================================================
# 16. MAIN
# ============================================================

def main() -> None:
    print("=" * 78)
    print("STAGE 010 — CIC-IDS2017 MODEL CONFIGURATION MATCHING")
    print("=" * 78)
    print(f"TRAIN      : {TRAIN_FILE}")
    print(f"VALIDATION : {VALIDATION_FILE}")
    print(f"FEATURES   : {len(FEATURE_COLUMNS)} Behavioral Features")
    print(f"BASELINE   : {BASELINE_BUNDLE_FILE}")
    print(f"RESULTS    : {RESULTS_FILE}")
    print("FINAL_TEST : NOT USED")
    print("Scaling    : Reuse Stage 007 scaled data")
    print(f"Thresholds : {THRESHOLD_PERCENTILES}")
    print(f"Normal FAR : <= {MAX_NORMAL_FAR:.0%}")
    print("Model hyperparameters: FIXED to self-created dataset")
    print("Features          : ALL 13 Behavioral Features")
    print("Configurations: IF=1, OCSVM=1, LOF=1 (total=3)")
    print("Thresholds: calibrated using VALIDATION Normal scores")
    print("[INFO] No model hyperparameter grid search is performed.")
    print("=" * 78)

    print("\n[1] Load 13 Behavioral Features")
    features = load_features()

    print("\n[2] Load STAGE 009 baseline")
    baseline_bundle, baseline_metrics, baseline_scores = load_baseline(features)
    baseline_configs = baseline_bundle["model_configs"]

    print("\nSTAGE 009 BASELINE RESULTS (REFERENCE)")
    print("-" * 78)
    display_columns = [
        col for col in [
            "model", "accuracy", "precision", "recall", "f1",
            "normal_far", "roc_auc", "pr_auc",
        ] if col in baseline_metrics.columns
    ]
    print(baseline_metrics[display_columns].to_string(
        index=False, float_format=lambda value: f"{value:.6f}"
    ))
    print(
        "[INFO] Baseline ใช้ Native Decision Boundary "
        "ส่วน Stage 010 ใช้ Validation-calibrated Threshold"
    )

    print("\n[3] Load TRAIN and VALIDATION")
    train = load_split(TRAIN_FILE, "TRAIN", features)
    validation = load_split(VALIDATION_FILE, "VALIDATION", features)

    print("\n[4] Validate data")
    y_validation = validate_data(train, validation, baseline_bundle, baseline_scores)

    print("\n[5] Prepare feature matrices")
    X_train = prepare_matrix(train, features, "TRAIN")
    X_validation = prepare_matrix(validation, features, "VALIDATION")
    print(f"TRAIN matrix      : {X_train.shape}")
    print(f"VALIDATION matrix : {X_validation.shape}")
    print("[PASS] Reusing Stage 007 scaled data; no second scaler")

    print("\n[6] Build fixed matched configurations")
    candidates = build_candidate_configs()
    for key, configs in candidates.items():
        print(f"{MODEL_NAMES[key]:<20}: {len(configs)} configuration(s)")
        print("  Fixed configuration: " + SEARCH_SPACE_DETAILS[key]["search_space_definition"])

    total_configs = sum(len(configs) for configs in candidates.values())
    total_trials = total_configs * len(THRESHOLD_PERCENTILES)
    print(f"Total configurations : {total_configs}")
    print(f"Maximum threshold trials: {total_trials}")

    print("\n[7] Evaluate fixed configurations and calibrate thresholds")
    all_rows = []
    for model_key, configs in candidates.items():
        model_name = MODEL_NAMES[model_key]
        baseline_config = baseline_configs[model_name]
        baseline_row = baseline_metrics.loc[baseline_metrics["model"] == model_name].iloc[0]
        baseline_reference = {
            "f1": float(baseline_row["f1"]),
            "normal_far": float(baseline_row["normal_far"]),
        }

        print("\n" + "=" * 78)
        print(f"MATCHED CONFIGURATION: {model_name} ({len(configs)} configuration)")
        print("=" * 78)
        for number, config in enumerate(configs, start=1):
            baseline_match = (
                config_signature(config, model_key)
                == config_signature(baseline_config, model_key)
            )
            marker = " [BASELINE PARAMETERS]" if baseline_match else ""
            print(f"[{number}/{len(configs)}] {model_name}: {config}{marker}")

            rows = tune_one_configuration(
                model_key=model_key,
                config=config,
                config_number=number,
                baseline_config=baseline_config,
                baseline_reference=baseline_reference,
                X_train=X_train,
                X_validation=X_validation,
                y_validation=y_validation,
            )
            all_rows.extend(rows)
            successful_rows = [row for row in rows if row["config_status"] == "SUCCESS"]
            if successful_rows:
                eligible_count = sum(bool(row["eligible"]) for row in successful_rows)
                print(f"  Thresholds: {len(successful_rows)}, eligible: {eligible_count}")
                warning_count = successful_rows[0]["warning_count"]
                if warning_count:
                    print(f"  [WARNING] Model generated {warning_count} warning(s)")
            else:
                print(f"  [FAILED] {rows[0]['error']}")

    if not all_rows:
        raise RuntimeError("ไม่มีผลลัพธ์จากการทดลอง")

    print("\n[8] Select threshold for each fixed configuration")
    results_df, selected_df = select_best_configurations(pd.DataFrame(all_rows))
    summary_df = build_tuning_summary(results_df, selected_df, baseline_metrics, candidates)

    print("\n[9] Save results")
    save_results(results_df, selected_df, summary_df)

    print("\n" + "=" * 78)
    print("STAGE 010 — SELECTED CONFIGURATIONS AND THRESHOLDS")
    print("=" * 78)
    for _, row in selected_df.iterrows():
        print(f"\nMODEL: {row['model']}")
        print(f"Status: {row['selection_status']}")
        if row["selection_status"] != "SELECTED":
            print("No eligible threshold found under the FAR constraint.")
            continue
        print(f"Configuration : {row['configuration']}")
        print(f"Baseline parameters selected: {bool(row['is_baseline_configuration'])}")
        print(f"Threshold      : P{row['threshold_percentile']}")
        print(f"Threshold value: {row['threshold']:.12f}")
        print(f"Precision      : {row['precision']:.6f}")
        print(f"Recall         : {row['recall']:.6f}")
        print(f"F1             : {row['f1']:.6f}")
        print(f"Normal FAR     : {row['normal_far']:.6f}")
        print(f"ROC-AUC        : {row['roc_auc']:.6f}")
        print(f"PR-AUC         : {row['pr_auc']:.6f}")
        print(f"Confusion Matrix: TN={row['tn']}, FP={row['fp']}, FN={row['fn']}, TP={row['tp']}")

    print("\n" + "=" * 78)
    print("STAGE 010 SUMMARY")
    print("=" * 78)
    print(f"TRAIN rows             : {len(train):,}")
    print(f"VALIDATION rows        : {len(validation):,}")
    print(f"Features               : {len(features)}")
    print(f"Models                 : {len(MODEL_NAMES)}")
    print(f"Fixed configurations   : {total_configs}")
    print(f"Maximum threshold trials: {total_trials}")
    print("Scaling                : Reuse Stage 007 scaled data")
    print("Fit data               : TRAIN only")
    print("Threshold selection    : VALIDATION only")
    print("Final Test             : NOT USED")
    print("Parameter match        : IF/OCSVM/LOF match self-created dataset")

    print("\nTUNING SUMMARY")
    print("-" * 78)
    summary_columns = [
        "model", "search_space_configurations", "configurations_attempted",
        "configurations_successful", "configurations_failed", "eligible_trials",
        "selection_status", "selected_threshold_percentile", "validation_f1",
        "validation_normal_far",
    ]
    print(summary_df[summary_columns].to_string(
        index=False, float_format=lambda value: f"{value:.6f}"
    ))
    print("\nSAVED FILES")
    print("-" * 78)
    print(f"All tuning results : {RESULTS_FILE}")
    print(f"Selected configs   : {SELECTED_CONFIG_FILE}")
    print(f"Tuning summary     : {SUMMARY_FILE}")
    print("\n" + "=" * 78)
    print("STAGE 010 COMPLETED")
    print("=" * 78)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n" + "=" * 78)
        print("STAGE 010 FAILED")
        print("=" * 78)
        print(f"Error: {exc}")
        sys.exit(1)