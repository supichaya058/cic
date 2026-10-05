
from __future__ import annotations

from pathlib import Path
import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# ============================================================
# STAGE 009 — CIC-IDS2017 BASELINE MODEL TRAINING
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
#   data/models/cic_baseline_isolation_forest_10s.joblib
#   data/models/cic_baseline_ocsvm_10s.joblib
#   data/models/cic_baseline_lof_10s.joblib
#   data/models/cic_baseline_models_10s.joblib
#   data/models/cic_baseline_validation_scores_10s.csv
#   data/models/cic_baseline_metrics_10s.csv
#   data/models/cic_baseline_summary_10s.csv
#
# FEATURE SELECTION:
#   Not performed in this stage. All 13 Behavioral Features are used.
#
# MODELS:
#   1. Isolation Forest
#   2. One-Class SVM
#   3. Local Outlier Factor (LOF)
#
# TRAIN:
#   Normal-only
#   Used for model fitting only.
#
# VALIDATION:
#   Used for baseline evaluation only.
#
# PREDICTION:
#   Native model decision boundary.
#
# THRESHOLD:
#   No threshold tuning at this stage.
#
# FINAL TEST:
#   Not loaded or used.
#
# SCORE CONVENTION:
#   Higher score = more anomalous
# ============================================================


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "time_windows"
MODEL_DIR = BASE_DIR / "data" / "models"

TRAIN_FILE = (
    DATA_DIR / "cic_train_scaled_10s.csv"
)

VALIDATION_FILE = (
    DATA_DIR / "cic_validation_scaled_10s.csv"
)

IF_MODEL_FILE = (
    MODEL_DIR / "cic_baseline_isolation_forest_10s.joblib"
)

OCSVM_MODEL_FILE = (
    MODEL_DIR / "cic_baseline_ocsvm_10s.joblib"
)

LOF_MODEL_FILE = (
    MODEL_DIR / "cic_baseline_lof_10s.joblib"
)

BUNDLE_FILE = (
    MODEL_DIR / "cic_baseline_models_10s.joblib"
)

SCORES_FILE = (
    MODEL_DIR / "cic_baseline_validation_scores_10s.csv"
)

METRICS_FILE = (
    MODEL_DIR / "cic_baseline_metrics_10s.csv"
)

SUMMARY_FILE = (
    MODEL_DIR / "cic_baseline_summary_10s.csv"
)

WINDOW_SECONDS = 10

# All 13 Behavioral Features from Stage 005/007.
# No Feature Selection is performed in this stage.
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

SPLIT_NAMES = [
    "TRAIN",
    "VALIDATION",
]

NORMAL_LABELS = {
    "normal",
    "benign",
    "legitimate",
    "clean",
    "background",
    "0",
    "0.0",
    "false",
    "no",
}

ATTACK_LABELS = {
    "attack",
    "malicious",
    "1",
    "1.0",
    "true",
    "yes",
}

EXPECTED_ATTACK_TYPES = {
    "dosslowloris",
    "dosslowhttptest",
    "doshulk",
    "dosgoldeneye",
    "ddosloic",
}

# Baseline configurations.
# Initial settings only; no tuning at this stage.
MODEL_CONFIGS = {
    "Isolation Forest": {
        "n_estimators": 300,
        "max_samples": "auto",
        "max_features": 1.0,
        "contamination": 0.01,
        "random_state": 42,
        "n_jobs": -1,
    },
    "One-Class SVM": {
        "kernel": "rbf",
        "nu": 0.01,
        "gamma": "scale",
    },
    "LOF": {
        "n_neighbors": 20,
        "contamination": 0.01,
        "novelty": True,
        "metric": "minkowski",
        "p": 2,
        "n_jobs": -1,
    },
}

MODEL_FILE_MAP = {
    "Isolation Forest": IF_MODEL_FILE,
    "One-Class SVM": OCSVM_MODEL_FILE,
    "LOF": LOF_MODEL_FILE,
}

SHORT_NAMES = {
    "Isolation Forest": "if",
    "One-Class SVM": "ocsvm",
    "LOF": "lof",
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

    duplicated = result.columns[
        result.columns.duplicated()
    ].tolist()

    if duplicated:
        raise ValueError(
            f"{description} มีชื่อคอลัมน์ซ้ำ: {duplicated}"
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


def atomic_joblib(
    obj,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_name(
        path.stem + ".tmp.joblib"
    )

    try:
        joblib.dump(
            obj,
            temp_path,
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
# 4. LABEL CONVERSION
# ============================================================

def convert_labels(
    series: pd.Series,
    description: str,
) -> np.ndarray:

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    unknown = sorted(
        set(normalized.unique())
        - NORMAL_LABELS
        - ATTACK_LABELS
    )

    if unknown:
        raise ValueError(
            f"{description} มี Binary Label ที่ไม่รู้จัก: "
            f"{unknown}"
        )

    labels = normalized.map(
        lambda value: (
            0 if value in NORMAL_LABELS else 1
        )
    )

    return labels.to_numpy(dtype=np.int8)


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


# ============================================================
# 5. LOAD DATA
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

    required = [
        "source_file",
        "window_id",
        "binary_label",
        "split",
        *features,
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{split_name} ขาดคอลัมน์: {missing}"
        )

    # Verify split identity.
    actual_splits = (
        df["split"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if not actual_splits.eq(split_name).all():
        raise ValueError(
            f"พบแถวที่ไม่ใช่ {split_name} ในไฟล์"
        )

    # Verify metadata.
    metadata = [
        "source_file",
        "window_id",
        "binary_label",
    ]

    if df[metadata].isna().any().any():
        raise ValueError(
            f"{split_name} มี Metadata ว่าง"
        )

    df["source_file"] = (
        df["source_file"]
        .astype(str)
        .str.strip()
    )

    window_id = pd.to_numeric(
        df["window_id"],
        errors="coerce",
    )

    if (
        window_id.isna().any()
        or not np.isfinite(window_id).all()
        or (window_id < 1).any()
        or (window_id % 1 != 0).any()
    ):
        raise ValueError(
            f"{split_name}: window_id ต้องเป็นจำนวนเต็มบวก"
        )

    df["window_id"] = window_id.astype("int64")

    if df.duplicated(
        ["source_file", "window_id"]
    ).any():
        raise ValueError(
            f"{split_name} มี Window ซ้ำ"
        )

    # Validate labels.
    convert_labels(
        df["binary_label"],
        split_name,
    )

    # Validate features.
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


# ============================================================
# 6. VALIDATE TRAIN AND VALIDATION
# ============================================================

def validate_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:

    print("\n[3] Validate data splits")
    print("-" * 76)

    y_train = convert_labels(
        train["binary_label"],
        "TRAIN",
    )

    y_validation = convert_labels(
        validation["binary_label"],
        "VALIDATION",
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

    print("[PASS] TRAIN is Normal-only")
    print("[PASS] VALIDATION contains Normal and Attack")

    # TRAIN and VALIDATION should not share a source.
    train_sources = set(
        train["source_file"].astype(str).str.strip()
    )

    validation_sources = set(
        validation["source_file"].astype(str).str.strip()
    )

    overlap = train_sources.intersection(
        validation_sources
    )

    if overlap:
        raise ValueError(
            "พบ Source File ซ้ำระหว่าง TRAIN และ VALIDATION:\n"
            f"{sorted(overlap)}"
        )

    print("[PASS] TRAIN and VALIDATION source separation")

    # Ensure there is no exact window overlap.
    train_windows = set(
        zip(
            train["source_file"].astype(str).str.strip(),
            train["window_id"].astype(int),
        )
    )

    validation_windows = set(
        zip(
            validation["source_file"].astype(str).str.strip(),
            validation["window_id"].astype(int),
        )
    )

    if train_windows.intersection(validation_windows):
        raise ValueError(
            "พบ Window ซ้ำระหว่าง TRAIN และ VALIDATION"
        )

    print("[PASS] No duplicated window between splits")

    # Validate the selected attack types on VALIDATION.
    if (
        "attack_type" not in validation.columns
        and "traffic_type" not in validation.columns
    ):
        raise ValueError(
            "ไม่พบ attack_type หรือ traffic_type "
            "สำหรับตรวจสอบประเภทการโจมตี"
        )

    validation_attacks = validation.loc[
        y_validation == 1
    ]

    observed_attacks = set()

    for _, row in validation_attacks.iterrows():
        attack_type = get_attack_type(row)

        if attack_type:
            observed_attacks.add(attack_type)

    missing_attacks = sorted(
        EXPECTED_ATTACK_TYPES - observed_attacks
    )

    if missing_attacks:
        raise ValueError(
            f"VALIDATION ขาด Attack Type: {missing_attacks}"
        )

    print("[PASS] VALIDATION contains all five attack types")

    print("\nVALIDATION LABEL DISTRIBUTION")
    print("-" * 76)
    print(f"Normal : {int(np.sum(y_validation == 0)):,}")
    print(f"Attack : {int(np.sum(y_validation == 1)):,}")

    return y_train, y_validation


# ============================================================
# 7. PREPARE FEATURE MATRICES
# ============================================================

def prepare_matrix(
    df: pd.DataFrame,
    features: list[str],
    description: str,
) -> np.ndarray:

    X = df[features].to_numpy(
        dtype=np.float64
    )

    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError(
            f"{description} ไม่มีข้อมูลสำหรับประมวลผล"
        )

    if not np.isfinite(X).all():
        raise ValueError(
            f"{description} มี NaN หรือ Inf"
        )

    return X


# ============================================================
# 8. BUILD BASELINE MODELS
# ============================================================

def build_models() -> dict[str, object]:

    return {
        "Isolation Forest": IsolationForest(
            **MODEL_CONFIGS["Isolation Forest"]
        ),

        "One-Class SVM": OneClassSVM(
            **MODEL_CONFIGS["One-Class SVM"]
        ),

        "LOF": LocalOutlierFactor(
            **MODEL_CONFIGS["LOF"]
        ),
    }


# ============================================================
# 9. FIT BASELINE MODELS
# ============================================================

def fit_models(
    models: dict[str, object],
    X_train: np.ndarray,
) -> dict[str, object]:

    fitted = {}

    for name, model in models.items():

        print(f"\n[FIT] {name}")
        print(
            "Configuration: "
            f"{MODEL_CONFIGS[name]}"
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X_train)

        fitted[name] = model

        if caught:
            print(
                f"[WARNING] {name}: {len(caught)} warning(s)"
            )

            for item in caught[:5]:
                print(f"  - {item.message}")

        print(f"[PASS] {name} fitted")

    return fitted


# ============================================================
# 10. ANOMALY SCORES
# ============================================================

def calculate_scores(
    name: str,
    model,
    X: np.ndarray,
) -> np.ndarray:

    # Higher score = more anomalous.
    if name == "Isolation Forest":
        scores = -model.score_samples(X)

    elif name == "One-Class SVM":
        scores = -model.decision_function(X)

    elif name == "LOF":
        scores = -model.score_samples(X)

    else:
        raise ValueError(
            f"ไม่รู้จัก Model: {name}"
        )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    ).reshape(-1)

    if not np.isfinite(scores).all():
        raise ValueError(
            f"{name} มี Anomaly Score ที่ไม่เป็น finite"
        )

    if len(scores) != len(X):
        raise RuntimeError(
            f"{name}: จำนวน Anomaly Score ไม่ตรงกับข้อมูล"
        )

    return scores


# ============================================================
# 11. NATIVE PREDICTION
# ============================================================

def calculate_native_predictions(
    model,
    X: np.ndarray,
) -> np.ndarray:

    # sklearn anomaly estimators:
    # -1 = outlier, 1 = inlier
    raw = np.asarray(
        model.predict(X)
    ).reshape(-1)

    if not np.isin(raw, [-1, 1]).all():
        raise ValueError(
            "Model คืนค่า Prediction ที่ไม่ใช่ -1 หรือ 1"
        )

    return (raw == -1).astype(np.int8)


# ============================================================
# 12. EVALUATION METRICS
# ============================================================

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> dict:

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    normal_count = int(
        np.sum(y_true == 0)
    )

    far = (
        float(fp / normal_count)
        if normal_count > 0
        else np.nan
    )

    # AUC requires both classes.
    if len(np.unique(y_true)) == 2:
        roc_auc = float(
            roc_auc_score(y_true, scores)
        )

        pr_auc = float(
            average_precision_score(y_true, scores)
        )
    else:
        roc_auc = np.nan
        pr_auc = np.nan

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "far": far,
        "normal_far": far,

        "roc_auc": roc_auc,
        "pr_auc": pr_auc,

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),

        "predicted_attack": int(fp + tp),
    }


# ============================================================
# 13. SCORE VALIDATION
# ============================================================

def score_validation(
    validation: pd.DataFrame,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    models: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:

    score_df = validation.copy()
    metrics_rows = []

    for name, model in models.items():

        short_name = SHORT_NAMES[name]

        print(f"\n[SCORE] {name}")

        scores = calculate_scores(
            name,
            model,
            X_validation,
        )

        predictions = calculate_native_predictions(
            model,
            X_validation,
        )

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

        metrics_rows.append({
            "stage": "009",
            "dataset": "CIC-IDS2017",
            "window_seconds": WINDOW_SECONDS,
            "model": name,
            "prediction_method": (
                "Native model decision boundary"
            ),
            "threshold_tuning": "NOT PERFORMED",
            "score_direction": (
                "higher_is_more_anomalous"
            ),
            "evaluation_split": "VALIDATION",
            "final_test_used": False,
            **metrics,
        })

        print(
            f"Accuracy  : {metrics['accuracy']:.6f}"
        )
        print(
            f"Precision : {metrics['precision']:.6f}"
        )
        print(
            f"Recall    : {metrics['recall']:.6f}"
        )
        print(
            f"F1        : {metrics['f1']:.6f}"
        )
        print(
            f"FAR       : {metrics['far']:.6f}"
        )
        print(
            f"ROC-AUC   : {metrics['roc_auc']:.6f}"
        )
        print(
            f"PR-AUC    : {metrics['pr_auc']:.6f}"
        )

        print("[PASS] Validation scored")

    return score_df, pd.DataFrame(metrics_rows)


# ============================================================
# 14. BUILD SUMMARY
# ============================================================

def build_summary(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:

    rows = []

    for name, config in MODEL_CONFIGS.items():

        rows.append({
            "stage": "009",
            "dataset": "CIC-IDS2017",
            "model": name,
            "status": "SUCCESS",
            "train_rows": len(train),
            "validation_rows": len(validation),
            "feature_count": len(features),
            "features": "|".join(features),
            "configuration": json.dumps(
                config,
                ensure_ascii=False,
                sort_keys=True,
            ),
            "fit_data": "TRAIN only",
            "evaluation_data": "VALIDATION only",
            "prediction_method": (
                "Native model decision boundary"
            ),
            "threshold_tuning": "NOT PERFORMED",
            "final_test_used": False,
        })

    return pd.DataFrame(rows)


# ============================================================
# 15. SAVE ARTIFACTS
# ============================================================

def save_artifacts(
    models: dict[str, object],
    score_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    features: list[str],
) -> None:

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n[7] Save baseline models and results")
    print("-" * 76)

    # Save individual models.
    for name, model in models.items():

        path = MODEL_FILE_MAP[name]

        atomic_joblib(
            model,
            path,
        )

        print(f"[SAVED] {path}")

    # Save combined model bundle.
    bundle = {
        "stage": "009",
        "dataset": "CIC-IDS2017",
        "window": "10 seconds",
        "features": features,
        "feature_selection": "NOT PERFORMED",
        "models": models,
        "model_configs": MODEL_CONFIGS,
        "score_direction": "higher_is_more_anomalous",
        "prediction_method": (
            "Native model decision boundary"
        ),
        "threshold_tuning": False,
        "fit_split": "TRAIN",
        "evaluation_split": "VALIDATION",
        "final_test_used": False,
    }

    atomic_joblib(
        bundle,
        BUNDLE_FILE,
    )

    print(f"[SAVED] {BUNDLE_FILE}")

    # Save validation scores.
    atomic_csv(
        score_df,
        SCORES_FILE,
    )

    print(f"[SAVED] {SCORES_FILE}")

    # Save metrics.
    atomic_csv(
        metrics_df,
        METRICS_FILE,
    )

    print(f"[SAVED] {METRICS_FILE}")

    # Save summary.
    atomic_csv(
        summary_df,
        SUMMARY_FILE,
    )

    print(f"[SAVED] {SUMMARY_FILE}")


# ============================================================
# 16. MAIN
# ============================================================

def main() -> None:

    print("=" * 76)
    print("STAGE 009 — CIC-IDS2017 BASELINE MODEL TRAINING")
    print("=" * 76)

    print(f"TRAIN      : {TRAIN_FILE}")
    print(f"VALIDATION : {VALIDATION_FILE}")
    print(f"FEATURES   : {len(FEATURE_COLUMNS)} Behavioral Features")
    print(f"OUTPUT     : {MODEL_DIR}")
    print("Final Test : NOT USED")
    print("Threshold  : Native model decision boundary")
    print("=" * 76)

    # 1. Load all 13 Behavioral Features from Stage 007.
    print("\n[1] Load 13 Behavioral Features")

    features = load_features()

    # 2. Load TRAIN and VALIDATION only.
    print("\n[2] Load TRAIN and VALIDATION")

    train = load_split(
        TRAIN_FILE,
        "TRAIN",
        features,
    )

    validation = load_split(
        VALIDATION_FILE,
        "VALIDATION",
        features,
    )

    # 3. Validate data.
    print("\n[3] Validate data")

    y_train, y_validation = validate_splits(
        train,
        validation,
    )

    # 4. Prepare feature matrices.
    print("\n[4] Prepare feature matrices")

    X_train = prepare_matrix(
        train,
        features,
        "TRAIN",
    )

    X_validation = prepare_matrix(
        validation,
        features,
        "VALIDATION",
    )

    print(f"TRAIN matrix      : {X_train.shape}")
    print(f"VALIDATION matrix : {X_validation.shape}")

    # 5. Fit baseline models on TRAIN only.
    print("\n[5] Fit baseline models")

    models = build_models()

    fitted_models = fit_models(
        models,
        X_train,
    )

    # 6. Score VALIDATION.
    print("\n[6] Score VALIDATION")

    score_df, metrics_df = score_validation(
        validation,
        X_validation,
        y_validation,
        fitted_models,
    )

    # 7. Build summary.
    summary_df = build_summary(
        train,
        validation,
        features,
    )

    # 8. Save models and results.
    save_artifacts(
        fitted_models,
        score_df,
        metrics_df,
        summary_df,
        features,
    )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print("\n" + "=" * 76)
    print("STAGE 009 SUMMARY")
    print("=" * 76)

    print(f"TRAIN rows       : {len(train):,}")
    print(f"VALIDATION rows  : {len(validation):,}")
    print(f"Features         : {len(features)}")
    print("Models           : 3")
    print("Fit data         : TRAIN only")
    print("Evaluation       : VALIDATION only")
    print("Threshold tuning : NO")
    print("Final Test       : NOT USED")

    print("\nBASELINE VALIDATION METRICS")
    print("-" * 76)

    print(
        metrics_df[
            [
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "far",
                "roc_auc",
                "pr_auc",
            ]
        ].to_string(index=False)
    )

    print("\nSAVED FILES")
    print("-" * 76)

    for path in [
        IF_MODEL_FILE,
        OCSVM_MODEL_FILE,
        LOF_MODEL_FILE,
        BUNDLE_FILE,
        SCORES_FILE,
        METRICS_FILE,
        SUMMARY_FILE,
    ]:
        print(path)

    print("=" * 76)
    print("STAGE 009 COMPLETED")
    print("=" * 76)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"\nSTAGE 009 FAILED: {exc}",
            file=sys.stderr,
        )
        raise