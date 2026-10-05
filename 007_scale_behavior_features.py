
from __future__ import annotations

from pathlib import Path
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ============================================================
# STAGE 007 — CIC-IDS2017 FEATURE SCALING
# ============================================================
#
# INPUT:
#   data/time_windows/cic_behavior_dataset_10s.csv
#
# OUTPUT:
#   data/time_windows/cic_behavior_dataset_scaled_10s.csv
#   data/time_windows/cic_train_scaled_10s.csv
#   data/time_windows/cic_validation_scaled_10s.csv
#   data/time_windows/cic_final_test_scaled_10s.csv
#   data/time_windows/cic_standard_scaler_10s.joblib
#   data/time_windows/cic_scaling_parameters_10s.csv
#   data/time_windows/cic_scaling_summary_10s.csv
#
# FEATURES:
#   All 13 Behavioral Features from Stage 006 are scaled.
#   Feature selection is performed later in Stage 008.
#
#   1. request_rate
#   2. total_requests
#   3. throughput
#   4. flow_count
#   5. error_rate
#   6. 4xx_count
#   7. 5xx_count
#   8. avg_latency
#   9. max_latency
#   10. latency_std_dev
#   11. unique_ip_count
#   12. new_ip_ratio
#   13. active_sessions
#
# SCALING:
#   - Fit StandardScaler using TRAIN only.
#   - Transform TRAIN, VALIDATION, FINAL_TEST
#     using the same fitted scaler.
#   - Preserve labels and metadata.
#   - Do not split, select, or remove windows.
#   - FINAL_TEST is never used for fitting.
# ============================================================


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "time_windows"

INPUT_FILE = (
    DATA_DIR / "cic_behavior_dataset_10s.csv"
)

OUTPUT_FILE = (
    DATA_DIR / "cic_behavior_dataset_scaled_10s.csv"
)

TRAIN_OUTPUT_FILE = (
    DATA_DIR / "cic_train_scaled_10s.csv"
)

VALIDATION_OUTPUT_FILE = (
    DATA_DIR / "cic_validation_scaled_10s.csv"
)

FINAL_TEST_OUTPUT_FILE = (
    DATA_DIR / "cic_final_test_scaled_10s.csv"
)

SCALER_FILE = (
    DATA_DIR / "cic_standard_scaler_10s.joblib"
)

PARAMETERS_FILE = (
    DATA_DIR / "cic_scaling_parameters_10s.csv"
)

SUMMARY_FILE = (
    DATA_DIR / "cic_scaling_summary_10s.csv"
)

WINDOW_SECONDS = 10

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

SPLIT_ORDER = [
    "TRAIN",
    "VALIDATION",
    "FINAL_TEST",
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

EPSILON = 1e-8


# ============================================================
# 2. UTILITIES
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


def normalize_label(value):
    if pd.isna(value):
        return None

    token = str(value).strip().lower()

    if token in NORMAL_LABELS:
        return 0

    if token in ATTACK_LABELS:
        return 1

    return None


def normalize_attack_type(value) -> str:
    if pd.isna(value):
        return ""

    return "".join(
        char.lower()
        for char in str(value)
        if char.isalnum()
    )


def get_attack_type(row: pd.Series) -> str | None:
    """
    Resolve attack type from attack_type or traffic_type.
    Return a canonical token for validation.
    """

    for column in ("attack_type", "traffic_type"):

        if column not in row.index:
            continue

        token = normalize_attack_type(row[column])

        if not token:
            continue

        for expected in EXPECTED_ATTACK_TYPES:
            if expected in token:
                return expected

    return None


def get_source_day(value) -> str | None:
    if pd.isna(value):
        return None

    name = (
        str(value)
        .replace("\\", "/")
        .split("/")[-1]
        .lower()
    )

    if "monday" in name:
        return "monday"

    if "wednesday" in name:
        return "wednesday"

    if "friday" in name:
        return "friday"

    return None


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
# 3. CHECK INPUT FILE
# ============================================================

def check_input_file() -> None:

    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            "ไม่พบไฟล์จาก Stage 006:\n"
            f"{INPUT_FILE}\n"
            "กรุณารัน 006_split_behavior_dataset.py ก่อน"
        )

    print("[PASS] Stage 006 input found")
    print(f"INPUT: {INPUT_FILE}")


# ============================================================
# 4. LOAD DATASET
# ============================================================

def load_dataset() -> pd.DataFrame:

    print("\n[1] Load Stage 006 dataset")

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
        encoding="utf-8-sig",
    )

    df = clean_columns(
        df,
        "Stage 006 dataset",
    )

    if df.empty:
        raise ValueError(
            "Stage 006 dataset ไม่มีข้อมูล"
        )

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns):,}")

    return df


# ============================================================
# 5. VALIDATE DATASET
# ============================================================

def validate_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:

    print("\n[2] Validate input dataset")
    print("-" * 76)

    df = df.copy()

    required_columns = [
        "source_file",
        "window_id",
        "binary_label",
        "split",
        *FEATURE_COLUMNS,
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Stage 006 ขาดคอลัมน์ที่จำเป็น: {missing}"
        )

    # ----------------------------------------
    # Metadata validation
    # ----------------------------------------

    if df[
        [
            "source_file",
            "window_id",
            "binary_label",
            "split",
        ]
    ].isna().any().any():
        raise ValueError(
            "พบค่าว่างใน Metadata"
        )

    df["source_file"] = (
        df["source_file"]
        .astype(str)
        .str.strip()
    )

    if df["source_file"].eq("").any():
        raise ValueError(
            "พบ source_file เป็นข้อความว่าง"
        )

    window_id = pd.to_numeric(
        df["window_id"],
        errors="coerce",
    )

    if (
        window_id.isna().any()
        or (~np.isfinite(window_id)).any()
        or (window_id < 1).any()
        or (window_id % 1 != 0).any()
    ):
        raise ValueError(
            "window_id ต้องเป็นจำนวนเต็มบวก"
        )

    df["window_id"] = window_id.astype("int64")

    # ----------------------------------------
    # Duplicate window validation
    # ----------------------------------------

    if df.duplicated(
        ["source_file", "window_id"]
    ).any():

        examples = df.loc[
            df.duplicated(
                ["source_file", "window_id"],
                keep=False,
            ),
            ["source_file", "window_id"],
        ].head(10)

        raise ValueError(
            "พบ source_file + window_id ซ้ำ:\n"
            + examples.to_string(index=False)
        )

    # ----------------------------------------
    # Validate split labels
    # ----------------------------------------

    df["split"] = (
        df["split"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    unknown_splits = sorted(
        set(df["split"].unique()) - set(SPLIT_ORDER)
    )

    if unknown_splits:
        raise ValueError(
            f"พบ Split ที่ไม่รู้จัก: {unknown_splits}"
        )

    for split_name in SPLIT_ORDER:
        if not df["split"].eq(split_name).any():
            raise ValueError(
                f"ไม่พบข้อมูล {split_name}"
            )

    # ----------------------------------------
    # Validate binary labels
    # ----------------------------------------

    df["_label_numeric"] = (
        df["binary_label"].map(normalize_label)
    )

    if df["_label_numeric"].isna().any():
        invalid_labels = (
            df.loc[
                df["_label_numeric"].isna(),
                "binary_label",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"พบ binary_label ที่ไม่รู้จัก: {invalid_labels}"
        )

    df["_label_numeric"] = (
        df["_label_numeric"].astype("int8")
    )

    # ----------------------------------------
    # Validate feature values
    # ----------------------------------------

    for feature in FEATURE_COLUMNS:

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
                f"{feature} มี NaN หรือ Inf "
                f"{int(invalid.sum()):,} ค่า"
            )

        if (values < 0).any():
            raise ValueError(
                f"พบ Feature ติดลบ: {feature}"
            )

        df[feature] = values.astype("float64")

    if (df["new_ip_ratio"] > 1).any():
        raise ValueError(
            "new_ip_ratio ต้องอยู่ในช่วง 0 ถึง 1"
        )

    # ----------------------------------------
    # Verify that all expected attack types
    # are represented in both evaluation sets.
    # ----------------------------------------

    if (
        "attack_type" not in df.columns
        and "traffic_type" not in df.columns
    ):
        raise ValueError(
            "ไม่พบ attack_type หรือ traffic_type "
            "สำหรับตรวจสอบประเภทการโจมตี"
        )

    print("Required columns : PASS")
    print("Window IDs       : PASS")
    print("Duplicate windows: PASS")
    print("Split values     : PASS")
    print("Binary labels    : PASS")
    print("Feature values   : PASS")

    return df, df["_label_numeric"].to_numpy()


# ============================================================
# 6. VALIDATE SPLIT INTEGRITY
# ============================================================

def validate_splits(
    df: pd.DataFrame,
    y: np.ndarray,
) -> None:

    print("\n[3] Validate splits")
    print("-" * 76)

    train_mask = df["split"].eq("TRAIN").to_numpy()
    validation_mask = df["split"].eq("VALIDATION").to_numpy()
    test_mask = df["split"].eq("FINAL_TEST").to_numpy()

    # TRAIN must contain only Normal.
    if not np.all(y[train_mask] == 0):
        raise ValueError(
            "TRAIN ต้องมีเฉพาะ Normal/BENIGN"
        )

    # Validation and Final Test must contain both classes.
    for name, mask in [
        ("VALIDATION", validation_mask),
        ("FINAL_TEST", test_mask),
    ]:
        labels = set(
            np.unique(y[mask]).tolist()
        )

        if labels != {0, 1}:
            raise ValueError(
                f"{name} ต้องมีทั้ง Normal และ Attack "
                f"แต่พบ labels: {labels}"
            )

    # TRAIN should remain separate from evaluation sources.
    train_sources = set(
        df.loc[train_mask, "source_file"].astype(str)
    )

    validation_sources = set(
        df.loc[validation_mask, "source_file"].astype(str)
    )

    test_sources = set(
        df.loc[test_mask, "source_file"].astype(str)
    )

    if train_sources.intersection(validation_sources):
        raise ValueError(
            "พบ TRAIN source ซ้ำใน VALIDATION"
        )

    if train_sources.intersection(test_sources):
        raise ValueError(
            "พบ TRAIN source ซ้ำใน FINAL_TEST"
        )

    # VALIDATION and FINAL_TEST may share source files
    # because Stage 006 uses chronological window split.
    validation_keys = set(
        zip(
            df.loc[validation_mask, "source_file"].astype(str),
            df.loc[validation_mask, "window_id"].astype(int),
        )
    )

    test_keys = set(
        zip(
            df.loc[test_mask, "source_file"].astype(str),
            df.loc[test_mask, "window_id"].astype(int),
        )
    )

    overlap = validation_keys.intersection(test_keys)

    if overlap:
        raise ValueError(
            "พบ Window เดียวกันใน VALIDATION และ FINAL_TEST: "
            f"{list(overlap)[:5]}"
        )

    # Verify selected attack types in validation/test.
    for split_name, mask in [
        ("VALIDATION", validation_mask),
        ("FINAL_TEST", test_mask),
    ]:
        attack_rows = df.loc[
            mask & (y == 1)
        ]

        observed = set()

        for _, row in attack_rows.iterrows():
            attack_type = get_attack_type(row)
            if attack_type:
                observed.add(attack_type)

        missing = sorted(
            EXPECTED_ATTACK_TYPES - observed
        )

        if missing:
            raise ValueError(
                f"{split_name} ขาด Attack Type: {missing}"
            )

    print("[PASS] TRAIN is Normal-only")
    print("[PASS] VALIDATION contains Normal and Attack")
    print("[PASS] FINAL_TEST contains Normal and Attack")
    print("[PASS] TRAIN is separated from evaluation sources")
    print("[PASS] No duplicated window between splits")
    print("[PASS] All five attack types exist in both evaluation splits")
    print(
        "[INFO] Validation/Final Test may share source files; "
        "their windows remain disjoint."
    )


# ============================================================
# 7. FIT STANDARD SCALER ON TRAIN ONLY
# ============================================================

def fit_scaler(
    df: pd.DataFrame,
) -> StandardScaler:

    print("\n[4] Fit StandardScaler using TRAIN only")
    print("-" * 76)

    train_df = df.loc[
        df["split"].eq("TRAIN")
    ]

    if train_df.empty:
        raise ValueError(
            "TRAIN ไม่มีข้อมูลสำหรับ Fit Scaler"
        )

    X_train = train_df[FEATURE_COLUMNS].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(X_train).all():
        raise ValueError(
            "TRAIN มี NaN หรือ Inf ก่อน Fit Scaler"
        )

    scaler = StandardScaler()
    scaler.fit(X_train)

    if not np.isfinite(scaler.mean_).all():
        raise ValueError(
            "พบ Mean ที่ไม่ใช่ finite ใน Scaler"
        )

    if not np.isfinite(scaler.scale_).all():
        raise ValueError(
            "พบ Scale ที่ไม่ใช่ finite ใน Scaler"
        )

    if (scaler.scale_ <= 0).any():
        raise ValueError(
            "พบ Scale ที่มีค่าไม่เป็นบวก"
        )

    print("[PASS] StandardScaler fitted")
    print(f"Fit rows : {len(train_df):,}")
    print(f"Features : {len(FEATURE_COLUMNS)}")
    print("Fit data : TRAIN only")

    return scaler


# ============================================================
# 8. TRANSFORM ALL SPLITS
# ============================================================

def transform_dataset(
    df: pd.DataFrame,
    scaler: StandardScaler,
) -> pd.DataFrame:

    print("\n[5] Transform TRAIN, VALIDATION and FINAL_TEST")
    print("-" * 76)

    result = df.copy()

    # Keep original non-feature columns unchanged.
    X = result[FEATURE_COLUMNS].to_numpy(
        dtype=np.float64
    )

    transformed = scaler.transform(X)

    if not np.isfinite(transformed).all():
        raise ValueError(
            "พบ NaN หรือ Inf หลัง Scaling"
        )

    result.loc[:, FEATURE_COLUMNS] = transformed

    # Confirm the transformed values are finite.
    scaled_values = result[FEATURE_COLUMNS].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(scaled_values).all():
        raise ValueError(
            "ข้อมูลหลัง Scaling มี NaN หรือ Inf"
        )

    # StandardScaler should yield near-zero TRAIN mean.
    train_scaled = result.loc[
        result["split"].eq("TRAIN"),
        FEATURE_COLUMNS,
    ].to_numpy(dtype=np.float64)

    train_mean = np.mean(
        train_scaled,
        axis=0,
    )

    if not np.allclose(
        train_mean,
        np.zeros(len(FEATURE_COLUMNS)),
        atol=EPSILON,
        rtol=0,
    ):
        raise RuntimeError(
            "Training scaled mean ไม่ใกล้ศูนย์ "
            "ตรวจสอบการ Fit/Transform"
        )

    print("[PASS] Same TRAIN scaler used on all splits")
    print("[PASS] Transformed features are finite")
    print("[PASS] TRAIN scaled mean approximately zero")

    return result


# ============================================================
# 9. BUILD SCALING PARAMETERS
# ============================================================

def create_scaling_parameters(
    scaler: StandardScaler,
) -> pd.DataFrame:

    rows = []

    for index, feature in enumerate(FEATURE_COLUMNS):
        rows.append({
            "feature": feature,
            "train_mean": float(scaler.mean_[index]),
            "train_variance": float(scaler.var_[index]),
            "scale": float(scaler.scale_[index]),
            "train_rows": int(scaler.n_samples_seen_),
            "scaler": "StandardScaler",
            "fit_split": "TRAIN",
        })

    return pd.DataFrame(rows)


# ============================================================
# 10. BUILD SCALING SUMMARY
# ============================================================

def create_scaling_summary(
    scaled_df: pd.DataFrame,
    scaler: StandardScaler,
) -> pd.DataFrame:

    rows = []

    for split_name in SPLIT_ORDER:

        part = scaled_df.loc[
            scaled_df["split"].eq(split_name)
        ]

        labels = part["_label_numeric"].to_numpy(
            dtype=np.int8
        )

        normal_count = int(np.sum(labels == 0))
        attack_count = int(np.sum(labels == 1))

        rows.append({
            "stage": "007",
            "dataset": "CIC-IDS2017",
            "split": split_name,
            "window_seconds": WINDOW_SECONDS,
            "rows": len(part),
            "normal_count": normal_count,
            "attack_count": attack_count,
            "source_file_count": int(part["source_file"].nunique()),
            "feature_count": len(FEATURE_COLUMNS),
            "features": "|".join(FEATURE_COLUMNS),
            "scaler": "StandardScaler",
            "scaler_fit_split": "TRAIN",
            "scaler_fit_rows": int(scaler.n_samples_seen_),
            "transformed": True,
            "final_test_used_for_fit": False,
        })

    return pd.DataFrame(rows)


# ============================================================
# 11. SAVE OUTPUTS
# ============================================================

def save_outputs(
    scaled_df: pd.DataFrame,
    scaler: StandardScaler,
    parameters_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> list[Path]:

    print("\n[6] Save scaled datasets and scaler")
    print("-" * 76)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_df = scaled_df.drop(
        columns=["_label_numeric"],
        errors="ignore",
    ).copy()

    train_df = output_df.loc[
        output_df["split"].eq("TRAIN")
    ].copy()

    validation_df = output_df.loc[
        output_df["split"].eq("VALIDATION")
    ].copy()

    final_test_df = output_df.loc[
        output_df["split"].eq("FINAL_TEST")
    ].copy()

    expected_rows = len(scaled_df)

    if len(output_df) != expected_rows:
        raise RuntimeError(
            "จำนวนแถวรวมเปลี่ยนแปลงก่อนบันทึก"
        )

    if (
        len(train_df)
        + len(validation_df)
        + len(final_test_df)
        != expected_rows
    ):
        raise RuntimeError(
            "ผลรวมจำนวนแถวแต่ละ Split ไม่ตรงกับ Dataset"
        )

    atomic_csv(output_df, OUTPUT_FILE)
    atomic_csv(train_df, TRAIN_OUTPUT_FILE)
    atomic_csv(validation_df, VALIDATION_OUTPUT_FILE)
    atomic_csv(final_test_df, FINAL_TEST_OUTPUT_FILE)

    atomic_joblib(scaler, SCALER_FILE)

    atomic_csv(parameters_df, PARAMETERS_FILE)
    atomic_csv(summary_df, SUMMARY_FILE)

    saved_files = [
        OUTPUT_FILE,
        TRAIN_OUTPUT_FILE,
        VALIDATION_OUTPUT_FILE,
        FINAL_TEST_OUTPUT_FILE,
        SCALER_FILE,
        PARAMETERS_FILE,
        SUMMARY_FILE,
    ]

    for path in saved_files:
        if not path.is_file():
            raise RuntimeError(
                f"ไม่พบไฟล์หลังบันทึก: {path}"
            )

    print("[PASS] All output files saved")

    return saved_files


# ============================================================
# 12. MAIN
# ============================================================

def main() -> None:

    print("=" * 76)
    print("STAGE 007 — CIC-IDS2017 FEATURE SCALING")
    print("=" * 76)

    print(f"Input      : {INPUT_FILE}")
    print(f"Output     : {OUTPUT_FILE}")
    print(f"Features   : {len(FEATURE_COLUMNS)} (all 13 Behavioral Features)")
    print("Scaler     : StandardScaler")
    print("Fit data   : TRAIN only")
    print("Final Test : Transform only")
    print("=" * 76)

    # 1. Check input file.
    print("\n[1] Check input file")
    check_input_file()

    # 2. Load dataset.
    df = load_dataset()

    # 3. Validate input and splits.
    df, y = validate_dataset(df)
    validate_splits(df, y)

    # 4. Fit scaler on TRAIN only.
    scaler = fit_scaler(df)

    # 5. Transform all splits.
    scaled_df = transform_dataset(
        df,
        scaler,
    )

    # 6. Save scaler parameters.
    parameters_df = create_scaling_parameters(
        scaler,
    )

    # 7. Create summary.
    summary_df = create_scaling_summary(
        scaled_df,
        scaler,
    )

    # 8. Save outputs.
    saved_files = save_outputs(
        scaled_df,
        scaler,
        parameters_df,
        summary_df,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print()
    print("=" * 76)
    print("STAGE 007 SUMMARY")
    print("=" * 76)

    print(f"Input rows      : {len(df):,}")
    print(f"Output rows     : {len(scaled_df):,}")
    print(f"Features        : {len(FEATURE_COLUMNS)}")

    for split_name in SPLIT_ORDER:
        count = int(
            scaled_df["split"].eq(split_name).sum()
        )
        print(f"{split_name:15}: {count:,}")

    print()
    print("SCALING VALIDATION")
    print("-" * 76)
    print("Scaler fit on TRAIN only       : PASS")
    print("Same scaler used on all splits : PASS")
    print("TRAIN scaled mean approx. zero : PASS")
    print("No NaN or Inf after scaling    : PASS")
    print("No row loss                    : PASS")
    print("FINAL_TEST used for fitting    : NO")

    print()
    print("SPLIT SUMMARY")
    print("-" * 76)

    print(
        summary_df[
            [
                "split",
                "rows",
                "normal_count",
                "attack_count",
                "source_file_count",
            ]
        ].to_string(index=False)
    )

    print()
    print("SCALING PARAMETERS")
    print("-" * 76)

    print(
        parameters_df[
            [
                "feature",
                "train_mean",
                "train_variance",
                "scale",
            ]
        ].to_string(index=False)
    )

    print()
    print("SAVED FILES")
    print("-" * 76)

    for path in saved_files:
        print(path)

    print("=" * 76)
    print("STAGE 007 COMPLETED")
    print("=" * 76)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"\nSTAGE 007 FAILED: {exc}",
            file=sys.stderr,
        )
        raise