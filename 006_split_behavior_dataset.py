from __future__ import annotations

from pathlib import Path
import os
import re
import sys

import numpy as np
import pandas as pd


# ============================================================
# STAGE 006 — CIC-IDS2017 SINGLE CHRONOLOGICAL SPLIT
# ============================================================
# Input:
#   data/time_windows/cic_behavior_features_10s.csv
#
# Output:
#   data/time_windows/cic_behavior_dataset_10s.csv
#   data/time_windows/cic_split_summary_10s.csv
#
# Split protocol:
#   Monday BENIGN                         -> TRAIN
#   Wednesday/Friday, first chronological
#     half of each source + traffic group -> VALIDATION
#   Wednesday/Friday, second half         -> FINAL_TEST
#
# Stage 005 provides the complete 13 Behavioral Features.
# Stage 006 preserves all 13 Features without feature selection.
#
# Feature selection is performed later in Stage 008.
#
# No random split, no Fold 1/2, no duplicated windows.
# VALIDATION and FINAL_TEST use different windows from the same
# Wednesday/Friday captures; this is a temporal window split,
# not a run-independent evaluation. FINAL_TEST is not used for
# model or threshold selection.
# ============================================================


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "time_windows"

INPUT_FILE = DATA_DIR / "cic_behavior_features_10s.csv"
OUTPUT_FILE = DATA_DIR / "cic_behavior_dataset_10s.csv"
SUMMARY_FILE = DATA_DIR / "cic_split_summary_10s.csv"

WINDOW_SECONDS = 10
VALIDATION_RATIO = 0.50

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

SPLIT_ORDER = ["TRAIN", "VALIDATION", "FINAL_TEST"]

EXPECTED_ATTACK_TYPES = {
    "dosslowloris": "DoS_Slowloris",
    "dosslowhttptest": "DoS_Slowhttptest",
    "doshulk": "DoS_Hulk",
    "dosgoldeneye": "DoS_GoldenEye",
    "ddosloic": "DDoS_LOIC",
}

EXPECTED_ATTACKS_BY_DAY = {
    "wednesday": {
        "DoS_Slowloris",
        "DoS_Slowhttptest",
        "DoS_Hulk",
        "DoS_GoldenEye",
    },
    "friday": {"DDoS_LOIC"},
}

DISALLOWED_ATTACK_TOKENS = ("portscan", "botnet", "heartbleed")

NORMAL_LABELS = {
    "normal", "benign", "legitimate", "clean", "background",
    "0", "0.0", "false", "no",
}

ATTACK_LABELS = {
    "attack", "malicious", "1", "1.0", "true", "yes",
}


# ============================================================
# 2. HELPERS
# ============================================================

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = (
        result.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    duplicated = result.columns[result.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(f"พบชื่อคอลัมน์ซ้ำ: {duplicated}")

    return result


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""

    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def get_source_day(value: object) -> str | None:
    if pd.isna(value):
        return None

    name = str(value).replace("\\", "/").split("/")[-1].lower()

    if "monday" in name:
        return "monday"
    if "wednesday" in name:
        return "wednesday"
    if "friday" in name:
        return "friday"

    return None


def normalize_binary_label(value) -> str | None:
    if pd.isna(value):
        return None

    token = str(value).strip().lower()

    if token in NORMAL_LABELS:
        return "Normal"
    if token in ATTACK_LABELS:
        return "Attack"

    return None


def canonical_attack_type(row: pd.Series) -> str:
    """Resolve one attack row to one of the five selected CIC types."""
    values = []

    for column in ("attack_type", "traffic_type"):
        if column not in row.index or pd.isna(row[column]):
            continue

        value = str(row[column]).strip()
        token = normalize_text(value)

        if not token or token in NORMAL_LABELS or token in ATTACK_LABELS:
            continue

        for excluded in DISALLOWED_ATTACK_TOKENS:
            if excluded in token:
                raise ValueError(
                    f"พบประเภทการโจมตีที่ Stage 004 ควรตัดออก: {value}"
                )

        values.append((value, token))

    for value, token in values:
        for key, canonical_name in EXPECTED_ATTACK_TYPES.items():
            if key in token:
                return canonical_name

    raise ValueError(
        "ไม่สามารถระบุ Attack Type จาก attack_type/traffic_type ได้: "
        f"{values}"
    )


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")

    try:
        df.to_csv(temp_path, index=False, encoding="utf-8-sig")
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


# ============================================================
# 3. LOAD AND VALIDATE INPUT
# ============================================================

def load_dataset() -> pd.DataFrame:
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"ไม่พบไฟล์ Stage 005:\n{INPUT_FILE}"
        )

    print("\n[1] Load Stage 005 behavioral features")
    df = pd.read_csv(INPUT_FILE, low_memory=False, encoding="utf-8-sig")
    df = clean_columns(df).reset_index(drop=True)

    if df.empty:
        raise ValueError("Stage 005 ไม่มีข้อมูล")

    required = [
        "source_file", "window_id", "binary_label", "traffic_type",
        *FEATURE_COLUMNS,
    ]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Stage 005 ขาดคอลัมน์ที่จำเป็น: {missing}")

    print(f"Input rows   : {len(df):,}")
    print(f"Input columns: {len(df.columns):,}")
    return df


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[2] Validate Stage 005 data")
    print("-" * 76)

    result = df.copy().reset_index(drop=True)

    if result["source_file"].isna().any():
        raise ValueError("พบ source_file ว่าง")
    if result["window_id"].isna().any():
        raise ValueError("พบ window_id ว่าง")

    result["source_file"] = result["source_file"].astype(str).str.strip()
    if result["source_file"].eq("").any():
        raise ValueError("พบ source_file เป็นข้อความว่าง")

    window_id = pd.to_numeric(result["window_id"], errors="coerce")
    invalid_window_id = (
        window_id.isna()
        | ~np.isfinite(window_id)
        | (window_id < 1)
        | (window_id % 1 != 0)
    )
    if invalid_window_id.any():
        raise ValueError(
            "window_id ต้องเป็นจำนวนเต็มบวก "
            f"(invalid={int(invalid_window_id.sum()):,})"
        )
    result["window_id"] = window_id.astype("int64")

    duplicated = result.duplicated(["source_file", "window_id"], keep=False)
    if duplicated.any():
        examples = result.loc[
            duplicated, ["source_file", "window_id"]
        ].head(10)
        raise ValueError(
            "พบ source_file + window_id ซ้ำ:\n"
            + examples.to_string(index=False)
        )

    result["_normalized_label"] = result["binary_label"].map(
        normalize_binary_label
    )
    if result["_normalized_label"].isna().any():
        invalid = (
            result.loc[
                result["_normalized_label"].isna(), "binary_label"
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(f"พบ binary_label ที่ไม่รู้จัก: {invalid}")

    for feature in FEATURE_COLUMNS:
        values = pd.to_numeric(result[feature], errors="coerce")
        invalid = values.isna() | ~np.isfinite(values)

        if invalid.any():
            raise ValueError(
                f"Feature {feature} มี NaN/Inf "
                f"{int(invalid.sum()):,} rows"
            )
        if (values < 0).any():
            raise ValueError(f"พบค่าติดลบใน Feature {feature}")

        result[feature] = values.astype("float64")

    if (result["new_ip_ratio"] > 1 + 1e-12).any():
        raise ValueError("new_ip_ratio ต้องอยู่ในช่วง 0 ถึง 1")

    result["source_day"] = result["source_file"].map(get_source_day)
    if result["source_day"].isna().any():
        unknown = sorted(
            result.loc[result["source_day"].isna(), "source_file"]
            .astype(str).unique().tolist()
        )
        raise ValueError(
            "ไม่สามารถระบุวันจาก source_file ได้:\n" + "\n".join(unknown)
        )

    unknown_days = sorted(
        set(result["source_day"].unique()) - {"monday", "wednesday", "friday"}
    )
    if unknown_days:
        raise ValueError(f"พบ Source Day ที่ไม่รองรับ: {unknown_days}")

    groups = []
    for _, row in result.iterrows():
        if row["_normalized_label"] == "Normal":
            groups.append("Normal")
        else:
            groups.append(canonical_attack_type(row))
    result["_split_group"] = groups

    monday_attacks = result.loc[
        result["source_day"].eq("monday")
        & result["_normalized_label"].eq("Attack")
    ]
    if not monday_attacks.empty:
        raise ValueError("พบ Attack ใน Monday แต่ TRAIN กำหนดให้เป็น Normal-only")

    for day, allowed_attacks in EXPECTED_ATTACKS_BY_DAY.items():
        day_attacks = result.loc[
            result["source_day"].eq(day)
            & result["_normalized_label"].eq("Attack"),
            "_split_group",
        ]
        unexpected = sorted(set(day_attacks.unique()) - allowed_attacks)
        if unexpected:
            raise ValueError(
                f"พบ Attack Type ไม่ตรงกับชุดข้อมูลของ {day}: {unexpected}"
            )

    observed_attacks = set(
        result.loc[
            result["_normalized_label"].eq("Attack"), "_split_group"
        ].unique()
    )
    expected_attacks = set(EXPECTED_ATTACK_TYPES.values())
    missing_attacks = sorted(expected_attacks - observed_attacks)
    unexpected_attacks = sorted(observed_attacks - expected_attacks)

    if missing_attacks:
        raise ValueError(f"ไม่พบ Attack Type ที่กำหนด: {missing_attacks}")
    if unexpected_attacks:
        raise ValueError(f"พบ Attack Type ที่ไม่อยู่ในรายการ: {unexpected_attacks}")

    print("[PASS] Required columns")
    print("[PASS] Positive integer window_id")
    print("[PASS] Unique source_file + window_id")
    print("[PASS] Binary labels")
    print(f"[PASS] {len(FEATURE_COLUMNS)} model features (finite and non-negative)")
    print("[PASS] Source day and attack schedule")
    print("[PASS] All five selected attack types")
    return result


# ============================================================
# 4. SINGLE CHRONOLOGICAL SPLIT
# ============================================================

def split_group_by_midpoint(
    group: pd.DataFrame,
) -> tuple[list[int], list[int]]:
    """Earlier 50% by window_id -> VALIDATION; later 50% -> FINAL_TEST."""
    ordered = group.sort_values("window_id", kind="mergesort")
    n_rows = len(ordered)

    if n_rows < 2:
        source = ordered["source_file"].iloc[0] if n_rows else "unknown"
        traffic = ordered["_split_group"].iloc[0] if n_rows else "unknown"
        raise ValueError(
            "แต่ละกลุ่มในชุดประเมินต้องมีอย่างน้อย 2 Windows "
            f"(source={source}, group={traffic}, rows={n_rows})"
        )

    cut = int(np.floor(n_rows * VALIDATION_RATIO))
    cut = min(max(cut, 1), n_rows - 1)

    return ordered.index[:cut].tolist(), ordered.index[cut:].tolist()


def assign_split(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[3] Assign one split")
    print("-" * 76)

    result = df.copy()
    result["split"] = pd.Series(index=result.index, dtype="object")

    # Monday BENIGN -> TRAIN.
    monday_mask = result["source_day"].eq("monday")
    result.loc[monday_mask, "split"] = "TRAIN"

    # Wednesday/Friday: split each source + traffic group chronologically.
    evaluation_df = result.loc[
        result["source_day"].isin(["wednesday", "friday"])
    ]

    validation_indices: list[int] = []
    final_test_indices: list[int] = []

    grouped = evaluation_df.groupby(
        ["source_file", "_split_group"], sort=True, dropna=False
    )
    for (source_file, traffic_group), group in grouped:
        val_idx, test_idx = split_group_by_midpoint(group)
        validation_indices.extend(val_idx)
        final_test_indices.extend(test_idx)

        print(
            f"[SPLIT] {source_file} | {traffic_group} | "
            f"total={len(group):,} | validation={len(val_idx):,} | "
            f"final_test={len(test_idx):,}"
        )

    result.loc[validation_indices, "split"] = "VALIDATION"
    result.loc[final_test_indices, "split"] = "FINAL_TEST"

    if result["split"].isna().any():
        raise ValueError(
            f"มี {int(result['split'].isna().sum()):,} Windows ที่ยังไม่ได้กำหนด Split"
        )

    return result


# ============================================================
# 5. SPLIT VALIDATION
# ============================================================

def validate_splits(df: pd.DataFrame) -> None:
    print("\n[4] Validate split integrity")
    print("-" * 76)

    if df["split"].isna().any():
        raise ValueError("พบแถวที่ไม่มี Split")

    parts = {
        name: df.loc[df["split"].eq(name)].copy()
        for name in SPLIT_ORDER
    }
    for name, part in parts.items():
        if part.empty:
            raise ValueError(f"{name} ไม่มีข้อมูล")

    train = parts["TRAIN"]
    validation = parts["VALIDATION"]
    final_test = parts["FINAL_TEST"]

    if not train["source_day"].eq("monday").all():
        raise ValueError("TRAIN ต้องมาจาก Monday เท่านั้น")
    if not train["_normalized_label"].eq("Normal").all():
        raise ValueError("TRAIN ต้องมีเฉพาะ Normal")

    # TRAIN is a separate capture from both evaluation partitions.
    train_sources = set(train["source_file"])
    validation_sources = set(validation["source_file"])
    final_test_sources = set(final_test["source_file"])

    if train_sources & validation_sources:
        raise ValueError("พบ Source File ของ TRAIN ซ้ำใน VALIDATION")
    if train_sources & final_test_sources:
        raise ValueError("พบ Source File ของ TRAIN ซ้ำใน FINAL_TEST")

    # No individual window can appear in more than one partition.
    split_keys = {
        name: set(zip(part["source_file"], part["window_id"].astype(int)))
        for name, part in parts.items()
    }
    for first, second in (
        ("TRAIN", "VALIDATION"),
        ("TRAIN", "FINAL_TEST"),
        ("VALIDATION", "FINAL_TEST"),
    ):
        overlap = split_keys[first] & split_keys[second]
        if overlap:
            raise ValueError(
                f"พบ Window ซ้ำระหว่าง {first} และ {second}: {list(overlap)[:5]}"
            )

    expected = set(EXPECTED_ATTACK_TYPES.values())
    for name in ("VALIDATION", "FINAL_TEST"):
        part = parts[name]
        if not part["_normalized_label"].eq("Normal").any():
            raise ValueError(f"{name} ไม่มี Normal")

        observed = set(
            part.loc[
                part["_normalized_label"].eq("Attack"), "_split_group"
            ].unique()
        )
        missing = sorted(expected - observed)
        if missing:
            raise ValueError(f"{name} ไม่มี Attack Type: {missing}")

    shared_eval_sources = validation_sources & final_test_sources

    print("[PASS] TRAIN is Monday Normal-only")
    print("[PASS] TRAIN source is separate from evaluation sources")
    print("[PASS] No duplicated windows between splits")
    print("[PASS] VALIDATION contains Normal and all five attack types")
    print("[PASS] FINAL_TEST contains Normal and all five attack types")
    print(
        "[INFO] VALIDATION and FINAL_TEST share "
        f"{len(shared_eval_sources)} evaluation capture source(s)."
    )
    print(
        "[LIMITATION] This is a chronological window split; "
        "VALIDATION and FINAL_TEST are not independent captures."
    )


# ============================================================
# 6. SUMMARY AND OUTPUT
# ============================================================

def create_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for split_name in SPLIT_ORDER:
        part = df.loc[df["split"].eq(split_name)]
        normal_count = int(part["_normalized_label"].eq("Normal").sum())
        attack_part = part.loc[part["_normalized_label"].eq("Attack")]
        attack_counts = attack_part.groupby("_split_group").size().sort_index()

        source_days = [
            day for day in ("monday", "wednesday", "friday")
            if day in set(part["source_day"])
        ]
        source_files = sorted(part["source_file"].astype(str).unique())

        rows.append({
            "split": split_name,
            "source_day": "|".join(source_days),
            "source_files": "|".join(source_files),
            "window_count": int(len(part)),
            "normal_count": normal_count,
            "attack_count": int(len(attack_part)),
            "attack_types": "|".join(attack_counts.index.astype(str)),
            "attack_type_counts": "|".join(
                f"{name}:{int(count)}" for name, count in attack_counts.items()
            ),
            "feature_count": len(FEATURE_COLUMNS),
            "features": "|".join(FEATURE_COLUMNS),
            "window_seconds": WINDOW_SECONDS,
            "split_method": "chronological 50/50 by source_file and traffic group",
            "run_independent": False,
            "final_test_used_for_selection": False,
        })

    return pd.DataFrame(rows)


def save_outputs(df: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    output = df.drop(
        columns=["_normalized_label", "_split_group"], errors="ignore"
    ).copy()

    # Preserve all 13 Behavioral Features from Stage 005.
    # Feature selection is intentionally NOT performed in Stage 006.
    metadata_columns = [
        column
        for column in output.columns
        if column not in FEATURE_COLUMNS
    ]
    output = output[metadata_columns + FEATURE_COLUMNS]

    if len(output) != len(df):
        raise RuntimeError("จำนวนแถวเปลี่ยนแปลงก่อนบันทึก")

    invalid_splits = sorted(set(output["split"].astype(str)) - set(SPLIT_ORDER))
    if invalid_splits:
        raise ValueError(f"พบ Split ที่ไม่รู้จัก: {invalid_splits}")

    atomic_csv(output, OUTPUT_FILE)
    atomic_csv(summary, SUMMARY_FILE)
    return output


# ============================================================
# 7. MAIN
# ============================================================

def main() -> None:
    print("=" * 78)
    print("STAGE 006 — CIC-IDS2017 SINGLE CHRONOLOGICAL SPLIT")
    print("=" * 78)
    print(f"Input          : {INPUT_FILE}")
    print(f"Output         : {OUTPUT_FILE}")
    print(f"Summary        : {SUMMARY_FILE}")
    print(f"Window         : {WINDOW_SECONDS} seconds")
    print(f"Features       : {len(FEATURE_COLUMNS)} (13 Behavioral Features; no feature selection)")
    print("TRAIN          : Monday Normal only")
    print("VALIDATION     : Earlier 50% of each Wednesday/Friday group")
    print("FINAL_TEST     : Later 50% of each Wednesday/Friday group")
    print("Random split   : NO")
    print("Fold 1 / 2     : NOT USED")
    print("Final Test for selection: NO")
    print("=" * 78)

    df = load_dataset()
    df = validate_input(df)
    df = assign_split(df)
    validate_splits(df)

    summary = create_summary(df)
    output = save_outputs(df, summary)

    print("\n" + "=" * 78)
    print("STAGE 006 SUMMARY")
    print("=" * 78)
    print(f"Input rows  : {len(df):,}")
    print(f"Output rows : {len(output):,}")
    print(f"Features    : {len(FEATURE_COLUMNS)} (13 Behavioral Features; no feature selection)")
    print("Window loss : 0")
    print("Random split: NO")
    print("Fold        : NONE")

    print("\nSPLIT DISTRIBUTION")
    print("-" * 78)
    print(summary[[
        "split", "window_count", "normal_count", "attack_count", "attack_types"
    ]].to_string(index=False))

    print("\nATTACK TYPE BY SPLIT")
    print("-" * 78)
    attack_summary = (
        df.loc[df["_normalized_label"].eq("Attack")]
        .groupby(["split", "_split_group"], dropna=False)
        .size().rename("count").reset_index()
    )
    if not attack_summary.empty:
        print(attack_summary.to_string(index=False))
    else:
        print("ไม่พบ Attack rows สำหรับแสดงสรุป")

    print("\nSAVED FILES")
    print(f"Dataset : {OUTPUT_FILE}")
    print(f"Summary : {SUMMARY_FILE}")
    print("=" * 78)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nSTAGE 006 FAILED: {exc}", file=sys.stderr)
        raise