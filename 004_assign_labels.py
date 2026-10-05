
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import os

import numpy as np
import pandas as pd


# ============================================================
# STAGE 004 - CIC-IDS2017 LABEL ASSIGNMENT
# COMPLETE 10-SECOND WINDOWS ONLY
# ============================================================
#
# INPUT:
#   data/time_windows/cic_time_windows_10s.csv
#
# OUTPUT:
#   data/time_windows/cic_labeled_time_windows_10s.csv
#   data/time_windows/cic_time_windows_10s_label_audit.csv
#   data/time_windows/cic_labeled_time_windows_10s_summary.csv
#
# DATA SELECTION:
#   Monday:
#       BENIGN -> Normal
#
#   Wednesday:
#       BENIGN
#       DoS slowloris
#       DoS Slowhttptest
#       DoS Hulk
#       DoS GoldenEye
#
#   Friday:
#       BENIGN
#       DDoS LOIC
#
# EXCLUDED:
#   PortScan, Botnet, Heartbleed,
#   and other unselected attack periods
#
# IMPORTANT:
#   - Read only the Stage 003 CSV.
#   - Do not scan PCAP/PCAPNG.
#   - Require complete 10-second windows.
#   - Use only Stage 003 window metadata.
#   - Assign labels using timestamp overlap.
#   - Exclude unselected attack periods.
#   - Never relabel excluded attacks as Normal.
# ============================================================


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "time_windows"
    / "cic_time_windows_10s.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "time_windows"

OUTPUT_FILE = (
    OUTPUT_DIR / "cic_labeled_time_windows_10s.csv"
)

AUDIT_FILE = (
    OUTPUT_DIR / "cic_time_windows_10s_label_audit.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR / "cic_labeled_time_windows_10s_summary.csv"
)

WINDOW_SECONDS = 10.0
WINDOW_TOLERANCE = 1e-5

# CIC-IDS2017 capture timestamps are converted to
# Atlantic Daylight Time (UTC-3).
CIC_TZ = timezone(
    timedelta(hours=-3),
    name="ADT",
)

UTC_TZ = timezone.utc

MONDAY = date(2017, 7, 3)
WEDNESDAY = date(2017, 7, 5)
FRIDAY = date(2017, 7, 7)

EXPECTED_DATES = {
    MONDAY,
    WEDNESDAY,
    FRIDAY,
}


# ============================================================
# 2. SELECTED ATTACK SCHEDULE
# ============================================================
#
# Format:
# (start time, end time, attack_type, original CIC label)
#
# End minute is inclusive.
# Example:
# 09:47-10:10 means [09:47:00, 10:11:00)
# ============================================================

SELECTED_EVENTS = {
    WEDNESDAY: [
        (
            "09:47",
            "10:10",
            "DoS_Slowloris",
            "DoS slowloris",
        ),
        (
            "10:14",
            "10:35",
            "DoS_Slowhttptest",
            "DoS Slowhttptest",
        ),
        (
            "10:43",
            "11:00",
            "DoS_Hulk",
            "DoS Hulk",
        ),
        (
            "11:10",
            "11:23",
            "DoS_GoldenEye",
            "DoS GoldenEye",
        ),
    ],
    FRIDAY: [
        (
            "15:56",
            "16:16",
            "DDoS_LOIC",
            "DDoS",
        ),
    ],
}


# ============================================================
# 3. EXCLUDED ATTACK SCHEDULE
# ============================================================

PORTSCAN_INTERVALS = [
    ("13:55", "13:57"),
    ("13:58", "14:00"),
    ("14:01", "14:04"),
    ("14:05", "14:07"),
    ("14:08", "14:10"),
    ("14:11", "14:13"),
    ("14:14", "14:16"),
    ("14:17", "14:19"),
    ("14:20", "14:21"),
    ("14:22", "14:24"),
    ("14:33", "14:33"),
    ("14:35", "14:35"),
    ("14:51", "14:53"),
    ("14:54", "14:56"),
    ("14:57", "14:59"),
    ("15:00", "15:02"),
    ("15:03", "15:05"),
    ("15:06", "15:07"),
    ("15:08", "15:10"),
    ("15:11", "15:12"),
    ("15:13", "15:15"),
    ("15:16", "15:18"),
    ("15:19", "15:21"),
    ("15:22", "15:24"),
    ("15:25", "15:25"),
    ("15:26", "15:27"),
    ("15:28", "15:29"),
]

EXCLUDED_EVENTS = {
    WEDNESDAY: [
        ("15:12", "15:32", "Heartbleed"),
    ],
    FRIDAY: [
        ("10:02", "11:02", "Botnet_ARES"),
        *[
            (start, end, "PortScan")
            for start, end in PORTSCAN_INTERVALS
        ],
    ],
}

ALLOWED_ATTACKS = {
    "DoS_Slowloris",
    "DoS_Slowhttptest",
    "DoS_Hulk",
    "DoS_GoldenEye",
    "DDoS_LOIC",
}

EXCLUDED_NAMES = {
    "PortScan",
    "Botnet_ARES",
    "Heartbleed",
}


# ============================================================
# 4. TIME UTILITIES
# ============================================================

def find_column(
    df: pd.DataFrame,
    candidates: tuple[str, ...],
) -> str | None:
    """ค้นหาคอลัมน์โดยไม่สนใจตัวพิมพ์เล็กและตัวพิมพ์ใหญ่"""

    lookup = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        result = lookup.get(candidate.lower())

        if result is not None:
            return result

    return None


def epoch_to_seconds(
    values: pd.Series,
    column_name: str,
) -> pd.Series:
    """
    แปลง Epoch ให้เป็นวินาที

    รองรับ:
        Seconds
        Milliseconds
        Microseconds
        Nanoseconds
    """

    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    if numeric.isna().any():
        raise ValueError(
            f"พบค่าเวลาไม่ถูกต้องใน {column_name}"
        )

    values_array = numeric.to_numpy(dtype=float)

    if not np.isfinite(values_array).all():
        raise ValueError(
            f"พบค่าเวลาไม่เป็นจำนวนจำกัดใน {column_name}"
        )

    magnitude = float(numeric.abs().median())

    if magnitude >= 1e17:
        factor = 1e-9
    elif magnitude >= 1e14:
        factor = 1e-6
    elif magnitude >= 1e11:
        factor = 1e-3
    else:
        factor = 1.0

    result = numeric.astype(float) * factor

    if not np.isfinite(result.to_numpy()).all():
        raise ValueError(
            f"ไม่สามารถแปลงเวลาใน {column_name} ได้"
        )

    return result


def get_epoch_seconds(
    df: pd.DataFrame,
    epoch_candidates: tuple[str, ...],
    utc_candidates: tuple[str, ...],
    description: str,
) -> pd.Series:
    """
    อ่านเวลาโดยเลือก Epoch ก่อน
    หากไม่มี Epoch ให้ใช้คอลัมน์ UTC
    """

    epoch_col = find_column(
        df,
        epoch_candidates,
    )

    if epoch_col is not None:
        return epoch_to_seconds(
            df[epoch_col],
            epoch_col,
        )

    utc_col = find_column(
        df,
        utc_candidates,
    )

    if utc_col is not None:
        parsed = pd.to_datetime(
            df[utc_col],
            errors="coerce",
            utc=True,
        )

        if parsed.isna().any():
            raise ValueError(
                f"พบเวลาไม่ถูกต้องใน {utc_col}"
            )

        return parsed.astype("int64") / 1e9

    raise ValueError(
        f"ไม่พบคอลัมน์เวลา {description}\n"
        f"รองรับ: {epoch_candidates} หรือ {utc_candidates}"
    )


def schedule_epoch(
    day: date,
    hhmm: str,
    is_end: bool = False,
) -> float:
    """แปลงเวลาเหตุการณ์จาก ADT เป็น Unix Epoch"""

    parsed_time = datetime.strptime(
        hhmm,
        "%H:%M",
    ).time()

    local_dt = datetime.combine(
        day,
        parsed_time,
    ).replace(tzinfo=CIC_TZ)

    if is_end:
        # รวมเวลาทั้งนาทีสุดท้าย
        local_dt += timedelta(minutes=1)

    return local_dt.astimezone(
        UTC_TZ
    ).timestamp()


def overlap_mask(
    start_epoch: pd.Series,
    end_epoch: pd.Series,
    day: date,
    start_hhmm: str,
    end_hhmm: str,
) -> pd.Series:
    """
    ตรวจสอบว่า Window ทับซ้อนกับช่วงเหตุการณ์หรือไม่

    ใช้ช่วงเวลาแบบ [start, end)
    """

    attack_start = schedule_epoch(
        day,
        start_hhmm,
        is_end=False,
    )

    attack_end = schedule_epoch(
        day,
        end_hhmm,
        is_end=True,
    )

    return (
        (start_epoch < attack_end)
        & (end_epoch > attack_start)
    )


# ============================================================
# 5. VALIDATE INPUT WINDOWS
# ============================================================

def validate_input(
    df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """
    ตรวจสอบโครงสร้างข้อมูลและ Window
    ก่อนดำเนินการกำหนด Label
    """

    required = {
        "source_file",
        "window_id",
        "window_start_epoch",
        "window_end_epoch",
    }

    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            "ไฟล์ Stage 003 ขาดคอลัมน์ที่จำเป็น:\n"
            + "\n".join(missing)
        )

    if df.empty:
        raise ValueError(
            "ไฟล์ Time Window ไม่มีข้อมูล"
        )

    if df["source_file"].isna().any():
        raise ValueError(
            "พบ source_file ที่ว่าง"
        )

    if df["source_file"].astype(str).str.strip().eq("").any():
        raise ValueError(
            "พบ source_file ที่เป็นข้อความว่าง"
        )

    # อ่านเวลาเริ่มต้นและสิ้นสุด
    start_epoch = get_epoch_seconds(
        df,
        ("window_start_epoch",),
        ("window_start_utc",),
        "เริ่มต้น Window",
    )

    end_epoch = get_epoch_seconds(
        df,
        ("window_end_epoch",),
        ("window_end_utc",),
        "สิ้นสุด Window",
    )

    # --------------------------------------------------------
    # ตรวจสอบความยาว Window
    # --------------------------------------------------------

    durations = end_epoch - start_epoch

    invalid_duration = (
        (durations <= 0)
        | (
            ~np.isclose(
                durations,
                WINDOW_SECONDS,
                rtol=0,
                atol=WINDOW_TOLERANCE,
            )
        )
    )

    if invalid_duration.any():
        bad = int(invalid_duration.sum())

        examples = df.loc[
            invalid_duration,
            ["source_file", "window_id"],
        ].head(10).to_dict("records")

        raise ValueError(
            f"พบ Window ที่ไม่ครบ 10 วินาที {bad:,} แถว\n"
            f"ตัวอย่าง: {examples}\n"
            "โปรดตรวจสอบ Stage 003 ก่อนกำหนด Label"
        )

    # --------------------------------------------------------
    # ตรวจสอบ Window ID
    # --------------------------------------------------------

    window_id = pd.to_numeric(
        df["window_id"],
        errors="coerce",
    )

    if window_id.isna().any():
        raise ValueError(
            "พบ window_id ที่ไม่ถูกต้อง"
        )

    if (
        (window_id < 1).any()
        or not np.equal(
            window_id,
            np.floor(window_id),
        ).all()
    ):
        raise ValueError(
            "window_id ต้องเป็นจำนวนเต็มตั้งแต่ 1 ขึ้นไป"
        )

    if df.duplicated(
        ["source_file", "window_id"]
    ).any():
        duplicates = (
            df.loc[
                df.duplicated(
                    ["source_file", "window_id"],
                    keep=False,
                ),
                ["source_file", "window_id"],
            ]
            .head(10)
            .to_dict("records")
        )

        raise ValueError(
            "พบ source_file + window_id ซ้ำ: "
            f"{duplicates}"
        )

    # --------------------------------------------------------
    # ตรวจสอบ Window ทับซ้อน
    # --------------------------------------------------------

    check_df = df.copy()
    check_df["_start"] = start_epoch
    check_df["_end"] = end_epoch

    for source, group in check_df.groupby("source_file"):
        group = group.sort_values("_start")

        previous_end = None

        # ใช้ name=None เพื่อป้องกันปัญหา
        # ชื่อคอลัมน์ที่ขึ้นต้นด้วย underscore
        for current_start, current_end in (
            group[["_start", "_end"]]
            .itertuples(index=False, name=None)
        ):
            if (
                previous_end is not None
                and current_start
                < previous_end - WINDOW_TOLERANCE
            ):
                raise ValueError(
                    f"พบ Window ทับซ้อนกันใน {source}"
                )

            previous_end = current_end

    print("Input validation: PASS")
    print("Window duration : 10 seconds")
    print(f"Input rows      : {len(df):,}")

    return start_epoch, end_epoch


# ============================================================
# 6. ATOMIC CSV OUTPUT
# ============================================================

def save_csv(
    df: pd.DataFrame,
    path: Path,
) -> None:
    """บันทึกไฟล์ผ่านไฟล์ชั่วคราวก่อนแทนที่ไฟล์จริง"""

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


# ============================================================
# 7. MAIN
# ============================================================

def main() -> None:
    print("=" * 78)
    print("STAGE 004 - CIC-IDS2017 LABEL ASSIGNMENT")
    print("=" * 78)
    print(f"Input            : {INPUT_FILE}")
    print(f"Window           : {WINDOW_SECONDS} seconds")
    print("Window validation: Complete 10-second only")
    print("Monday           : BENIGN only")
    print("Wednesday        : BENIGN + 4 selected DoS")
    print("Friday           : BENIGN + DDoS LOIC")
    print("Excluded         : PortScan, Botnet, Heartbleed")
    print("Label method     : Timestamp schedule")
    print("PCAP scan        : NO")
    print("=" * 78)

    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            "ไม่พบไฟล์ Time Window:\n"
            f"{INPUT_FILE}\n"
            "โปรดรัน 003_create_time_windows.py ก่อน"
        )

    # --------------------------------------------------------
    # LOAD INPUT
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    # ตรวจสอบโครงสร้างและ Window ก่อน Label
    start_epoch, end_epoch = validate_input(df)

    # --------------------------------------------------------
    # CONVERT TIMEZONE
    # --------------------------------------------------------

    start_local = (
        pd.to_datetime(
            start_epoch,
            unit="s",
            utc=True,
        )
        .dt.tz_convert(CIC_TZ)
    )

    end_local = (
        pd.to_datetime(
            end_epoch,
            unit="s",
            utc=True,
        )
        .dt.tz_convert(CIC_TZ)
    )

    local_dates = start_local.dt.date

    unexpected_dates = sorted(
        set(local_dates) - EXPECTED_DATES
    )

    if unexpected_dates:
        raise ValueError(
            "พบวันที่ไม่อยู่ในชุดข้อมูลที่กำหนด "
            "หลังแปลงเป็นเวลา ADT:\n"
            + ", ".join(
                str(d) for d in unexpected_dates
            )
        )

    # --------------------------------------------------------
    # ADD LOCAL TIME COLUMNS
    # --------------------------------------------------------

    df["capture_date_adt"] = local_dates.astype(str)

    df["window_start_adt"] = (
        start_local.dt.strftime(
            "%Y-%m-%d %H:%M:%S%z"
        )
    )

    df["window_end_adt"] = (
        end_local.dt.strftime(
            "%Y-%m-%d %H:%M:%S%z"
        )
    )

    # --------------------------------------------------------
    # INITIAL LABELS
    # --------------------------------------------------------

    # กำหนดค่าเริ่มต้นเป็น BENIGN
    df["Label"] = "BENIGN"
    df["binary_label"] = "Normal"
    df["traffic_type"] = "Normal"
    df["attack_type"] = "Normal"
    df["label_status"] = "INCLUDED"
    df["exclude_reason"] = ""

    # ใช้ป้องกันการกำหนด Label ที่ขัดแย้งกัน
    event_owner = pd.Series(
        "",
        index=df.index,
        dtype="object",
    )

    def apply_event(
        mask: pd.Series,
        event_key: str,
    ) -> None:
        conflict = (
            mask
            & event_owner.ne("")
            & event_owner.ne(event_key)
        )

        if conflict.any():
            examples = (
                df.loc[
                    conflict,
                    "window_start_adt",
                ]
                .head(5)
                .tolist()
            )

            raise ValueError(
                "พบ Window ที่ทับซ้อนกับเหตุการณ์ "
                f"มากกว่า 1 รายการ: {event_key}: {examples}"
            )

        event_owner.loc[mask] = event_key

    # --------------------------------------------------------
    # ASSIGN SELECTED ATTACKS
    # --------------------------------------------------------

    selected_counts = {}

    for day, events in SELECTED_EVENTS.items():
        day_mask = local_dates.eq(day)

        for (
            start_hhmm,
            end_hhmm,
            attack_type,
            cic_label,
        ) in events:

            mask = (
                day_mask
                & overlap_mask(
                    start_epoch,
                    end_epoch,
                    day,
                    start_hhmm,
                    end_hhmm,
                )
            )

            apply_event(
                mask,
                attack_type,
            )

            df.loc[mask, "Label"] = cic_label
            df.loc[mask, "binary_label"] = "Attack"
            df.loc[mask, "traffic_type"] = attack_type
            df.loc[mask, "attack_type"] = attack_type

            selected_counts[attack_type] = (
                selected_counts.get(attack_type, 0)
                + int(mask.sum())
            )

    # --------------------------------------------------------
    # EXCLUDE UNSELECTED ATTACKS
    # --------------------------------------------------------

    excluded_counts = {}

    for day, events in EXCLUDED_EVENTS.items():
        day_mask = local_dates.eq(day)

        for (
            start_hhmm,
            end_hhmm,
            excluded_type,
        ) in events:

            mask = (
                day_mask
                & overlap_mask(
                    start_epoch,
                    end_epoch,
                    day,
                    start_hhmm,
                    end_hhmm,
                )
            )

            apply_event(
                mask,
                f"EXCLUDED_{excluded_type}",
            )

            df.loc[mask, "Label"] = "EXCLUDED"
            df.loc[mask, "binary_label"] = pd.NA
            df.loc[mask, "traffic_type"] = (
                f"EXCLUDED_{excluded_type}"
            )
            df.loc[mask, "attack_type"] = excluded_type
            df.loc[mask, "label_status"] = "EXCLUDED"
            df.loc[mask, "exclude_reason"] = excluded_type

            excluded_counts[excluded_type] = (
                excluded_counts.get(excluded_type, 0)
                + int(mask.sum())
            )

    # --------------------------------------------------------
    # VALIDATE LABELS
    # --------------------------------------------------------

    clean_df = df.loc[
        df["label_status"].eq("INCLUDED")
    ].copy()

    if clean_df.empty:
        raise ValueError(
            "ไม่พบ Window ที่สามารถนำไปใช้ได้"
        )

    # ตรวจสอบประเภท Attack ที่อนุญาต
    actual_attacks = set(
        clean_df.loc[
            clean_df["binary_label"].eq("Attack"),
            "attack_type",
        ].unique()
    )

    unexpected_attacks = (
        actual_attacks - ALLOWED_ATTACKS
    )

    if unexpected_attacks:
        raise RuntimeError(
            "พบ Attack ที่ไม่ได้อยู่ในรายการที่กำหนด:\n"
            + ", ".join(sorted(unexpected_attacks))
        )

    # Monday ต้องเป็น Normal ทั้งหมด
    monday_df = clean_df.loc[
        clean_df["capture_date_adt"].eq(str(MONDAY))
    ]

    if not monday_df["binary_label"].eq("Normal").all():
        raise RuntimeError(
            "พบข้อมูลที่ไม่ใช่ Normal ใน Monday"
        )

    # ตรวจสอบว่าไม่มีประเภทที่ต้องตัดออกใน Included
    included_excluded = (
        clean_df["attack_type"].isin(EXCLUDED_NAMES)
    )

    if included_excluded.any():
        raise RuntimeError(
            "พบประเภทที่ต้องตัดออกอยู่ในข้อมูล Included"
        )

    # Included ต้องมี Binary Label ครบ
    if clean_df["binary_label"].isna().any():
        raise RuntimeError(
            "พบ Binary Label ที่หายไปในข้อมูล Included"
        )

    # ตรวจสอบระยะเวลาของ Window อีกครั้ง
    final_durations = (
        end_epoch.loc[clean_df.index]
        - start_epoch.loc[clean_df.index]
    )

    if not np.isclose(
        final_durations.to_numpy(),
        WINDOW_SECONDS,
        rtol=0,
        atol=WINDOW_TOLERANCE,
    ).all():
        raise RuntimeError(
            "พบ Window ที่ไม่ครบ 10 วินาทีในผลลัพธ์"
        )

    # --------------------------------------------------------
    # BUILD AUDIT AND SUMMARY
    # --------------------------------------------------------

    summary = (
        df.groupby(
            [
                "capture_date_adt",
                "label_status",
                "Label",
                "binary_label",
                "traffic_type",
                "attack_type",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="windows")
        .sort_values(
            [
                "capture_date_adt",
                "label_status",
                "traffic_type",
            ]
        )
    )

    # --------------------------------------------------------
    # SAVE FILES
    # --------------------------------------------------------

    save_csv(
        clean_df,
        OUTPUT_FILE,
    )

    save_csv(
        df,
        AUDIT_FILE,
    )

    save_csv(
        summary,
        SUMMARY_FILE,
    )

    # --------------------------------------------------------
    # PRINT REPORT
    # --------------------------------------------------------

    print("\n" + "=" * 78)
    print("LABELING SUMMARY")
    print("=" * 78)

    for day in sorted(EXPECTED_DATES):
        day_df = clean_df.loc[
            clean_df["capture_date_adt"].eq(str(day))
        ]

        print(f"\n{day}:")
        print(f"  Included windows: {len(day_df):,}")
        print(
            day_df["traffic_type"]
            .value_counts(dropna=False)
            .to_string()
        )

    print("\nSelected attack matches:")

    if selected_counts:
        for name, count in sorted(selected_counts.items()):
            print(f"  {name}: {count:,}")
    else:
        print("  None")

    print("\nExcluded attack matches:")

    if excluded_counts:
        for name, count in sorted(excluded_counts.items()):
            print(f"  {name}: {count:,}")
    else:
        print("  None")

    print("\nFinal counts:")
    print(f"  Input windows : {len(df):,}")
    print(f"  Included      : {len(clean_df):,}")
    print(
        f"  Excluded      : "
        f"{int(df['label_status'].eq('EXCLUDED').sum()):,}"
    )
    print(
        f"  Normal        : "
        f"{int(clean_df['binary_label'].eq('Normal').sum()):,}"
    )
    print(
        f"  Attack        : "
        f"{int(clean_df['binary_label'].eq('Attack').sum()):,}"
    )

    print("\nValidation:")
    print("  Complete 10-second windows: PASS")
    print("  Monday Normal only        : PASS")
    print("  Allowed Attack types only : PASS")
    print("  Excluded types removed    : PASS")
    print("  Included labels complete  : PASS")
    print("  PCAP read                 : NO")
    print("  Packet scan               : NO")

    print("\nSaved files:")
    print(f"  Labeled : {OUTPUT_FILE}")
    print(f"  Audit   : {AUDIT_FILE}")
    print(f"  Summary : {SUMMARY_FILE}")
    print("=" * 78)


if __name__ == "__main__":
    main()