from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import csv
import math
import os
import shutil
import subprocess
import sys
from typing import Iterable

import numpy as np
import pandas as pd


# ============================================================
# STAGE 005 — CIC-IDS2017 BEHAVIORAL FEATURE EXTRACTION
# 10-SECOND WINDOWS / 13 BEHAVIORAL FEATURES
# ============================================================
#
# STAGE RESPONSIBILITIES
#   1) Read the labeled Time Windows from Stage 004.
#   2) Read the CLEANED CIC-IDS2017 PCAP/PCAPNG files from
#      data/cleaned_pcaps/.
#   3) Calculate the 13 Behavioral Features inside each
#      10-second Window.
#   4) Preserve the labels assigned by Stage 004.
#
# THIS STAGE DOES NOT
#   - create Time Windows
#   - assign labels
#   - split train/validation/test
#   - scale features
#   - perform feature selection
#
# 13 FEATURES
#   Volume
#     1. Request Rate
#     2. Total Requests
#     3. Throughput
#     4. Flow Count
#   Error
#     5. Error Rate
#     6. 4xx Count
#     7. 5xx Count
#   Latency
#     8. Avg Latency
#     9. Max Latency
#    10. Latency Std Dev
#   Source
#    11. Unique IP Count
#    12. New IP Ratio
#   Connection
#    13. Active Sessions
#
# FEATURE DEFINITIONS FOLLOW THE THESIS TABLE:
#   Request Rate      = Total HTTP Requests / 10 seconds
#   Total Requests    = HTTP Request count in Window
#   Throughput        = Total Frame Bytes / 10 seconds
#   Flow Count        = Number of unique Five-tuples in Window
#   Error Rate        = (4xx Count + 5xx Count) / Total Responses
#   4xx Count         = HTTP responses 400-499
#   5xx Count         = HTTP responses 500-599
#   Avg Latency       = Mean of usable HTTP timing values
#   Max Latency       = Maximum usable HTTP timing value
#   Latency Std Dev   = Population standard deviation of usable HTTP timing
#   Unique IP Count   = Number of unique source IP addresses
#   New IP Ratio      = New IP Count / Unique IP Count
#   Active Sessions   = Number of distinct TCP streams present in Window
#
# HTTP request/response/timing fields are obtained from TShark.
# Throughput / Flow / Source / Connection metrics are derived
# directly from packet-level fields.
#
# IMPORTANT:
#   Stage 003 creates only temporal Window metadata.
#   Therefore this Stage MUST NOT depend on packet_count,
#   total_bytes, flow_count_feature, unique_ip_count, etc.
#   from the Stage 003 CSV.
# ============================================================


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TIME_WINDOW_DIR = BASE_DIR / "data" / "time_windows"
CLEANED_PCAP_DIR = BASE_DIR / "data" / "cleaned_pcaps"

INPUT_FILE = TIME_WINDOW_DIR / "cic_labeled_time_windows_10s.csv"
OUTPUT_FILE = TIME_WINDOW_DIR / "cic_behavior_features_10s.csv"
FEATURE_LIST_FILE = TIME_WINDOW_DIR / "cic_behavior_feature_list_10s.csv"
AUDIT_FILE = TIME_WINDOW_DIR / "cic_behavior_features_10s_audit.csv"
SUMMARY_FILE = TIME_WINDOW_DIR / "cic_behavior_features_10s_summary.csv"

WINDOW_SECONDS = 10.0
WINDOW_TOLERANCE = 1e-5
TIMESTAMP_EPSILON = 1e-9
PROGRESS_EVERY = 1_000_000

# Stage 003 stores the original CIC capture name in source_file.
# Stage 005 maps it to the cleaned PCAP/PCAPNG file.
CLEANED_CAPTURE_MAP = {
    "Monday-WorkingHours.pcap": "Monday-WorkingHours_clean.pcapng",
    "Wednesday-workingHours.pcap": "Wednesday-workingHours_clean.pcapng",
    "Friday-WorkingHours.pcap": "Friday-WorkingHours_clean.pcapng",
}

# Fallback matching is kept only for filename capitalization differences.
CLEANED_CAPTURE_ALIASES = {
    "Monday-WorkingHours.pcap": (
        "Monday-WorkingHours_clean.pcapng",
        "Monday-WorkingHours_clean.pcap",
    ),
    "Wednesday-workingHours.pcap": (
        "Wednesday-workingHours_clean.pcapng",
        "Wednesday-workingHours_clean.pcap",
    ),
    "Friday-WorkingHours.pcap": (
        "Friday-WorkingHours_clean.pcapng",
        "Friday-WorkingHours_clean.pcap",
    ),
}

FEATURE_ORDER = [
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

FEATURE_DEFINITIONS = [
    {
        "group": "Volume",
        "feature": "request_rate",
        "definition": "Total HTTP Requests / 10 seconds",
        "unit": "requests/s",
    },
    {
        "group": "Volume",
        "feature": "total_requests",
        "definition": "Number of HTTP Requests in Window",
        "unit": "count",
    },
    {
        "group": "Volume",
        "feature": "throughput",
        "definition": "Total Frame Bytes / 10 seconds",
        "unit": "bytes/s",
    },
    {
        "group": "Volume",
        "feature": "flow_count",
        "definition": "Number of unique Five-tuples in Window",
        "unit": "count",
    },
    {
        "group": "Error",
        "feature": "error_rate",
        "definition": "(4xx Count + 5xx Count) / Total Responses",
        "unit": "ratio",
    },
    {
        "group": "Error",
        "feature": "4xx_count",
        "definition": "Number of HTTP responses with status 400-499",
        "unit": "count",
    },
    {
        "group": "Error",
        "feature": "5xx_count",
        "definition": "Number of HTTP responses with status 500-599",
        "unit": "count",
    },
    {
        "group": "Latency",
        "feature": "avg_latency",
        "definition": "Mean of usable HTTP timing values",
        "unit": "s",
    },
    {
        "group": "Latency",
        "feature": "max_latency",
        "definition": "Maximum usable HTTP timing value",
        "unit": "s",
    },
    {
        "group": "Latency",
        "feature": "latency_std_dev",
        "definition": "Population standard deviation of usable HTTP timing values",
        "unit": "s",
    },
    {
        "group": "Source",
        "feature": "unique_ip_count",
        "definition": "Number of unique source IP addresses",
        "unit": "count",
    },
    {
        "group": "Source",
        "feature": "new_ip_ratio",
        "definition": "New IP Count / Unique IP Count; 0 when denominator is 0",
        "unit": "ratio",
    },
    {
        "group": "Connection",
        "feature": "active_sessions",
        "definition": "Number of distinct TCP streams present in Window",
        "unit": "count",
    },
]

REQUIRED_STAGE004_COLUMNS = [
    "source_file",
    "window_id",
    "window_start_epoch",
    "window_end_epoch",
    "binary_label",
    "traffic_type",
    "attack_type",
    "label_status",
]

TSHARK_FIELDS = [
    "frame.time_epoch",
    "frame.len",
    "ip.src",
    "ipv6.src",
    "ip.dst",
    "ipv6.dst",
    "tcp.stream",
    "tcp.srcport",
    "tcp.dstport",
    "udp.srcport",
    "udp.dstport",
    "http.request.method",
    "http.response.code",
    "http.time",
]


# ============================================================
# 2. DATA STRUCTURES
# ============================================================

@dataclass
class OnlineStats:
    """Welford online statistics for latency values."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    maximum: float = 0.0

    def add(self, value: float) -> None:
        if not math.isfinite(value) or value < 0:
            return

        self.count += 1

        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

        if self.count == 1 or value > self.maximum:
            self.maximum = value

    @property
    def std_population(self) -> float:
        if self.count <= 0:
            return 0.0

        variance = max(self.m2 / self.count, 0.0)
        return math.sqrt(variance)


@dataclass
class WindowAggregate:
    total_bytes: int = 0
    total_requests: int = 0
    total_responses: int = 0
    count_4xx: int = 0
    count_5xx: int = 0
    latency: OnlineStats = field(default_factory=OnlineStats)
    flows: set[tuple] = field(default_factory=set)
    source_ips: set[str] = field(default_factory=set)
    tcp_streams: set[str] = field(default_factory=set)


# ============================================================
# 3. GENERAL UTILITIES
# ============================================================

def normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).strip()


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


def detect_tshark() -> str:
    env_path = os.environ.get("TSHARK_PATH", "").strip()
    if env_path:
        candidate = Path(env_path)
        if candidate.is_file():
            return str(candidate)
        raise FileNotFoundError(
            "TSHARK_PATH ถูกกำหนดไว้แต่ไม่พบไฟล์:\n"
            f"{candidate}"
        )

    found = shutil.which("tshark")
    if found:
        return found

    windows_candidates = [
        Path(r"C:\Program Files\Wireshark\tshark.exe"),
        Path(r"C:\Program Files (x86)\Wireshark\tshark.exe"),
        Path(r"D:\download\Wireshark\tshark.exe"),
    ]

    for candidate in windows_candidates:
        if candidate.is_file():
            return str(candidate)

    raise FileNotFoundError(
        "ไม่พบ tshark.exe ใน PATH และไม่พบในตำแหน่งติดตั้ง Wireshark มาตรฐาน\n"
        "สามารถกำหนดตำแหน่งเองได้ด้วย environment variable TSHARK_PATH"
    )


def resolve_cleaned_capture(source_file: str) -> Path:
    source_file = normalize_text(source_file)

    direct_name = CLEANED_CAPTURE_MAP.get(source_file)
    if direct_name:
        direct_path = CLEANED_PCAP_DIR / direct_name
        if direct_path.is_file():
            return direct_path

    aliases = CLEANED_CAPTURE_ALIASES.get(source_file, ())
    for alias in aliases:
        candidate = CLEANED_PCAP_DIR / alias
        if candidate.is_file():
            return candidate

    normalized_source = source_file.lower()
    files = list(CLEANED_PCAP_DIR.glob("*.pcap")) + list(
        CLEANED_PCAP_DIR.glob("*.pcapng")
    )

    for candidate in files:
        stem = candidate.stem.lower()
        cleaned_stem = stem.replace("_clean", "")
        source_stem = Path(source_file).stem.lower()
        if cleaned_stem == source_stem:
            return candidate
        if normalized_source in candidate.name.lower():
            return candidate

    raise FileNotFoundError(
        "ไม่พบ Cleaned PCAP/PCAPNG สำหรับ source_file:\n"
        f"{source_file}\n"
        f"Cleaned directory: {CLEANED_PCAP_DIR}"
    )


def read_stage004() -> pd.DataFrame:
    print("\n[1] LOAD STAGE 004")
    print("-" * 78)

    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            "ไม่พบไฟล์จาก Stage 004:\n"
            f"{INPUT_FILE}\n\n"
            "โปรดรัน 004_assign_labels.py ก่อน"
        )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
        encoding="utf-8-sig",
    )

    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    missing = [
        col for col in REQUIRED_STAGE004_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Stage 004 ขาดคอลัมน์ที่จำเป็น:\n"
            + "\n".join(missing)
        )

    if df.empty:
        raise ValueError("Stage 004 ไม่มีข้อมูล")

    print(f"Input file : {INPUT_FILE}")
    print(f"Input rows : {len(df):,}")
    print(f"Input cols : {len(df.columns):,}")

    return df


def validate_stage004(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[2] VALIDATE STAGE 004")
    print("-" * 78)

    result = df.copy()

    status = (
        result["label_status"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    if not status.eq("INCLUDED").all():
        counts = status.value_counts(dropna=False).to_dict()
        raise ValueError(
            "Stage 005 ต้องรับเฉพาะ INCLUDED windows\n"
            f"พบ label_status: {counts}"
        )

    window_id = pd.to_numeric(result["window_id"], errors="coerce")
    start_epoch = pd.to_numeric(
        result["window_start_epoch"],
        errors="coerce",
    )
    end_epoch = pd.to_numeric(
        result["window_end_epoch"],
        errors="coerce",
    )

    if window_id.isna().any():
        raise ValueError("window_id มีค่าที่ไม่ใช่ตัวเลข")
    if start_epoch.isna().any() or end_epoch.isna().any():
        raise ValueError("Window epoch มีค่าที่ไม่ใช่ตัวเลข")

    duration = end_epoch - start_epoch
    valid_duration = np.isclose(
        duration.to_numpy(dtype=float),
        WINDOW_SECONDS,
        rtol=0,
        atol=WINDOW_TOLERANCE,
    )
    if not valid_duration.all():
        bad = int((~valid_duration).sum())
        raise ValueError(
            f"พบ Window ที่ไม่ครบ {WINDOW_SECONDS:.0f} วินาที: {bad:,} แถว"
        )

    duplicate_mask = result.duplicated(
        ["source_file", "window_id"],
        keep=False,
    )
    if duplicate_mask.any():
        raise ValueError(
            "พบ source_file + window_id ซ้ำใน Stage 004"
        )

    binary = (
        result["binary_label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    allowed_binary = {"normal", "attack"}
    unknown_binary = set(binary.unique()) - allowed_binary
    if unknown_binary:
        raise ValueError(
            f"พบ binary_label ที่ไม่รองรับ: {sorted(unknown_binary)}"
        )

    result["window_id"] = window_id.astype("int64")
    result["window_start_epoch"] = start_epoch.astype(float)
    result["window_end_epoch"] = end_epoch.astype(float)
    result["binary_label"] = binary.map(
        {"normal": "Normal", "attack": "Attack"}
    )
    result["source_file"] = result["source_file"].astype(str).str.strip()

    print("[PASS] Stage 004 file exists")
    print("[PASS] Required columns")
    print("[PASS] INCLUDED windows only")
    print("[PASS] Complete 10-second windows")
    print("[PASS] Unique source_file + window_id")
    print("[PASS] Valid Binary Label")

    print("\nInput distribution:")
    print(
        result["binary_label"]
        .value_counts()
        .to_string()
    )

    return result


def load_window_index(df: pd.DataFrame) -> dict[str, dict]:
    """
    Load the COMPLETE Stage 003 window table and build the window index.

    Important:
      Stage 004 intentionally removes excluded windows. Therefore Stage 004
      cannot be used to validate Window continuity. The authoritative window
      sequence comes from Stage 003 (all complete 10-second windows).

    Stage 005 then extracts packet/HTTP features for the full Stage 003 window
    sequence and keeps only the windows present in Stage 004.
    """

    print("\n[3] BUILD STAGE 003 WINDOW INDEX")
    print("-" * 78)

    stage003_file = TIME_WINDOW_DIR / "cic_time_windows_10s.csv"

    if not stage003_file.is_file():
        raise FileNotFoundError(
            "ไม่พบไฟล์ Time Window จาก Stage 003:\n"
            f"{stage003_file}\n"
            "โปรดรัน 003_create_time_windows.py ก่อน"
        )

    stage003 = pd.read_csv(
        stage003_file,
        low_memory=False,
        encoding="utf-8-sig",
    )

    stage003.columns = (
        stage003.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    required = {
        "source_file",
        "window_id",
        "window_start_epoch",
        "window_end_epoch",
    }

    missing = sorted(required - set(stage003.columns))
    if missing:
        raise ValueError(
            "ไฟล์ Stage 003 ขาดคอลัมน์ที่จำเป็น: "
            + ", ".join(missing)
        )

    if stage003.empty:
        raise ValueError("ไฟล์ Stage 003 ไม่มี Window")

    index: dict[str, dict] = {}

    for source_file, group in stage003.groupby("source_file", sort=False):
        group = group.sort_values("window_id", kind="mergesort").reset_index(drop=True)

        window_ids = pd.to_numeric(
            group["window_id"], errors="coerce"
        )
        starts = pd.to_numeric(
            group["window_start_epoch"], errors="coerce"
        )
        ends = pd.to_numeric(
            group["window_end_epoch"], errors="coerce"
        )

        if window_ids.isna().any() or starts.isna().any() or ends.isna().any():
            raise ValueError(
                f"Stage 003 มี Window metadata ไม่ถูกต้องใน {source_file}"
            )

        ids = window_ids.astype(int).to_numpy()
        start_values = starts.to_numpy(dtype=float)
        end_values = ends.to_numpy(dtype=float)

        if ids[0] != 1:
            raise ValueError(
                f"Stage 003 ของ {source_file} ไม่มี Window 1"
            )

        expected_ids = np.arange(1, len(ids) + 1, dtype=int)
        if not np.array_equal(ids, expected_ids):
            raise ValueError(
                f"Window ID ของ {source_file} ใน Stage 003 ไม่ต่อเนื่อง"
            )

        durations = end_values - start_values
        if not np.allclose(
            durations,
            WINDOW_SECONDS,
            rtol=0,
            atol=WINDOW_TOLERANCE,
        ):
            raise ValueError(
                f"พบ Window ที่ไม่ครบ 10 วินาทีใน Stage 003: {source_file}"
            )

        expected_starts = (
            start_values[0]
            + np.arange(len(start_values), dtype=float) * WINDOW_SECONDS
        )

        if not np.allclose(
            start_values,
            expected_starts,
            rtol=0,
            atol=WINDOW_TOLERANCE,
        ):
            raise ValueError(
                f"Window start ของ {source_file} ใน Stage 003 ไม่ต่อเนื่อง"
            )

        index[source_file] = {
            "capture_start_epoch": float(start_values[0]),
            "max_window_id": int(ids[-1]),
            "window_ids": set(ids.tolist()),
        }

        print(
            f"{source_file}: "
            f"start={start_values[0]:.6f}, "
            f"windows={int(ids[-1]):,}, "
            "continuity=PASS"
        )

    # Confirm that every source_file needed by Stage 004 exists in Stage 003.
    missing_sources = sorted(
        set(df["source_file"].astype(str).str.strip()) - set(index)
    )
    if missing_sources:
        raise ValueError(
            "Stage 004 มี source_file ที่ไม่พบใน Stage 003: "
            + ", ".join(missing_sources)
        )

    print(
        f"Stage 003 total windows : {len(stage003):,}"
    )
    print(
        f"Stage 004 included rows : {len(df):,}"
    )
    print(
        "[PASS] Window continuity is validated from Stage 003, "
        "before Stage 004 exclusions"
    )

    return index


def make_aggregates(max_window_id: int) -> dict[int, WindowAggregate]:
    return {
        window_id: WindowAggregate()
        for window_id in range(1, max_window_id + 1)
    }


# ============================================================
# 4. TSHARK STREAMING EXTRACTION
# ============================================================

def build_tshark_command(
    tshark_path: str,
    capture_path: Path,
) -> list[str]:
    command = [
        tshark_path,
        "-r",
        str(capture_path),
        "-n",
        "-2",
        "-T",
        "fields",
        "-E",
        "separator=\t",
        "-E",
        "quote=n",
        "-E",
        "occurrence=f",
        "-o",
        "tcp.desegment_tcp_streams:TRUE",
        "-o",
        "http.desegment_headers:TRUE",
    ]

    for field_name in TSHARK_FIELDS:
        command.extend(["-e", field_name])

    return command


def split_tsv_line(line: str, expected_fields: int) -> list[str]:
    parts = line.rstrip("\r\n").split("\t")
    if len(parts) < expected_fields:
        parts.extend([""] * (expected_fields - len(parts)))
    elif len(parts) > expected_fields:
        parts = parts[:expected_fields]
    return parts


def parse_float(value: str) -> float | None:
    value = normalize_text(value)
    if not value:
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    if not math.isfinite(result):
        return None
    return result


def parse_int(value: str) -> int | None:
    value = normalize_text(value)
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return int(float(value))
        except ValueError:
            return None


def first_nonempty(*values: str) -> str:
    for value in values:
        value = normalize_text(value)
        if value:
            return value
    return ""


def extract_capture_features(
    tshark_path: str,
    capture_path: Path,
    source_file: str,
    window_meta: dict,
) -> tuple[dict[int, WindowAggregate], dict]:
    print("\n" + "=" * 78)
    print(f"[4] EXTRACT PACKET/HTTP DATA — {source_file}")
    print("=" * 78)
    print(f"Cleaned PCAP : {capture_path}")
    print(f"TShark       : {tshark_path}")
    print(f"Window       : {WINDOW_SECONDS:.0f} seconds")
    print("Features     : 13")
    print()

    max_window_id = int(window_meta["max_window_id"])
    capture_start = float(window_meta["capture_start_epoch"])
    valid_window_ids = window_meta["window_ids"]

    aggregates = make_aggregates(max_window_id)

    command = build_tshark_command(
        tshark_path=tshark_path,
        capture_path=capture_path,
    )

    print("Running TShark...")

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    packet_rows = 0
    assigned_rows = 0
    outside_rows = 0
    malformed_rows = 0

    try:
        assert process.stdout is not None

        for raw_line in process.stdout:
            packet_rows += 1
            fields = split_tsv_line(raw_line, len(TSHARK_FIELDS))

            ts = parse_float(fields[0])
            frame_len = parse_int(fields[1])

            if ts is None:
                malformed_rows += 1
                continue

            if frame_len is None or frame_len < 0:
                frame_len = 0

            delta = ts - capture_start
            if delta < -WINDOW_TOLERANCE:
                outside_rows += 1
                continue

            window_id = int(
                math.floor(
                    delta / WINDOW_SECONDS + TIMESTAMP_EPSILON
                )
                + 1
            )

            if window_id not in valid_window_ids:
                outside_rows += 1
                continue

            assigned_rows += 1
            aggregate = aggregates[window_id]
            aggregate.total_bytes += frame_len

            src_ip = first_nonempty(fields[2], fields[3])
            dst_ip = first_nonempty(fields[4], fields[5])

            if src_ip:
                aggregate.source_ips.add(src_ip)

            tcp_stream = normalize_text(fields[6])
            if tcp_stream:
                aggregate.tcp_streams.add(tcp_stream)

            tcp_src = normalize_text(fields[7])
            tcp_dst = normalize_text(fields[8])
            udp_src = normalize_text(fields[9])
            udp_dst = normalize_text(fields[10])

            if src_ip and dst_ip:
                if tcp_stream or tcp_src or tcp_dst:
                    protocol = "TCP"
                    src_port = tcp_src
                    dst_port = tcp_dst
                elif udp_src or udp_dst:
                    protocol = "UDP"
                    src_port = udp_src
                    dst_port = udp_dst
                else:
                    protocol = "IP"
                    src_port = ""
                    dst_port = ""

                aggregate.flows.add(
                    (
                        src_ip,
                        dst_ip,
                        src_port,
                        dst_port,
                        protocol,
                    )
                )

            http_method = normalize_text(fields[11])
            if http_method:
                aggregate.total_requests += 1

            response_code = parse_int(fields[12])
            if response_code is not None and 100 <= response_code <= 599:
                aggregate.total_responses += 1

                if 400 <= response_code <= 499:
                    aggregate.count_4xx += 1
                elif 500 <= response_code <= 599:
                    aggregate.count_5xx += 1

            http_time = parse_float(fields[13])
            if http_time is not None and http_time >= 0:
                aggregate.latency.add(http_time)

            if packet_rows % PROGRESS_EVERY == 0:
                print(
                    f"  Packets/frames processed: {packet_rows:,}",
                    flush=True,
                )

        stderr_text = ""
        if process.stderr is not None:
            stderr_text = process.stderr.read()

        return_code = process.wait()

    except Exception:
        process.kill()
        process.wait()
        raise

    if return_code != 0:
        message = stderr_text.strip()
        raise RuntimeError(
            "TShark failed with exit code "
            f"{return_code}\n"
            + (message if message else "No TShark error message was returned.")
        )

    print(f"  Frames read          : {packet_rows:,}")
    print(f"  Assigned to Windows  : {assigned_rows:,}")
    print(f"  Outside Window range : {outside_rows:,}")
    print(f"  Malformed rows       : {malformed_rows:,}")

    capture_report = {
        "frames_read": packet_rows,
        "assigned_frames": assigned_rows,
        "outside_frames": outside_rows,
        "malformed_rows": malformed_rows,
    }

    return aggregates, capture_report


# ============================================================
# 5. CALCULATE THE 13 FEATURES
# ============================================================

def calculate_feature_rows(
    stage004: pd.DataFrame,
    window_index: dict[str, dict],
    aggregate_by_source: dict[str, dict[int, WindowAggregate]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n[5] CALCULATE 13 BEHAVIORAL FEATURES")
    print("-" * 78)

    result = stage004.copy()

    # New IP Ratio is defined relative to previous Windows in the same
    # capture/run, so source IP sets must be processed in Window order.
    new_ip_ratio_by_key: dict[tuple[str, int], float] = {}
    new_ip_count_by_key: dict[tuple[str, int], int] = {}
    latency_samples_by_key: dict[tuple[str, int], int] = {}
    response_count_by_key: dict[tuple[str, int], int] = {}
    source_ip_count_by_key: dict[tuple[str, int], int] = {}

    for source_file, meta in window_index.items():
        aggregates = aggregate_by_source[source_file]
        seen_source_ips: set[str] = set()

        for window_id in range(1, int(meta["max_window_id"]) + 1):
            agg = aggregates[window_id]

            unique_ips = agg.source_ips
            new_ips = unique_ips - seen_source_ips

            unique_count = len(unique_ips)
            new_count = len(new_ips)

            ratio = (
                new_count / unique_count
                if unique_count > 0
                else 0.0
            )

            key = (source_file, window_id)
            new_ip_count_by_key[key] = new_count
            new_ip_ratio_by_key[key] = ratio
            latency_samples_by_key[key] = agg.latency.count
            response_count_by_key[key] = agg.total_responses
            source_ip_count_by_key[key] = unique_count

            seen_source_ips.update(unique_ips)

    rows = []

    for _, base in result.iterrows():
        source_file = normalize_text(base["source_file"])
        window_id = int(base["window_id"])
        key = (source_file, window_id)

        agg = aggregate_by_source[source_file][window_id]

        total_responses = agg.total_responses
        error_rate = (
            (agg.count_4xx + agg.count_5xx) / total_responses
            if total_responses > 0
            else 0.0
        )

        # The thesis defines latency from usable HTTP timing values.
        # When a Window has no usable HTTP timing value, the three latency
        # features are set to 0 and latency_sample_count records that no
        # timing observation was available. This is NOT treated as an
        # observed 0-second response time.
        if agg.latency.count > 0:
            avg_latency = agg.latency.mean
            max_latency = agg.latency.maximum
            latency_std = agg.latency.std_population
        else:
            avg_latency = 0.0
            max_latency = 0.0
            latency_std = 0.0

        feature_values = {
            "request_rate": agg.total_requests / WINDOW_SECONDS,
            "total_requests": agg.total_requests,
            "throughput": agg.total_bytes / WINDOW_SECONDS,
            "flow_count": len(agg.flows),
            "error_rate": error_rate,
            "4xx_count": agg.count_4xx,
            "5xx_count": agg.count_5xx,
            "avg_latency": avg_latency,
            "max_latency": max_latency,
            "latency_std_dev": latency_std,
            "unique_ip_count": len(agg.source_ips),
            "new_ip_ratio": new_ip_ratio_by_key[key],
            "active_sessions": len(agg.tcp_streams),
        }

        row = base.to_dict()
        row.update(feature_values)

        # Audit-only helper columns are prefixed so they are clearly not
        # among the 13 model features.
        row["_total_responses"] = total_responses
        row["_latency_samples"] = latency_samples_by_key[key]
        row["_new_ip_count"] = new_ip_count_by_key[key]
        row["_capture_unique_ip_count"] = source_ip_count_by_key[key]

        rows.append(row)

    feature_df = pd.DataFrame(rows)

    audit_df = (
        feature_df[
            [
                "source_file",
                "window_id",
                "binary_label",
                "traffic_type",
                "attack_type",
                "_total_responses",
                "_latency_samples",
                "_new_ip_count",
                "_capture_unique_ip_count",
                *FEATURE_ORDER,
            ]
        ]
        .rename(
            columns={
                "_total_responses": "total_responses",
                "_latency_samples": "latency_samples",
                "_new_ip_count": "new_ip_count",
                "_capture_unique_ip_count": "unique_ip_count_audit",
            }
        )
    )

    print("[PASS] Request Rate")
    print("[PASS] Total Requests")
    print("[PASS] Throughput")
    print("[PASS] Flow Count")
    print("[PASS] Error Rate")
    print("[PASS] 4xx Count")
    print("[PASS] 5xx Count")
    print("[PASS] Avg Latency")
    print("[PASS] Max Latency")
    print("[PASS] Latency Std Dev")
    print("[PASS] Unique IP Count")
    print("[PASS] New IP Ratio")
    print("[PASS] Active Sessions")

    return feature_df, audit_df


# ============================================================
# 6. VALIDATION
# ============================================================

def validate_features(df: pd.DataFrame) -> None:
    print("\n[6] VALIDATE GENERATED FEATURES")
    print("-" * 78)

    missing = [feature for feature in FEATURE_ORDER if feature not in df.columns]
    if missing:
        raise ValueError(
            "Generated dataset ขาด Feature:\n"
            + "\n".join(missing)
        )

    for feature in FEATURE_ORDER:
        values = pd.to_numeric(df[feature], errors="coerce")
        if values.isna().any():
            raise ValueError(
                f"Feature {feature} มี Missing Value"
            )
        values_array = values.to_numpy(dtype=float)
        if not np.isfinite(values_array).all():
            raise ValueError(
                f"Feature {feature} มี NaN/Infinite"
            )
        if (values_array < 0).any():
            raise ValueError(
                f"Feature {feature} มีค่าติดลบ"
            )

    error_rate = df["error_rate"].to_numpy(dtype=float)
    new_ip_ratio = df["new_ip_ratio"].to_numpy(dtype=float)

    if (error_rate > 1.0 + 1e-12).any():
        raise ValueError("Error Rate ต้องอยู่ในช่วง 0 ถึง 1")
    if (new_ip_ratio > 1.0 + 1e-12).any():
        raise ValueError("New IP Ratio ต้องอยู่ในช่วง 0 ถึง 1")

    # Every Stage 004 row must still exist.
    if len(df) != len(stage004_reference):
        raise ValueError(
            "จำนวนแถว Stage 005 ไม่เท่ากับ Stage 004"
        )

    expected_keys = set(
        zip(
            stage004_reference["source_file"].astype(str),
            stage004_reference["window_id"].astype(int),
        )
    )
    actual_keys = set(
        zip(
            df["source_file"].astype(str),
            df["window_id"].astype(int),
        )
    )

    if expected_keys != actual_keys:
        raise ValueError(
            "source_file + window_id ของ Stage 005 ไม่ตรงกับ Stage 004"
        )

    # Ensure labels were not changed by feature extraction.
    compare_columns = [
        "binary_label",
        "traffic_type",
        "attack_type",
        "label_status",
    ]
    for column in compare_columns:
        left = (
            stage004_reference[column]
            .astype(str)
            .fillna("")
            .tolist()
        )
        right_map = {
            (str(row["source_file"]), int(row["window_id"])): str(row[column])
            for _, row in df.iterrows()
        }
        right = [
            right_map[(str(row["source_file"]), int(row["window_id"]))]
            for _, row in stage004_reference.iterrows()
        ]
        if left != right:
            raise ValueError(
                f"Label preservation failed for {column}"
            )

    print(f"[PASS] 13 features present")
    print(f"[PASS] No missing values in 13 features")
    print(f"[PASS] No infinite values in 13 features")
    print(f"[PASS] Error Rate in [0,1]")
    print(f"[PASS] New IP Ratio in [0,1]")
    print(f"[PASS] Row count preserved: {len(df):,}")
    print(f"[PASS] Window keys preserved")
    print(f"[PASS] Labels preserved")


def build_feature_list() -> pd.DataFrame:
    rows = []
    for order, definition in enumerate(FEATURE_DEFINITIONS, start=1):
        rows.append(
            {
                "stage": "005",
                "dataset": "CIC-IDS2017",
                "window_seconds": WINDOW_SECONDS,
                "feature_order": order,
                "group": definition["group"],
                "feature": definition["feature"],
                "definition": definition["definition"],
                "unit": definition["unit"],
            }
        )
    return pd.DataFrame(rows)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "source_file",
        "binary_label",
        "traffic_type",
        "attack_type",
    ]

    summary = (
        df.groupby(group_columns, dropna=False)
        .agg(
            windows=("window_id", "size"),
            total_requests=("total_requests", "sum"),
            total_responses=("_total_responses", "sum"),
            total_4xx=("4xx_count", "sum"),
            total_5xx=("5xx_count", "sum"),
            mean_request_rate=("request_rate", "mean"),
            mean_throughput=("throughput", "mean"),
            mean_flow_count=("flow_count", "mean"),
            mean_error_rate=("error_rate", "mean"),
            mean_avg_latency=("avg_latency", "mean"),
            max_latency=("max_latency", "max"),
            mean_unique_ip_count=("unique_ip_count", "mean"),
            mean_new_ip_ratio=("new_ip_ratio", "mean"),
            mean_active_sessions=("active_sessions", "mean"),
            windows_without_latency=(
                "_latency_samples",
                lambda values: int((values == 0).sum()),
            ),
        )
        .reset_index()
        .sort_values(group_columns, kind="mergesort")
        .reset_index(drop=True)
    )

    summary["feature_count"] = len(FEATURE_ORDER)
    summary["window_seconds"] = WINDOW_SECONDS

    return summary


# ============================================================
# 7. SAVE
# ============================================================

def save_outputs(
    feature_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    aggregate_reports: list[dict],
) -> None:
    print("\n[7] SAVE OUTPUTS")
    print("-" * 78)

    output_df = feature_df.sort_values(
        ["source_file", "window_start_epoch", "window_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    # The final model dataset contains exactly the Stage 004 metadata plus
    # the 13 behavioral features. Audit-only internals remain in separate
    # CSV files and are not part of the model input.
    model_columns = [
        column
        for column in output_df.columns
        if not column.startswith("_")
    ]

    model_df = output_df[model_columns].copy()

    # Keep the 13 features together at the end for readability.
    metadata_columns = [
        column
        for column in model_df.columns
        if column not in FEATURE_ORDER
    ]
    model_df = model_df[metadata_columns + FEATURE_ORDER]

    atomic_csv(model_df, OUTPUT_FILE)
    atomic_csv(build_feature_list(), FEATURE_LIST_FILE)

    audit_out = audit_df.sort_values(
        ["source_file", "window_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    atomic_csv(audit_out, AUDIT_FILE)

    # Add capture scan information to a separate audit summary before the
    # ordinary behavioral summary.
    summary_df = build_summary(output_df)
    atomic_csv(summary_df, SUMMARY_FILE)

    capture_report_df = pd.DataFrame(aggregate_reports)
    capture_report_file = TIME_WINDOW_DIR / "cic_stage005_capture_audit_10s.csv"
    atomic_csv(capture_report_df, capture_report_file)

    print(f"[SAVED] {OUTPUT_FILE}")
    print(f"[SAVED] {FEATURE_LIST_FILE}")
    print(f"[SAVED] {AUDIT_FILE}")
    print(f"[SAVED] {SUMMARY_FILE}")
    print(f"[SAVED] {capture_report_file}")


# ============================================================
# 8. MAIN
# ============================================================

stage004_reference: pd.DataFrame


def main() -> None:
    global stage004_reference

    print("=" * 78)
    print("STAGE 005 — CIC-IDS2017 BEHAVIORAL FEATURE EXTRACTION")
    print("10-SECOND WINDOWS / 13 FEATURES")
    print("=" * 78)
    print(f"Stage 004 input : {INPUT_FILE}")
    print(f"Cleaned PCAP   : {CLEANED_PCAP_DIR}")
    print(f"Output         : {OUTPUT_FILE}")
    print(f"Window         : {WINDOW_SECONDS:.0f} seconds")
    print("Feature count  : 13")
    print("Data split     : NO")
    print("Labeling       : NO")
    print("PCAP read      : YES — TShark")
    print("Window index   : Stage 003")
    print("=" * 78)

    stage004 = read_stage004()
    stage004 = validate_stage004(stage004)
    stage004_reference = stage004.copy()

    window_index = load_window_index(stage004)

    tshark_path = detect_tshark()
    print(f"\nTShark detected: {tshark_path}")

    aggregate_by_source: dict[str, dict[int, WindowAggregate]] = {}
    capture_reports: list[dict] = []

    source_files = list(window_index.keys())

    print("\n[4A] RESOLVE CLEANED CAPTURES")
    print("-" * 78)

    capture_paths: dict[str, Path] = {}
    for source_file in source_files:
        capture_path = resolve_cleaned_capture(source_file)
        capture_paths[source_file] = capture_path
        print(f"{source_file} -> {capture_path.name}")

    print("\n[4B] EXTRACT FROM CLEANED PCAP/PCAPNG")
    print("-" * 78)

    for source_file in source_files:
        capture_path = capture_paths[source_file]

        aggregates, capture_report = extract_capture_features(
            tshark_path=tshark_path,
            capture_path=capture_path,
            source_file=source_file,
            window_meta=window_index[source_file],
        )

        aggregate_by_source[source_file] = aggregates

        capture_report["source_file"] = source_file
        capture_report["cleaned_capture"] = capture_path.name
        capture_report["window_count"] = window_index[source_file]["max_window_id"]
        capture_reports.append(capture_report)

    feature_df, audit_df = calculate_feature_rows(
        stage004=stage004,
        window_index=window_index,
        aggregate_by_source=aggregate_by_source,
    )

    validate_features(feature_df)

    save_outputs(
        feature_df=feature_df,
        audit_df=audit_df,
        aggregate_reports=capture_reports,
    )

    print("\n" + "=" * 78)
    print("STAGE 005 COMPLETED")
    print("=" * 78)
    print(f"Stage 004 windows : {len(stage004):,}")
    print(f"Feature rows      : {len(feature_df):,}")
    print(f"Behavioral feats  : {len(FEATURE_ORDER):,}")

    print("\n13 Behavioral Features:")
    for number, feature in enumerate(FEATURE_ORDER, start=1):
        print(f"  {number:>2}. {feature}")

    print("\nDistribution:")
    print(
        feature_df["binary_label"]
        .value_counts()
        .to_string()
    )

    print("\nValidation:")
    print("  Stage 004 input       : PASS")
    print("  Cleaned PCAP/PCAPNG   : PASS")
    print("  Packet/HTTP extraction: PASS")
    print("  10-second alignment   : PASS")
    print("  13-feature creation   : PASS")
    print("  Label preservation    : PASS")
    print("  No labeling           : PASS")
    print("  No data splitting     : PASS")
    print("  No feature selection  : PASS")

    print("\nNext stage:")
    print("  006_split_behavior_dataset.py")
    print("=" * 78)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n" + "=" * 78)
        print("STAGE 005 FAILED")
        print("=" * 78)
        print(f"{type(exc).__name__}: {exc}")
        sys.exit(1)
