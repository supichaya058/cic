from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
import csv
import os
import sys

from scapy.utils import RawPcapReader, RawPcapNgReader


# ============================================================
# STAGE 003 — CIC-IDS2017 CREATE TIME WINDOWS
# ============================================================
# Stage 003 has ONE responsibility:
#   Create complete 10-second temporal Windows.
#
# It MUST NOT create Behavioral Features.
# Feature extraction belongs to Stage 005.
#
# INPUT:
#   data/cleaned_pcaps/
#     Monday-WorkingHours_clean.pcapng
#     Wednesday-workingHours_clean.pcapng
#     Friday-WorkingHours_clean.pcapng
#
# OUTPUT:
#   data/time_windows/cic_time_windows_10s.csv
#   data/time_windows/cic_time_windows_10s_summary.csv
#
# Window CSV contains ONLY time-window metadata:
#   source_file
#   window_id
#   window_start_offset_sec
#   window_end_offset_sec
#   window_start_epoch
#   window_end_epoch
#   window_start_utc
#   window_end_utc
#
# No packet/behavior features are written here.
# ============================================================


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "data" / "cleaned_pcaps"
OUTPUT_DIR = BASE_DIR / "data" / "time_windows"

OUTPUT_FILE = OUTPUT_DIR / "cic_time_windows_10s.csv"
SUMMARY_FILE = OUTPUT_DIR / "cic_time_windows_10s_summary.csv"

TEMP_FILE = OUTPUT_DIR / "cic_time_windows_10s.tmp"
TEMP_SUMMARY = OUTPUT_DIR / "cic_time_windows_10s_summary.tmp"

WINDOW_SECONDS = Decimal("10")
PROGRESS_EVERY = 1_000_000

PCAP_FILES = [
    ("Monday-WorkingHours_clean.pcapng", "Monday-WorkingHours.pcap"),
    ("Wednesday-workingHours_clean.pcapng", "Wednesday-workingHours.pcap"),
    ("Friday-WorkingHours_clean.pcapng", "Friday-WorkingHours.pcap"),
]

WINDOW_FIELDS = [
    "source_file",
    "window_id",
    "window_start_offset_sec",
    "window_end_offset_sec",
    "window_start_epoch",
    "window_end_epoch",
    "window_start_utc",
    "window_end_utc",
]

SUMMARY_FIELDS = [
    "source_file",
    "input_file",
    "capture_type",
    "packets_read",
    "capture_start_utc",
    "capture_end_utc",
    "duration_seconds",
    "timestamp_regressions",
    "max_backward_seconds",
    "complete_windows",
    "status",
]

PCAPNG_MAGIC = b"\x0a\x0d\x0d\x0a"
PCAP_MAGIC = {
    b"\xd4\xc3\xb2\xa1",
    b"\xa1\xb2\xc3\xd4",
    b"\x4d\x3c\xb2\xa1",
    b"\xa1\xb2\x3c\x4d",
}


def detect_capture_type(path: Path) -> str:
    with path.open("rb") as f:
        magic = f.read(4)

    if magic == PCAPNG_MAGIC:
        return "PCAPNG"
    if magic in PCAP_MAGIC:
        return "PCAP"

    raise ValueError(
        f"Unsupported capture format: {path.name} "
        f"(magic={magic.hex()})"
    )


def open_reader(path: Path, capture_type: str):
    if capture_type == "PCAPNG":
        return RawPcapNgReader(str(path))
    if capture_type == "PCAP":
        return RawPcapReader(str(path))
    raise ValueError(f"Unsupported capture type: {capture_type}")


def get_timestamp(meta, capture_type: str, reader) -> Decimal:
    try:
        if capture_type == "PCAP":
            sec = getattr(meta, "sec", None)
            fraction = getattr(meta, "usec", None)

            if sec is None or fraction is None:
                raise ValueError("PCAP packet has no timestamp")

            sec = int(sec)
            fraction = int(fraction)

            scale = (
                Decimal("1000000000")
                if bool(getattr(reader, "nano", False))
                else Decimal("1000000")
            )

            if sec < 0 or fraction < 0 or Decimal(fraction) >= scale:
                raise ValueError("Invalid PCAP timestamp")

            timestamp = Decimal(sec) + Decimal(fraction) / scale

        elif capture_type == "PCAPNG":
            high = getattr(meta, "tshigh", None)
            low = getattr(meta, "tslow", None)
            resolution = getattr(meta, "tsresol", None)

            if high is None or low is None or resolution is None:
                raise ValueError("PCAPNG packet has no timestamp")

            high = int(high)
            low = int(low)
            resolution = Decimal(str(resolution))

            if not (0 <= high <= 0xFFFFFFFF):
                raise ValueError("Invalid PCAPNG timestamp high")
            if not (0 <= low <= 0xFFFFFFFF):
                raise ValueError("Invalid PCAPNG timestamp low")
            if not resolution.is_finite() or resolution <= 0:
                raise ValueError("Invalid PCAPNG timestamp resolution")

            timestamp = Decimal((high << 32) | low) / resolution

        else:
            raise ValueError(f"Unsupported capture type: {capture_type}")

        if not timestamp.is_finite() or timestamp < 0:
            raise ValueError("Invalid packet timestamp")

        return timestamp

    except (ValueError, TypeError, InvalidOperation, OverflowError) as exc:
        raise ValueError(f"Cannot read packet timestamp: {exc}") from exc


def utc_text(timestamp: Decimal) -> str:
    return datetime.fromtimestamp(
        float(timestamp),
        tz=timezone.utc,
    ).isoformat(timespec="microseconds")


def scan_capture(path: Path, capture_type: str) -> dict:
    packets_read = 0
    first_ts: Decimal | None = None
    last_ts: Decimal | None = None
    previous_ts: Decimal | None = None

    timestamp_regressions = 0
    max_backward = Decimal("0")

    print(f"\nScanning timestamps: {path.name}", flush=True)

    with open_reader(path, capture_type) as reader:
        for _packet_data, meta in reader:
            packets_read += 1

            ts = get_timestamp(meta, capture_type, reader)

            if first_ts is None:
                first_ts = ts
                last_ts = ts
            else:
                if previous_ts is not None and ts < previous_ts:
                    timestamp_regressions += 1
                    backward = previous_ts - ts
                    if backward > max_backward:
                        max_backward = backward

                if last_ts is None or ts > last_ts:
                    last_ts = ts

            previous_ts = ts

            if packets_read % PROGRESS_EVERY == 0:
                print(
                    f"  Packets scanned: {packets_read:,}",
                    flush=True,
                )

    if packets_read == 0 or first_ts is None or last_ts is None:
        raise ValueError(f"No packets found in {path.name}")

    duration = last_ts - first_ts
    complete_windows = int(duration // WINDOW_SECONDS)

    if complete_windows <= 0:
        raise ValueError(
            f"No complete 10-second Window in {path.name}"
        )

    return {
        "packets_read": packets_read,
        "first_ts": first_ts,
        "last_ts": last_ts,
        "duration": duration,
        "timestamp_regressions": timestamp_regressions,
        "max_backward": max_backward,
        "complete_windows": complete_windows,
    }


def make_window_row(
    source_file: str,
    window_index: int,
    capture_start: Decimal,
) -> dict:
    start = capture_start + Decimal(window_index) * WINDOW_SECONDS
    end = start + WINDOW_SECONDS

    return {
        "source_file": source_file,
        "window_id": window_index + 1,
        "window_start_offset_sec": window_index * 10,
        "window_end_offset_sec": (window_index + 1) * 10,
        "window_start_epoch": f"{start:.6f}",
        "window_end_epoch": f"{end:.6f}",
        "window_start_utc": utc_text(start),
        "window_end_utc": utc_text(end),
    }


def process_capture(
    path: Path,
    source_file: str,
    writer,
) -> dict:
    capture_type = detect_capture_type(path)

    scan = scan_capture(path, capture_type)

    for index in range(scan["complete_windows"]):
        writer.writerow(
            make_window_row(
                source_file=source_file,
                window_index=index,
                capture_start=scan["first_ts"],
            )
        )

    status = (
        "WARNING_TIMESTAMP_REGRESSION"
        if scan["timestamp_regressions"] > 0
        else "PASS"
    )

    result = {
        "source_file": source_file,
        "input_file": path.name,
        "capture_type": capture_type,
        "packets_read": scan["packets_read"],
        "capture_start_utc": utc_text(scan["first_ts"]),
        "capture_end_utc": utc_text(scan["last_ts"]),
        "duration_seconds": f"{scan['duration']:.9f}",
        "timestamp_regressions": scan["timestamp_regressions"],
        "max_backward_seconds": f"{scan['max_backward']:.9f}",
        "complete_windows": scan["complete_windows"],
        "status": status,
    }

    print(f"  Capture type          : {capture_type}")
    print(f"  Packets read          : {scan['packets_read']:,}")
    print(f"  Complete windows      : {scan['complete_windows']:,}")
    print(f"  Timestamp regressions : {scan['timestamp_regressions']:,}")
    print(f"  Status                : {status}")

    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    missing = [
        cleaned_name
        for cleaned_name, _source_name in PCAP_FILES
        if not (INPUT_DIR / cleaned_name).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing cleaned capture files:\n"
            + "\n".join(missing)
            + f"\n\nInput directory: {INPUT_DIR}"
        )

    summaries = []
    total_packets = 0
    total_windows = 0

    try:
        with (
            TEMP_FILE.open(
                "w",
                newline="",
                encoding="utf-8-sig",
            ) as output_file,
            TEMP_SUMMARY.open(
                "w",
                newline="",
                encoding="utf-8-sig",
            ) as summary_file,
        ):
            writer = csv.DictWriter(
                output_file,
                fieldnames=WINDOW_FIELDS,
                extrasaction="raise",
            )
            writer.writeheader()

            summary_writer = csv.DictWriter(
                summary_file,
                fieldnames=SUMMARY_FIELDS,
                extrasaction="raise",
            )
            summary_writer.writeheader()

            print("=" * 78)
            print("STAGE 003 — CIC-IDS2017 CREATE TIME WINDOWS")
            print("=" * 78)
            print(f"Input directory : {INPUT_DIR}")
            print(f"Output directory: {OUTPUT_DIR}")
            print("Window size     : 10 seconds")
            print("Feature creation: NO")
            print("Label assignment: NO")
            print("Data split      : NO")
            print("=" * 78)

            for i, (cleaned_name, source_name) in enumerate(
                PCAP_FILES, start=1
            ):
                print(f"\n[{i}/{len(PCAP_FILES)}]")

                result = process_capture(
                    path=INPUT_DIR / cleaned_name,
                    source_file=source_name,
                    writer=writer,
                )

                summary_writer.writerow(result)
                summaries.append(result)

                total_packets += result["packets_read"]
                total_windows += result["complete_windows"]

        os.replace(TEMP_FILE, OUTPUT_FILE)
        os.replace(TEMP_SUMMARY, SUMMARY_FILE)

    except Exception:
        for temp in (TEMP_FILE, TEMP_SUMMARY):
            if temp.exists():
                temp.unlink()
        raise

    print("\n" + "=" * 78)
    print("STAGE 003 COMPLETED")
    print("=" * 78)
    print(f"Files processed : {len(summaries)}")
    print(f"Total packets   : {total_packets:,}")
    print(f"Total windows   : {total_windows:,}")
    print(f"Window output   : {OUTPUT_FILE}")
    print(f"Summary output  : {SUMMARY_FILE}")
    print("=" * 78)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n" + "=" * 78)
        print("STAGE 003 FAILED")
        print("=" * 78)
        print(f"{type(exc).__name__}: {exc}")
        sys.exit(1)
