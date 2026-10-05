
from pathlib import Path
from decimal import Decimal, InvalidOperation
import csv
import os
import shutil

from scapy.utils import RawPcapNgReader, RawPcapNgWriter

# ============================================================
# CIC-IDS2017 STAGE 02: CLEAN DATA
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "data" / "pcaps"
OUTPUT_DIR = BASE_DIR / "data" / "cleaned_pcaps"

PCAP_FILES = [
    "Monday-WorkingHours.pcap",
    "Wednesday-workingHours.pcap",
    "Friday-WorkingHours.pcap",
]

PROGRESS_EVERY = 500_000
MIN_FREE_SPACE = 512 * 1024**2

SUMMARY_FILE = OUTPUT_DIR / "cic_clean_summary.csv"
TEMP_SUMMARY = OUTPUT_DIR / "cic_clean_summary.tmp"

SUMMARY_FIELDS = [
    "source_file",
    "output_file",
    "input_packets",
    "kept_packets",
    "removed_packets",
    "empty_packets",
    "invalid_timestamps",
    "invalid_lengths",
    "timestamp_regressions",
    "max_backward_seconds",
    "source_size_bytes",
    "output_size_bytes",
    "output_method",
    "status",
]


# ============================================================
# 1. TIMESTAMP VALIDATION
# ============================================================

def get_timestamp(meta):
    """แปลง Timestamp จาก PCAPNG เป็น Decimal วินาที"""

    high = getattr(meta, "tshigh", None)
    low = getattr(meta, "tslow", None)
    resolution = getattr(meta, "tsresol", None)

    if high is None or low is None or resolution is None:
        return None

    try:
        high = int(high)
        low = int(low)
        resolution = Decimal(str(resolution))

        if not (0 <= high <= 0xFFFFFFFF):
            return None

        if not (0 <= low <= 0xFFFFFFFF):
            return None

        if not resolution.is_finite() or resolution <= 0:
            return None

        ticks = (high << 32) | low
        timestamp = Decimal(ticks) / resolution

        if not timestamp.is_finite() or timestamp < 0:
            return None

        return timestamp

    except (ValueError, TypeError, InvalidOperation, OverflowError):
        return None


# ============================================================
# 2. PACKET VALIDATION
# ============================================================

def validate_packet(data, meta):
    """
    ตรวจสอบ Packet ที่ว่าง Timestamp และความยาว

    ไม่ตรวจว่าต้องมี IP/TCP/UDP ทุก Packet
    เพราะการไม่มีฟิลด์เหล่านี้อาจเป็นข้อมูลที่ถูกต้อง
    """

    if len(data) == 0:
        return "EMPTY_PACKET", None

    timestamp = get_timestamp(meta)

    if timestamp is None:
        return "INVALID_TIMESTAMP", None

    try:
        wirelen = int(meta.wirelen)
        linktype = int(meta.linktype)
    except (TypeError, ValueError):
        return "INVALID_LENGTH", timestamp

    if wirelen < len(data) or wirelen < 0:
        return "INVALID_LENGTH", timestamp

    if not (0 <= linktype <= 65535):
        return "INVALID_LENGTH", timestamp

    return None, timestamp


# ============================================================
# 3. SCAN PCAPNG
# ============================================================

def scan_pcap(file_path):
    print(f"\nScanning: {file_path.name}", flush=True)

    counts = {
        "input_packets": 0,
        "empty_packets": 0,
        "invalid_timestamps": 0,
        "invalid_lengths": 0,
        "timestamp_regressions": 0,
    }

    previous_timestamp = None
    max_backward = Decimal("0")
    linktypes = set()

    with RawPcapNgReader(str(file_path)) as reader:
        for data, meta in reader:
            counts["input_packets"] += 1

            reason, timestamp = validate_packet(data, meta)

            if reason == "EMPTY_PACKET":
                counts["empty_packets"] += 1
            elif reason == "INVALID_TIMESTAMP":
                counts["invalid_timestamps"] += 1
            elif reason == "INVALID_LENGTH":
                counts["invalid_lengths"] += 1

            if timestamp is not None:
                if (
                    previous_timestamp is not None
                    and timestamp < previous_timestamp
                ):
                    counts["timestamp_regressions"] += 1
                    backward = previous_timestamp - timestamp
                    max_backward = max(max_backward, backward)

                previous_timestamp = timestamp

            if reason is None:
                linktypes.add(int(meta.linktype))

            if counts["input_packets"] % PROGRESS_EVERY == 0:
                removed = (
                    counts["empty_packets"]
                    + counts["invalid_timestamps"]
                    + counts["invalid_lengths"]
                )
                print(
                    f"  Read: {counts['input_packets']:,}"
                    f" | Invalid packets: {removed:,}",
                    flush=True,
                )

    if counts["input_packets"] == 0:
        raise ValueError(f"No packets in {file_path.name}")

    counts["removed_packets"] = (
        counts["empty_packets"]
        + counts["invalid_timestamps"]
        + counts["invalid_lengths"]
    )

    counts["kept_packets"] = (
        counts["input_packets"] - counts["removed_packets"]
    )
    counts["max_backward_seconds"] = max_backward

    if counts["kept_packets"] == 0:
        raise ValueError(
            f"No valid packets remain in {file_path.name}"
        )

    return counts


# ============================================================
# 4. WRITE CLEAN PCAPNG
# ============================================================

def normalize_ifname(meta, linktype):
    """สร้างชื่อ Interface ที่ไม่ชนกันระหว่าง Link Type"""

    name = getattr(meta, "ifname", None)

    if isinstance(name, str):
        name = name.encode("utf-8", "backslashreplace")
    elif not isinstance(name, bytes):
        name = b""

    # แยก interface ตาม Link Type
    return name + b"_linktype_" + str(linktype).encode("ascii")


def write_clean_pcap(source, destination, expected_kept):
    """
    เขียนเฉพาะ Packet ที่ผ่านการตรวจสอบลง PCAPNG ใหม่
    คงข้อมูล Packet ดิบและ Timestamp ไว้
    """

    temp = destination.with_name(destination.name + ".tmp")

    if temp.exists():
        temp.unlink()

    reader = None
    writer = None
    written = 0

    try:
        reader = RawPcapNgReader(str(source))
        writer = RawPcapNgWriter(str(temp))

        for data, meta in reader:
            reason, timestamp = validate_packet(data, meta)

            if reason is not None:
                continue

            linktype = int(meta.linktype)
            wirelen = int(meta.wirelen)
            ifname = normalize_ifname(meta, linktype)

            comments = getattr(meta, "comments", None)
            if comments is not None and isinstance(comments, bytes):
                comments = [comments]
            elif comments is not None and not isinstance(
                comments, (list, tuple)
            ):
                comments = [str(comments).encode("utf-8")]

            writer._write_packet(
                data,
                linktype=linktype,
                sec=timestamp,
                usec=0,
                caplen=len(data),
                wirelen=wirelen,
                ifname=ifname,
                direction=getattr(meta, "direction", None),
                comments=comments,
            )

            written += 1

            if written % PROGRESS_EVERY == 0:
                print(f"  Cleaned: {written:,} packets", flush=True)

        if writer is not None:
            writer.close()
            writer = None

        reader.close()
        reader = None

        if written != expected_kept:
            raise RuntimeError(
                f"Packet count mismatch: expected {expected_kept:,}, "
                f"got {written:,}"
            )

        os.replace(temp, destination)

    except Exception:
        if writer is not None:
            writer.close()
        if reader is not None:
            reader.close()
        if temp.exists():
            temp.unlink()
        raise

    return written


# ============================================================
# 5. PROCESS ONE PCAP
# ============================================================

def clean_one_file(file_path):
    output_name = (
        file_path.stem + "_clean.pcapng"
    )
    output_path = OUTPUT_DIR / output_name

    print(f"\nCleaning: {file_path.name}", flush=True)

    # ตรวจสอบ PCAP ทั้งไฟล์
    counts = scan_pcap(file_path)

    print(f"  Input packets : {counts['input_packets']:,}")
    print(f"  Valid packets : {counts['kept_packets']:,}")
    print(f"  Empty         : {counts['empty_packets']:,}")
    print(
        f"  Invalid timestamps: "
        f"{counts['invalid_timestamps']:,}"
    )
    print(f"  Invalid lengths: {counts['invalid_lengths']:,}")
    print(
        f"  Timestamp regressions: "
        f"{counts['timestamp_regressions']:,}"
    )

    removed = counts["removed_packets"]

    if removed == 0:
        # ไม่มีข้อมูลที่ต้องตัดออก
        # รักษาต้นฉบับไว้ด้วย Hard Link หรือสำเนา
        temp = output_path.with_name(output_path.name + ".tmp")

        if temp.exists():
            temp.unlink()

        try:
            try:
                os.link(file_path, temp)
                method = "HARDLINK_UNCHANGED"
            except OSError:
                required = file_path.stat().st_size + MIN_FREE_SPACE
                free = shutil.disk_usage(OUTPUT_DIR).free

                if free < required:
                    raise OSError(
                        "Not enough disk space to copy the PCAP file"
                    )

                shutil.copy2(file_path, temp)
                method = "COPY_UNCHANGED"

            os.replace(temp, output_path)

        except Exception:
            if temp.exists():
                temp.unlink()
            raise

    else:
        # พบข้อมูลที่ไม่ผ่านเกณฑ์ จึงเขียนไฟล์ Clean ใหม่
        required = file_path.stat().st_size + MIN_FREE_SPACE
        free = shutil.disk_usage(OUTPUT_DIR).free

        if free < required:
            raise OSError(
                f"Insufficient disk space for {file_path.name}"
            )

        write_clean_pcap(
            file_path,
            output_path,
            counts["kept_packets"],
        )
        method = "FILTERED_PCAPNG"

    if output_path.stat().st_size == 0:
        raise RuntimeError("Cleaned file is empty")

    return {
        "source_file": file_path.name,
        "output_file": output_name,
        "input_packets": counts["input_packets"],
        "kept_packets": counts["kept_packets"],
        "removed_packets": counts["removed_packets"],
        "empty_packets": counts["empty_packets"],
        "invalid_timestamps": counts["invalid_timestamps"],
        "invalid_lengths": counts["invalid_lengths"],
        "timestamp_regressions": counts["timestamp_regressions"],
        "max_backward_seconds": (
            f"{float(counts['max_backward_seconds']):.9f}"
        ),
        "source_size_bytes": file_path.stat().st_size,
        "output_size_bytes": output_path.stat().st_size,
        "output_method": method,
        "status": "PASS",
    }


# ============================================================
# 6. MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    missing = [
        name for name in PCAP_FILES
        if not (INPUT_DIR / name).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing PCAP files:\n" + "\n".join(missing)
        )

    results = []

    print("=" * 70)
    print("CIC-IDS2017 STAGE 02: CLEAN DATA")
    print("=" * 70)
    print(f"Input : {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    for index, filename in enumerate(PCAP_FILES, start=1):
        print(f"\n[{index}/{len(PCAP_FILES)}]")

        result = clean_one_file(INPUT_DIR / filename)
        results.append(result)

        print(f"  Output method: {result['output_method']}")
        print(f"  Status       : {result['status']}")

    # สรุปผลรวม
    total_fields = [
        "input_packets",
        "kept_packets",
        "removed_packets",
        "empty_packets",
        "invalid_timestamps",
        "invalid_lengths",
        "timestamp_regressions",
        "source_size_bytes",
        "output_size_bytes",
    ]

    total = {
        "source_file": "TOTAL",
        "output_file": "-",
    }

    for field in total_fields:
        total[field] = sum(
            int(result[field]) for result in results
        )

    total["max_backward_seconds"] = ""
    total["output_method"] = "MULTIPLE"
    total["status"] = "PASS"

    results.append(total)

    # เขียน Summary โดยใช้ไฟล์ชั่วคราวก่อน
    try:
        with TEMP_SUMMARY.open(
            "w", newline="", encoding="utf-8-sig"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=SUMMARY_FIELDS,
            )
            writer.writeheader()
            writer.writerows(results)

        os.replace(TEMP_SUMMARY, SUMMARY_FILE)

    except Exception:
        if TEMP_SUMMARY.exists():
            TEMP_SUMMARY.unlink()
        raise

    print("\n" + "=" * 70)
    print("STAGE 02 CLEANING COMPLETED")
    print("=" * 70)
    print(f"Input packets : {total['input_packets']:,}")
    print(f"Kept packets  : {total['kept_packets']:,}")
    print(f"Removed       : {total['removed_packets']:,}")
    print(f"Output        : {OUTPUT_DIR}")
    print(f"Summary       : {SUMMARY_FILE}")


if __name__ == "__main__":
    main()