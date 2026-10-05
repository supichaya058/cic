
from pathlib import Path
from datetime import datetime, timezone
import csv
import math

from scapy.utils import PcapReader
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6

# ============================================================
# CIC-IDS2017 STAGE 01: CHECK PCAP DATA
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PCAP_DIR = BASE_DIR / "data" / "pcaps"
OUTPUT_DIR = BASE_DIR / "data" / "checks"

OUTPUT_FILE = OUTPUT_DIR / "01_pcap_check_summary.csv"

PCAP_FILES = [
    "Monday-WorkingHours.pcap",
    "Wednesday-workingHours.pcap",
    "Friday-WorkingHours.pcap",
]

SAMPLE_LIMIT = 10_000


def utc_text(timestamp):
    return datetime.fromtimestamp(
        timestamp,
        tz=timezone.utc,
    ).isoformat(timespec="microseconds")


def check_pcap(file_path):
    packet_count = 0
    ipv4_count = 0
    ipv6_count = 0
    tcp_count = 0
    udp_count = 0
    other_count = 0

    first_ts = None
    last_ts = None
    previous_ts = None
    timestamp_order_errors = 0

    with PcapReader(str(file_path)) as reader:
        for packet in reader:
            packet_count += 1
            timestamp = float(packet.time)

            if not math.isfinite(timestamp):
                raise ValueError(
                    f"Invalid timestamp at packet {packet_count}"
                )

            if first_ts is None:
                first_ts = timestamp

            if (
                previous_ts is not None
                and timestamp < previous_ts
            ):
                timestamp_order_errors += 1

            previous_ts = timestamp
            last_ts = timestamp

            if IP in packet:
                ipv4_count += 1
            if IPv6 in packet:
                ipv6_count += 1
            if TCP in packet:
                tcp_count += 1
            if UDP in packet:
                udp_count += 1

            if not (TCP in packet or UDP in packet):
                other_count += 1

            if packet_count >= SAMPLE_LIMIT:
                break

    if packet_count == 0:
        raise ValueError("No packets found")

    return {
        "file": file_path.name,
        "size_bytes": file_path.stat().st_size,
        "size_gb": round(
            file_path.stat().st_size / (1024**3), 4
        ),
        "sample_packets": packet_count,
        "first_sample_timestamp_utc": utc_text(first_ts),
        "last_sample_timestamp_utc": utc_text(last_ts),
        "ipv4_packets": ipv4_count,
        "ipv6_packets": ipv6_count,
        "tcp_packets": tcp_count,
        "udp_packets": udp_count,
        "other_packets": other_count,
        "timestamp_order_errors": timestamp_order_errors,
        "status": (
            "PASS"
            if timestamp_order_errors == 0
            else "WARNING"
        ),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CIC-IDS2017 STAGE 01: PCAP DATA CHECK")
    print("=" * 70)
    print(f"PCAP directory : {PCAP_DIR}")
    print(f"Sample limit   : {SAMPLE_LIMIT:,} packets/file")
    print()

    results = []
    has_error = False

    for index, filename in enumerate(PCAP_FILES, start=1):
        file_path = PCAP_DIR / filename

        print(f"[{index}/{len(PCAP_FILES)}] {filename}")

        if not file_path.is_file():
            print("  ERROR: File not found")
            results.append({
                "file": filename,
                "size_bytes": 0,
                "size_gb": 0,
                "sample_packets": 0,
                "first_sample_timestamp_utc": "",
                "last_sample_timestamp_utc": "",
                "ipv4_packets": 0,
                "ipv6_packets": 0,
                "tcp_packets": 0,
                "udp_packets": 0,
                "other_packets": 0,
                "timestamp_order_errors": 0,
                "status": "MISSING",
            })
            has_error = True
            continue

        try:
            result = check_pcap(file_path)
            results.append(result)

            print(f"  Size           : {result['size_gb']:.2f} GB")
            print(f"  Sample packets : {result['sample_packets']:,}")
            print(f"  IPv4           : {result['ipv4_packets']:,}")
            print(f"  IPv6           : {result['ipv6_packets']:,}")
            print(f"  TCP            : {result['tcp_packets']:,}")
            print(f"  UDP            : {result['udp_packets']:,}")
            print(
                f"  Timestamp order errors: "
                f"{result['timestamp_order_errors']}"
            )
            print(f"  Status         : {result['status']}")

            if result["status"] != "PASS":
                has_error = True

        except Exception as error:
            print(f"  ERROR: {error}")
            results.append({
                "file": filename,
                "size_bytes": (
                    file_path.stat().st_size
                    if file_path.exists() else 0
                ),
                "size_gb": 0,
                "sample_packets": 0,
                "first_sample_timestamp_utc": "",
                "last_sample_timestamp_utc": "",
                "ipv4_packets": 0,
                "ipv6_packets": 0,
                "tcp_packets": 0,
                "udp_packets": 0,
                "other_packets": 0,
                "timestamp_order_errors": 0,
                "status": "ERROR",
            })
            has_error = True

        print()

    if results:
        with OUTPUT_FILE.open(
            "w", newline="", encoding="utf-8-sig"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(results[0].keys()),
            )
            writer.writeheader()
            writer.writerows(results)

    print("=" * 70)
    print("STAGE 01 SUMMARY")
    print("=" * 70)
    print(f"Files checked : {len(results)}")
    print(f"Passed        : {sum(r['status'] == 'PASS' for r in results)}")
    print(f"Output        : {OUTPUT_FILE}")

    if has_error:
        raise SystemExit("Stage 01 completed with errors or warnings.")

    print("Stage 01 completed successfully.")


if __name__ == "__main__":
    main()