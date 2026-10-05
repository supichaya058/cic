from pathlib import Path
from huggingface_hub import hf_hub_download

# ============================================================
# CIC-IDS2017 PCAP DOWNLOAD
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

REPO_ID = "bencorn/CICIDS2017"

PCAP_FILES = [
    "pcaps/Monday-WorkingHours.pcap",
    "pcaps/Wednesday-workingHours.pcap",
    "pcaps/Friday-WorkingHours.pcap",
]


def main():
    print("=" * 70)
    print("CIC-IDS2017 PCAP DOWNLOAD")
    print("=" * 70)
    print(f"Repository : {REPO_ID}")
    print(f"Save to    : {DATA_DIR}")
    print(f"Total files: {len(PCAP_FILES)}")
    print("=" * 70)

    for index, filename in enumerate(PCAP_FILES, start=1):
        print(f"\n[{index}/{len(PCAP_FILES)}] Downloading")
        print(f"File: {filename}")

        try:
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=filename,
                local_dir=str(DATA_DIR),
            )

            file_size = Path(local_path).stat().st_size

            print("Download successful")
            print(f"Path: {local_path}")
            print(f"Size: {file_size / (1024**3):.2f} GB")

        except Exception as e:
            print(f"\nDownload failed: {filename}")
            print(f"Error: {e}")
            raise

    print("\n" + "=" * 70)
    print("ALL DOWNLOADS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()