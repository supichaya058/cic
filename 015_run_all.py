from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


# ============================================================
# STAGE 015 — RUN CIC PIPELINE (STAGE 006–014; STAGE 008 REMOVED)
# ============================================================
# รันตามลำดับ Stage 006, 007, 009 ถึง Stage 014
# ไม่เรียก Stage 001–004
#
# วิธีใช้:
#   uv run python 015_run_all.py
#
# เงื่อนไข:
#   วางไฟล์ 015_run_all.py และสคริปต์ Stage 005–014
#   ไว้ในโฟลเดอร์โปรเจกต์เดียวกัน
#
# หาก Stage ใดเกิดข้อผิดพลาด จะหยุดทันทีและแสดง Exit Code
# เพื่อป้องกันการรัน Stage ถัดไปบนผลลัพธ์ที่ไม่สมบูรณ์
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

STAGES: list[tuple[int, str]] = [
    (6, "006_split_behavior_dataset.py"),
    (7, "007_scale_behavior_features.py"),
    (9, "009_baseline_models.py"),
    (10, "010_tune_models.py"),
    (11, "011_statistical_threshold_baseline.py"),
    (12, "012_train_final_models.py"),
    (13, "013_evaluate_final_test.py"),
    (14, "014_visualize_final_test.py"),
]


def check_stage_files() -> list[Path]:
    """ตรวจสอบไฟล์ Stage 006–014 ให้ครบก่อนเริ่ม Pipeline"""
    missing: list[str] = []
    paths: list[Path] = []

    expected_stage_numbers = [6, 7, 9, 10, 11, 12, 13, 14]
    actual_stage_numbers = [stage_no for stage_no, _ in STAGES]

    if actual_stage_numbers != expected_stage_numbers:
        print(
            "[ERROR] STAGES configuration ต้องเป็น Stage 006, 007, 009–014 "
            "ตามลำดับเท่านั้น.",
            flush=True,
        )
        print(
            f"พบ: {actual_stage_numbers}",
            flush=True,
        )
        return []

    for stage_no, filename in STAGES:
        path = BASE_DIR / filename
        if not path.is_file():
            missing.append(f"Stage {stage_no:03d}: {filename}")
        else:
            paths.append(path)

    if missing:
        print("[ERROR] Missing required stage file(s):", flush=True)
        for item in missing:
            print(f"  - {item}", flush=True)
        print(
            "\nPlace all Stage 005–014 scripts in the same folder as "
            "015_run_all.py, then run again.",
            flush=True,
        )
        return []

    print("[PASS] All required Stage 006–014 scripts found.", flush=True)
    return paths


def run_pipeline() -> int:
    stage_paths = check_stage_files()
    if not stage_paths:
        return 2

    print("=" * 68, flush=True)
    print("STAGE 015 — CIC PIPELINE RUNNER", flush=True)
    print("Stages: 006, 007, 009–014 (Stage 008 removed; Stages 001–005 are skipped)", flush=True)
    print(f"Project directory: {BASE_DIR}", flush=True)
    print("=" * 68, flush=True)

    pipeline_start = time.perf_counter()

    for (stage_no, filename), script_path in zip(STAGES, stage_paths):
        print("\n" + "=" * 68, flush=True)
        print(f"[RUN] STAGE {stage_no:03d} — {filename}", flush=True)
        print("=" * 68, flush=True)

        stage_start = time.perf_counter()
        try:
            result = subprocess.run(
                [sys.executable, "-u", str(script_path)],
                cwd=BASE_DIR,
                check=False,
            )
        except KeyboardInterrupt:
            print(f"\n[STOP] Interrupted during Stage {stage_no:03d}.", flush=True)
            return 130
        except OSError as exc:
            print(f"[ERROR] Could not start Stage {stage_no:03d}: {exc}", flush=True)
            return 1

        elapsed = time.perf_counter() - stage_start
        print(
            f"\n[INFO] Stage {stage_no:03d} finished in {elapsed:.1f} seconds "
            f"(exit code: {result.returncode}).",
            flush=True,
        )

        if result.returncode != 0:
            print(
                f"[STOP] Pipeline stopped at Stage {stage_no:03d}. "
                "Fix the error before running the next stage.",
                flush=True,
            )
            return result.returncode or 1

        print(f"[PASS] Stage {stage_no:03d} completed.", flush=True)

    total_elapsed = time.perf_counter() - pipeline_start
    print("\n" + "=" * 68, flush=True)
    print("[SUCCESS] All stages 006, 007, 009–014 completed successfully.", flush=True)
    print(
        "[SUCCESS] CIC pipeline reached FINAL TEST evaluation and visualization.",
        flush=True,
    )
    print(f"Total runtime: {total_elapsed:.1f} seconds", flush=True)
    print("=" * 68, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_pipeline())
