#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Follow-up experiments for the best forensic preprocessing candidate."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "outputs_preprocessing_forensics" / "run_20260521_231226"
OUT_ROOT = RUN_DIR / "followup_runs"
H5_100 = RUN_DIR / "converted_candidates" / "c11_fixed20_official_global_spc100.h5"
H5_200 = RUN_DIR / "converted_candidates" / "c11_fixed20_official_global.h5"
H5_500 = ROOT / "outputs_preprocessing_forensics" / "run_20260521_231928" / "converted_candidates" / "c11_fixed20_official_global.h5"


FIELDS = [
    "run_id", "candidate_id", "samples_per_class", "external_ratio", "seed", "best_val_acc",
    "best_epoch", "final_val_acc", "macro_f1", "prediction_distribution", "output_dir",
    "checkpoint_path", "status", "error_message",
]


def run_one(run_id: str, h5_path: Path, spc: int, ratio: float, seed: int) -> dict:
    cmd = [
        sys.executable, "seed_multiscale_crnn_experiment.py",
        "--model", "multiscale_crnn",
        "--epochs", "8",
        "--patience", "4",
        "--batch-size", "64",
        "--seed", str(seed),
        "--output-root", str(OUT_ROOT),
        "--run-name", run_id,
        "--use-supplement",
        "--supplement-h5", str(h5_path),
        "--external-ratio", str(ratio),
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (RUN_DIR / "followup_logs").mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "followup_logs" / f"{run_id}.log").write_text(proc.stdout, encoding="utf-8", errors="replace")
    row = {
        "run_id": run_id,
        "candidate_id": "c11_fixed20_official_global",
        "samples_per_class": spc,
        "external_ratio": ratio,
        "seed": seed,
        "status": "failed" if proc.returncode else "ok",
        "error_message": "" if proc.returncode == 0 else f"exit {proc.returncode}",
    }
    if proc.returncode == 0:
        dirs = sorted(OUT_ROOT.glob(f"{run_id}_*"), key=lambda p: p.stat().st_mtime)
        res = json.loads((dirs[-1] / "run_results.json").read_text(encoding="utf-8"))
        val = res.get("validation_metrics_at_best") or {}
        row.update({
            "best_val_acc": res.get("best_val_acc", ""),
            "best_epoch": res.get("best_epoch", ""),
            "final_val_acc": res.get("final_val_acc", ""),
            "macro_f1": res.get("best_val_macro_f1", ""),
            "prediction_distribution": json.dumps(val.get("prediction_distribution", {}), ensure_ascii=False),
            "output_dir": res.get("output_dir", ""),
            "checkpoint_path": res.get("best_acc_checkpoint", ""),
        })
    return row


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    specs = [
        ("c11_spc200_r0005_s42", H5_200, 200, 0.005, 42),
        ("c11_spc200_r001_s42", H5_200, 200, 0.01, 42),
        ("c11_spc200_r002_s42", H5_200, 200, 0.02, 42),
        ("c11_spc200_r005_s42", H5_200, 200, 0.05, 42),
        ("c11_spc100_r002_s42", H5_100, 100, 0.02, 42),
        ("c11_spc500_r002_s42", H5_500, 500, 0.02, 42),
        ("c11_spc200_r002_s3407", H5_200, 200, 0.02, 3407),
    ]
    rows = []
    for spec in specs:
        print(f"followup {spec[0]}")
        rows.append(run_one(*spec))
    out = RUN_DIR / "preprocessing_followup_results.csv"
    with out.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows([{k: r.get(k, "") for k in FIELDS} for r in rows])
    print(f"followup_results: {out}")


if __name__ == "__main__":
    main()
