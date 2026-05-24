#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train top preprocessing candidates selected by forensic statistical distance."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent
SUPPLEMENT = ROOT / "outputs_external" / "supplement_seed_like.h5"


FIELDS = [
    "candidate_id", "data_source", "windowing", "normalization", "stat_score_rank",
    "external_ratio", "best_val_acc", "best_epoch", "final_val_acc", "macro_f1",
    "prediction_distribution", "output_dir", "checkpoint_path", "status", "error_message",
]


def run_cmd(cmd: List[str], log_path: Path) -> int:
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    return proc.returncode


def read_result(out_root: Path, run_name: str) -> Dict[str, object]:
    dirs = sorted(out_root.glob(f"{run_name}_*"), key=lambda p: p.stat().st_mtime)
    if not dirs:
        raise RuntimeError(f"No output directory for {run_name}")
    return json.loads((dirs[-1] / "run_results.json").read_text(encoding="utf-8"))


def train_one(run_dir: Path, item: Dict[str, object], rank: int) -> Dict[str, object]:
    out_root = run_dir / "training_runs"
    run_name = f"forensic_{item['candidate_id']}"
    cmd = [
        sys.executable, "seed_multiscale_crnn_experiment.py",
        "--model", "multiscale_crnn",
        "--epochs", "8",
        "--patience", "4",
        "--batch-size", "64",
        "--seed", "42",
        "--output-root", str(out_root),
        "--run-name", run_name,
    ]
    ratio = "0.02"
    if item["candidate_id"] != "no_external":
        cmd += ["--use-supplement", "--supplement-h5", str(item["h5_path"]), "--external-ratio", ratio]
    code = run_cmd(cmd, run_dir / "training_logs" / f"{run_name}.log")
    if code != 0:
        return {
            "candidate_id": item["candidate_id"],
            "data_source": item.get("data_source", ""),
            "windowing": item.get("windowing", ""),
            "normalization": item.get("normalization", ""),
            "stat_score_rank": rank,
            "external_ratio": "" if item["candidate_id"] == "no_external" else ratio,
            "status": "failed",
            "error_message": f"training command exited {code}",
        }
    result = read_result(out_root, run_name)
    val = result.get("validation_metrics_at_best") or {}
    return {
        "candidate_id": item["candidate_id"],
        "data_source": item.get("data_source", ""),
        "windowing": item.get("windowing", ""),
        "normalization": item.get("normalization", ""),
        "stat_score_rank": rank,
        "external_ratio": "" if item["candidate_id"] == "no_external" else ratio,
        "best_val_acc": result.get("best_val_acc", ""),
        "best_epoch": result.get("best_epoch", ""),
        "final_val_acc": result.get("final_val_acc", ""),
        "macro_f1": result.get("best_val_macro_f1", ""),
        "prediction_distribution": json.dumps(val.get("prediction_distribution", {}), ensure_ascii=False),
        "output_dir": result.get("output_dir", ""),
        "checkpoint_path": result.get("best_acc_checkpoint", ""),
        "status": "ok",
        "error_message": "",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stats_path = args.run_dir / "preprocessing_candidate_stats.csv"
    rows = list(csv.DictReader(stats_path.open(encoding="utf-8")))
    rows.sort(key=lambda r: float(r["overall_stat_score"]))
    candidates: List[Dict[str, object]] = [
        {"candidate_id": "no_external", "data_source": "official_only", "windowing": "", "normalization": "", "h5_path": ""}
    ]
    if SUPPLEMENT.exists():
        candidates.append({
            "candidate_id": "current_supplement_seed_like",
            "data_source": "existing_supplement",
            "windowing": "prior_random_windows",
            "normalization": "prior_conversion",
            "h5_path": str(SUPPLEMENT),
        })
    candidates += rows[:args.top_k]
    out_rows = []
    for rank, item in enumerate(candidates, start=0):
        print(f"training {item['candidate_id']}")
        out_rows.append(train_one(args.run_dir, item, rank))
    out_path = args.run_dir / "preprocessing_training_results.csv"
    with out_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDS)
        writer.writeheader()
        for row in out_rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})
    print(f"training_results: {out_path}")


if __name__ == "__main__":
    main()
