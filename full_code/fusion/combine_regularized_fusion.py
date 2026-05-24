#!/usr/bin/env python3
"""Combine v2 and v3 clean candidates, then rerun a compact regularized fusion check."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import car_block_feature_refinement as base
import clean_feature_block_fusion_v2 as v2
import regularized_clean_fusion_search as reg


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "course_project" / "SEED"
V2_OUT = ROOT / "outputs_experiments" / "clean_feature_block_fusion_v2_20260522_085528"
COMBINED_DIR = V2_OUT / "regularized_fusion_search"

PoolItem = Tuple[str, np.ndarray, np.ndarray, Dict[str, Any]]


def latest_v3_dir() -> Path:
    dirs = sorted((ROOT / "outputs_experiments").glob("clean_feature_block_fusion_v3_*"))
    if not dirs:
        raise FileNotFoundError("No clean_feature_block_fusion_v3_* output directory found.")
    return dirs[-1]


def read_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def restore_v3_pool(v3_dir: Path, yv: np.ndarray) -> List[PoolItem]:
    rows = read_rows(v3_dir / "all_clean_candidates_summary.csv")
    pool: List[PoolItem] = []
    for row in rows:
        name = row.get("name", "")
        if row.get("risk_level") != "clean-low":
            continue
        if not reg.falseish(row.get("uses_order_prior")) or not reg.falseish(row.get("uses_source_matching")):
            continue
        if reg.clean_forbidden_check(row):
            continue
        vp = v3_dir / "all_val_probs" / f"{v2.safe_name(name)}.npy"
        tp = v3_dir / "all_test_probs" / f"{v2.safe_name(name)}.npy"
        if not vp.exists() or not tp.exists():
            continue
        pv = v2.normalize_prob(np.load(vp))
        pte = v2.normalize_prob(np.load(tp))
        m = reg.evaluate_prob(yv, pv)
        combined_name = f"v3__{name}"
        meta = {**row, **m, "score": reg.score(m), "origin_dir": str(v3_dir), "name": combined_name}
        pool.append((combined_name, pv, pte, meta))
    return pool


def dedupe(pool: List[PoolItem]) -> List[PoolItem]:
    out: List[PoolItem] = []
    seen = set()
    for item in sorted(pool, key=lambda x: (x[3]["val_acc"], x[3]["macro_f1"], x[3]["score"]), reverse=True):
        key = (np.round(item[1], 8).tobytes(), np.round(item[2], 8).tobytes())
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def main() -> None:
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    reg.SEARCH_DIR = COMBINED_DIR
    _, yv = base.load_xy(DATA / "val.h5", True)
    v3_dir = latest_v3_dir()
    v2_pool = reg.restore_pool(yv)
    v3_pool = restore_v3_pool(v3_dir, yv)
    pool = dedupe(v2_pool + v3_pool)
    splits = reg.make_splits(len(yv))

    records = []
    best_single = pool[0]
    records.append(reg.candidate_record("combined_best_single", best_single[1], best_single[2], yv, {"selected": [best_single[0]], "weights": {best_single[0]: 1.0}, "strategy": "combined_best_single"}, splits))

    gv, gt, gmeta = reg.greedy_full(pool, yv)
    gmeta["strategy"] = "combined_greedy_unlimited"
    records.append(reg.candidate_record("combined_greedy_unlimited", gv, gt, yv, gmeta, splits))

    dv, dt, dmeta = reg.greedy_full(pool, yv, max_selected=8, corr_threshold=0.95)
    dmeta["strategy"] = "combined_diversity_corr_0p95"
    records.append(reg.candidate_record("combined_diversity_corr_0p95", dv, dt, yv, dmeta, splits))

    selected = gmeta["selected"]
    if len(selected) >= 2:
        rv, rt, rmeta = reg.random_weight_search(
            pool,
            selected,
            yv,
            "score",
            n=30000,
            constraints={
                "center_weights": gmeta["weights"],
                "concentration": 80.0,
                "max_any_weight": 0.55,
                "min_class2_test": 60,
                "max_class1_test": 230,
                "max_class0_test": 230,
            },
        )
        rmeta["strategy"] = "combined_distribution_regularized"
        records.append(reg.candidate_record("combined_distribution_regularized", rv, rt, yv, rmeta, splits))

    ranked = sorted(records, key=lambda r: (r["val_acc"], r["macro_f1"], r["combined_split_eval_acc_mean"]), reverse=True)
    v2.write_csv(COMBINED_DIR / "v2_v3_combined_fusion_candidates.csv", ranked)
    best = ranked[0]
    previous = json.loads((V2_OUT / "regularized_fusion_search" / "regularized_fusion_run_results.json").read_text(encoding="utf-8"))
    old_aggressive = previous["aggressive_candidate"]
    old_balanced = previous["balanced_candidate"]
    exceeded = bool(best["val_acc"] > old_aggressive["val_acc"] + 1e-12)
    introduced_v3 = any(str(name).startswith("v3__") for name in best.get("selected_candidates", []))
    recommendation = best if exceeded else old_balanced
    md = [
        "# v2 + v3 Combined Fusion Recommendation",
        "",
        f"v2 pool size: `{len(v2_pool)}`",
        f"v3 pool size: `{len(v3_pool)}`",
        f"combined deduped pool size: `{len(pool)}`",
        f"v3 output dir: `{v3_dir}`",
        "",
        f"Best combined candidate: `{best['candidate_name']}`",
        f"- val_acc: `{best['val_acc']}`",
        f"- macro-F1: `{best['macro_f1']}`",
        f"- min_recall: `{best['min_recall']}`",
        f"- test distribution: `{json.dumps(best['test_prediction_distribution'], ensure_ascii=False)}`",
        f"- selected includes v3: `{introduced_v3}`",
        f"- exceeds original regularized aggressive val_acc: `{exceeded}`",
        "",
        "Recommendation:",
        f"- Final recommended candidate remains: `{recommendation['candidate_name']}`",
        f"- SEED path: `{recommendation.get('balanced_copy_path') or recommendation.get('SEED_txt_path')}`",
        "",
        "The v3 targeted EEG features did not improve over the v2 clean feature fusion in this run.",
    ]
    (COMBINED_DIR / "v2_v3_final_recommendation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    v2.write_json(COMBINED_DIR / "v2_v3_combined_run_results.json", {"status": "completed", "v3_dir": str(v3_dir), "best_combined": best, "exceeded_regularized": exceeded, "introduced_v3_in_best": introduced_v3})
    print(json.dumps(v2.jsonable({"best_combined": best, "exceeded_regularized": exceeded, "introduced_v3_in_best": introduced_v3}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
