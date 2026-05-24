#!/usr/bin/env python3
"""Stability-aware fusion search over existing saved candidate probabilities."""

from __future__ import annotations

import json
import traceback
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

import car_block_feature_refinement as base
import two_day_rescue_utils as u


PoolItem = Tuple[str, np.ndarray, np.ndarray, Dict[str, object]]


def weighted_reference(label: str, weights: Dict[str, float], yv: np.ndarray) -> PoolItem:
    val = None
    test = None
    for cand_name, w in weights.items():
        vp = u.V2_OUT / "all_val_probs" / f"{u.safe_name(cand_name)}.npy"
        tp = u.V2_OUT / "all_test_probs" / f"{u.safe_name(cand_name)}.npy"
        if not vp.exists() or not tp.exists():
            raise FileNotFoundError(f"Missing weighted reference probability: {cand_name}")
        pv = u.normalize_prob(np.load(vp))
        pt = u.normalize_prob(np.load(tp))
        val = pv * w if val is None else val + pv * w
        test = pt * w if test is None else test + pt * w
    val = u.normalize_prob(val)
    test = u.normalize_prob(test)
    m = u.evaluate_prob(yv, val)
    return label, val, test, {"name": label, **m, "score": u.score_metrics(m), "weights": weights, "source": "weighted_reference"}


def fuse(items: Sequence[PoolItem], weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pv = np.zeros_like(items[0][1], dtype=np.float64)
    pt = np.zeros_like(items[0][2], dtype=np.float64)
    for item, w in zip(items, weights):
        pv += item[1] * float(w)
        pt += item[2] * float(w)
    return u.normalize_prob(pv), u.normalize_prob(pt)


def test_distribution(prob: np.ndarray) -> Dict[str, int]:
    return {str(k): int(v) for k, v in Counter(u.normalize_prob(prob).argmax(axis=1).tolist()).items()}


def weight_meta(items: Sequence[PoolItem], weights: np.ndarray) -> Dict[str, float]:
    return {item[0]: float(w) for item, w in sorted(zip(items, weights), key=lambda x: -x[1]) if w > 1e-5}


def record(name: str, pv: np.ndarray, pt: np.ndarray, yv: np.ndarray, out_dir: Path, meta: Dict[str, object]) -> Dict[str, object]:
    m = u.evaluate_prob(yv, pv)
    split = u.split_metrics(yv, pv, u.make_splits(len(yv)))
    seed_path = out_dir / f"{name}_SEED.txt"
    seed_check = u.write_seed_from_prob(seed_path, pt)
    np.save(out_dir / f"{name}_val_probs.npy", u.normalize_prob(pv))
    np.save(out_dir / f"{name}_test_probs.npy", u.normalize_prob(pt))
    weights = meta.get("weights", {})
    max_weight = max(weights.values()) if isinstance(weights, dict) and weights else 1.0
    td = seed_check["distribution"]
    max_class = max(td.values()) if td else 450
    stability_objective = (
        split["combined_split_eval_acc_mean"]
        + 0.30 * m["macro_f1"]
        + 0.20 * m["min_recall"]
        - 0.35 * max(0.0, max_weight - 0.55)
        - 0.001 * max(0, max_class - 230)
    )
    return {
        "candidate_name": name,
        **m,
        **split,
        "test_prediction_distribution": td,
        "weights": weights,
        "selected_candidates": meta.get("selected_candidates", []),
        "max_weight": float(max_weight),
        "stability_objective": float(stability_objective),
        "SEED_txt_path": str(seed_path),
        "seed_validation": seed_check,
        "meta": meta,
    }


def random_search(pool: Sequence[PoolItem], yv: np.ndarray, out_dir: Path) -> List[Dict[str, object]]:
    rng = np.random.default_rng(20260522)
    records: List[Dict[str, object]] = []
    top = list(pool[:14])
    for i, item in enumerate(top[:10]):
        records.append(record(f"single_rank_{i+1}", item[1], item[2], yv, out_dir, {"weights": {item[0]: 1.0}, "selected_candidates": [item[0]], "strategy": "single"}))
    concentrations = [12.0, 30.0, 60.0, 120.0]
    for k in [4, 6, 8, 10, 12, 14]:
        items = top[:k]
        for conc in concentrations:
            alpha = np.ones(k) * conc / k
            best = None
            trials = 7000 if k <= 8 else 4000
            for _ in range(trials):
                w = rng.dirichlet(alpha)
                if w.max() > 0.62:
                    continue
                pv, pt = fuse(items, w)
                m = u.evaluate_prob(yv, pv)
                td = test_distribution(pt)
                if m["min_recall"] < 0.6666666:
                    continue
                if max(td.values()) > 245:
                    continue
                split = u.split_metrics(yv, pv, u.make_splits(len(yv)))
                obj = (
                    1.15 * m["val_acc"]
                    + 0.35 * m["macro_f1"]
                    + 0.20 * m["min_recall"]
                    + 0.45 * split["combined_split_eval_acc_mean"]
                    + 0.20 * split["block10_even_odd_eval_acc_mean"]
                    - 0.30 * max(0.0, float(w.max()) - 0.50)
                    - 0.0012 * max(0, max(td.values()) - 220)
                )
                if best is None or obj > best[0]:
                    best = (obj, w.copy(), pv, pt)
            if best is not None:
                weights = weight_meta(items, best[1])
                records.append(
                    record(
                        f"dirichlet_k{k}_conc{str(conc).replace('.', 'p')}",
                        best[2],
                        best[3],
                        yv,
                        out_dir,
                        {"weights": weights, "selected_candidates": list(weights), "strategy": "regularized_dirichlet", "k": k, "concentration": conc, "objective": float(best[0])},
                    )
                )
    return records


def pick_three(records: Sequence[Dict[str, object]]) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, object]]:
    feasible = [r for r in records if r["min_recall"] >= 0.6666666 and r["max_weight"] <= 0.62]
    best_val = max(feasible or records, key=lambda r: (r["val_acc"], r["macro_f1"], r["min_recall"]))
    stable_pool = [r for r in feasible if max(r["test_prediction_distribution"].values()) <= 235]
    best_stable = max(stable_pool or feasible or records, key=lambda r: (r["stability_objective"], r["combined_split_eval_acc_mean"], r["val_acc"]))
    safe_pool = [r for r in feasible if max(r["test_prediction_distribution"].values()) <= 225 and r["block10_even_odd_eval_acc_mean"] >= 0.66]
    best_safe = max(safe_pool or stable_pool or feasible or records, key=lambda r: (r["block10_even_odd_eval_acc_mean"], r["combined_split_eval_acc_mean"], r["min_recall"], r["val_acc"]))
    return best_val, best_stable, best_safe


def copy_named_seed(src_rec: Dict[str, object], out_dir: Path, dest_name: str) -> None:
    src = Path(str(src_rec["SEED_txt_path"]))
    dst = out_dir / f"{dest_name}_SEED.txt"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    src_name = str(src_rec["candidate_name"])
    src_val = out_dir / f"{src_name}_val_probs.npy"
    src_test = out_dir / f"{src_name}_test_probs.npy"
    if src_val.exists():
        np.save(out_dir / f"{dest_name}_val_probs.npy", np.load(src_val))
    if src_test.exists():
        np.save(out_dir / f"{dest_name}_test_probs.npy", np.load(src_test))


def main() -> None:
    out_dir = u.OUTPUT_ROOT / f"regularized_stable_fusion_search_{u.stamp()}"
    out_dir.mkdir(parents=True, exist_ok=False)
    result: Dict[str, object] = {"status": "running", "output_dir": str(out_dir), "errors": []}
    try:
        _, yv = base.load_xy(u.DATA / "val.h5", True)
        pool = u.load_candidate_pool(yv, max_candidates=100)
        v25_json = json.loads((u.V25_OUT / "run_results.json").read_text(encoding="utf-8"))
        references = [
            weighted_reference("old_balanced_reference", v25_json["picks"]["old_balanced"]["weights"], yv),
            weighted_reference("old_aggressive_reference", v25_json["picks"]["old_aggressive"]["weights"], yv),
            weighted_reference("v25_balanced_plus_reference", v25_json["picks"]["balanced_plus"]["weights"], yv),
            weighted_reference("v25_safe_reference", v25_json["picks"]["safest"]["weights"], yv),
        ]
        by_key = set()
        merged: List[PoolItem] = []
        for item in references + pool:
            key = (np.round(item[1], 8).tobytes(), np.round(item[2], 8).tobytes())
            if key in by_key:
                continue
            by_key.add(key)
            merged.append(item)
        merged.sort(key=lambda x: (x[3]["val_acc"], x[3]["macro_f1"], x[3]["score"]), reverse=True)
        records = random_search(merged, yv, out_dir)
        ranked = sorted(records, key=lambda r: (r["val_acc"], r["macro_f1"], r["stability_objective"]), reverse=True)
        best_val, best_stable, best_safe = pick_three(records)
        aliases = [
            ("best_val_fusion", best_val),
            ("best_stable_fusion", best_stable),
            ("best_safe_fusion", best_safe),
        ]
        for alias, rec in aliases:
            copy_named_seed(rec, out_dir, alias)
            prob = np.load(out_dir / f"{rec['candidate_name']}_test_probs.npy")
            rec[f"{alias}_seed_validation"] = u.validate_seed_against_prob(out_dir / f"{alias}_SEED.txt", prob)
        u.write_json(
            out_dir / "fusion_weights.json",
            {
                "best_val_fusion": best_val.get("weights", {}),
                "best_stable_fusion": best_stable.get("weights", {}),
                "best_safe_fusion": best_safe.get("weights", {}),
            },
        )
        u.write_csv(out_dir / "regularized_stable_fusion_candidates.csv", ranked)
        readme = [
            "# Regularized Stable Fusion Search",
            "",
            "This run does not expand the candidate pool. It searches only over existing saved clean candidate probabilities and reconstructed old balanced/safe references.",
            "",
            f"- Output directory: `{out_dir}`",
            f"- Candidate pool after dedupe: `{len(merged)}`",
            "",
            "## Recommended Three",
            "",
            u.md_table(
                [
                    {"tier": "best_val", **best_val},
                    {"tier": "best_stable", **best_stable},
                    {"tier": "best_safe", **best_safe},
                ],
                ["tier", "candidate_name", "val_acc", "macro_f1", "min_recall", "test_prediction_distribution", "max_weight", "combined_split_eval_acc_mean", "block10_even_odd_eval_acc_mean", "SEED_txt_path"],
            ),
            "",
            "## Top By Validation",
            "",
            u.md_table(ranked[:20], ["candidate_name", "val_acc", "macro_f1", "min_recall", "test_prediction_distribution", "max_weight", "combined_split_eval_acc_mean", "block10_even_odd_eval_acc_mean", "SEED_txt_path"]),
        ]
        (out_dir / "README_regularized_stable_fusion.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
        result.update({"status": "completed", "pool_size": len(merged), "best_val": best_val, "best_stable": best_stable, "best_safe": best_safe})
    except Exception as exc:
        result["status"] = "failed_partial"
        result["errors"].append({"error": repr(exc), "trace": traceback.format_exc()})
        (out_dir / "README_regularized_stable_fusion.md").write_text(
            "# Regularized Stable Fusion Search\n\nRun failed; see `run_results.json` for traceback.\n", encoding="utf-8"
        )
    finally:
        u.write_json(out_dir / "run_results.json", result)
        print(f"OUTPUT_DIR={out_dir}")


if __name__ == "__main__":
    main()
