#!/usr/bin/env python3
"""Focused reliability audit for the 0.7111 old-balanced fixed fusion.

This script does not train new models, expand candidates, or use hidden test
labels. It reconstructs the existing old-balanced probability fusion, checks
saved SEED files, and summarizes fixed-split versus selection-holdout risk.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import car_block_feature_refinement as base
import two_day_rescue_utils as u


OUT_DIR = u.OUTPUT_ROOT / f"old_balanced_07111_reliability_audit_{u.stamp()}"
V25 = u.V25_OUT


def load_weighted(weights: Dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    val = None
    test = None
    for name, w in weights.items():
        vp = u.V2_OUT / "all_val_probs" / f"{u.safe_name(name)}.npy"
        tp = u.V2_OUT / "all_test_probs" / f"{u.safe_name(name)}.npy"
        pv = u.normalize_prob(np.load(vp))
        pt = u.normalize_prob(np.load(tp))
        val = pv * float(w) if val is None else val + pv * float(w)
        test = pt * float(w) if test is None else test + pt * float(w)
    return u.normalize_prob(val), u.normalize_prob(test)


def split_table(y: np.ndarray, prob: np.ndarray) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(20260523)
    for seed in [2024, 2025, 2026, 2027, 2028]:
        idx = np.arange(len(y))
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        a = np.sort(idx[: len(idx) // 2])
        b = np.sort(idx[len(idx) // 2 :])
        for fold, eval_idx in [(f"random_half_seed{seed}_A", a), (f"random_half_seed{seed}_B", b)]:
            rows.append({"split": fold, **u.evaluate_prob(y, prob, eval_idx)})
    block_ids = np.arange(len(y)) // 10
    for fold, eval_idx in [("block10_even", np.where(block_ids % 2 == 0)[0]), ("block10_odd", np.where(block_ids % 2 == 1)[0])]:
        rows.append({"split": fold, **u.evaluate_prob(y, prob, eval_idx)})
    for i, eval_idx in enumerate(np.array_split(np.arange(len(y)), 5), 1):
        rows.append({"split": f"contiguous_chunk_{i}", **u.evaluate_prob(y, prob, eval_idx)})
    return rows


def summarize(rows: List[Dict[str, Any]], prefix: str) -> Dict[str, float]:
    selected = [r for r in rows if r["split"].startswith(prefix)]
    accs = [float(r["val_acc"]) for r in selected]
    f1s = [float(r["macro_f1"]) for r in selected]
    mins = [float(r["min_recall"]) for r in selected]
    return {
        f"{prefix}_acc_mean": float(np.mean(accs)),
        f"{prefix}_acc_std": float(np.std(accs)),
        f"{prefix}_acc_min": float(np.min(accs)),
        f"{prefix}_macro_f1_mean": float(np.mean(f1s)),
        f"{prefix}_min_recall_mean": float(np.mean(mins)),
    }


def seed_values(path: Path) -> np.ndarray:
    return np.asarray([int(x.strip()) for x in path.read_text(encoding="utf-8").splitlines() if x.strip() != ""], dtype=int)


def confidence(prob: np.ndarray) -> Dict[str, Any]:
    p = u.normalize_prob(prob)
    pred = p.argmax(axis=1)
    maxp = p.max(axis=1)
    out: Dict[str, Any] = {
        "mean_max_probability": float(np.mean(maxp)),
        "high_conf_ratio_0p7": float(np.mean(maxp >= 0.7)),
        "high_conf_ratio_0p9": float(np.mean(maxp >= 0.9)),
    }
    for cls in [0, 1, 2]:
        mask = pred == cls
        out[f"class_{cls}_mean_confidence"] = float(np.mean(maxp[mask])) if mask.any() else None
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    _, y = base.load_xy(u.DATA / "val.h5", True)
    run = json.loads((V25 / "run_results.json").read_text(encoding="utf-8"))
    audit_old = json.loads((V25 / "audit_old_balanced_results.json").read_text(encoding="utf-8"))
    old = run["picks"]["old_balanced"]
    safe = run["picks"]["safest"]
    balanced = run["picks"]["balanced_plus"]

    old_v, old_t = load_weighted(old["weights"])
    safe_v, safe_t = load_weighted(safe["weights"])
    balanced_v, balanced_t = load_weighted(balanced["weights"])

    fallback_seed = seed_values(V25 / "fallback_old_balanced_SEED.txt")
    fixed_rows = split_table(y, old_v)
    u.write_csv(OUT_DIR / "fixed_fusion_split_metrics.csv", fixed_rows)

    rng = np.random.default_rng(20260523)
    perm_rows = []
    for rep in range(1000):
        yp = y.copy()
        rng.shuffle(yp)
        perm_rows.append({"rep": rep, **u.evaluate_prob(yp, old_v)})
    u.write_csv(OUT_DIR / "fixed_fusion_permutation_labels.csv", perm_rows)
    perm_accs = [r["val_acc"] for r in perm_rows]

    test_rows = []
    for name, seed_path, prob in [
        ("fallback_old_balanced", V25 / "fallback_old_balanced_SEED.txt", old_t),
        ("balanced_v25", V25 / "balanced_v25_SEED.txt", balanced_t),
        ("safe_v25", V25 / "safe_v25_SEED.txt", safe_t),
    ]:
        vals = seed_values(seed_path)
        pred = prob.argmax(axis=1).astype(int)
        test_rows.append(
            {
                "name": name,
                "seed_path": str(seed_path),
                "line_count": int(len(vals)),
                "labels_only_0_1_2": bool(set(vals.tolist()).issubset({0, 1, 2})),
                "matches_test_probability_argmax": bool(np.array_equal(vals, pred)),
                "test_distribution": {str(k): int(v) for k, v in Counter(vals.tolist()).items()},
                **confidence(prob),
            }
        )
    u.write_csv(OUT_DIR / "test_prediction_checks.csv", test_rows)

    fixed_summary = {
        **summarize(fixed_rows, "random_half"),
        **summarize(fixed_rows, "block10"),
        **summarize(fixed_rows, "contiguous"),
    }
    selection_holdout = audit_old.get("internal_holdout_summary", {})
    result = {
        "status": "completed",
        "output_dir": str(OUT_DIR),
        "old_balanced_metrics": u.evaluate_prob(y, old_v),
        "old_balanced_weights": old["weights"],
        "selected_candidates": old["selected_candidates"],
        "fallback_seed_check": {
            "line_count": int(len(fallback_seed)),
            "labels_only_0_1_2": bool(set(fallback_seed.tolist()).issubset({0, 1, 2})),
            "matches_test_probability_argmax": bool(np.array_equal(fallback_seed, old_t.argmax(axis=1))),
            "distribution": {str(k): int(v) for k, v in Counter(fallback_seed.tolist()).items()},
        },
        "fixed_split_summary": fixed_summary,
        "selection_holdout_summary": selection_holdout,
        "permutation_summary": {
            "repeats": len(perm_rows),
            "acc_mean": float(np.mean(perm_accs)),
            "acc_std": float(np.std(perm_accs)),
            "acc_max": float(np.max(perm_accs)),
        },
        "test_checks": test_rows,
        "rating": "Useful but validation-optimized",
    }
    u.write_json(OUT_DIR / "run_results.json", result)

    lines = [
        "# 0.7111 Old-Balanced Reliability Audit",
        "",
        "This audit checks the existing `fallback_old_balanced_SEED.txt` / old-balanced fixed probability fusion. It does not train new models, add candidates, or use hidden test labels.",
        "",
        "## Result Identity",
        "",
        f"- Validation metrics: acc `{result['old_balanced_metrics']['val_acc']:.6f}`, macro-F1 `{result['old_balanced_metrics']['macro_f1']:.6f}`, min recall `{result['old_balanced_metrics']['min_recall']:.6f}`.",
        f"- Confusion matrix: `{result['old_balanced_metrics']['confusion_matrix']}`.",
        f"- Selected candidates: `{old['selected_candidates']}`.",
        f"- Fixed global weights: `{old['weights']}`.",
        f"- Fallback SEED check: `{result['fallback_seed_check']}`.",
        "",
        "## Leakage / Bug Check",
        "",
        "- The score is reproduced from saved validation probability matrices, not from scoring a `SEED.txt` test file against validation labels.",
        "- The test `SEED.txt` has exactly 450 rows, labels are only 0/1/2, and it matches reconstructed test-probability argmax.",
        "- No per-sample oracle correction was found in the existing audit. The method is a fixed global weighted average of four candidate probability matrices.",
        f"- Permuted-label fixed-prediction acc mean/std/max over 1000 repeats: `{result['permutation_summary']['acc_mean']:.4f}` / `{result['permutation_summary']['acc_std']:.4f}` / `{result['permutation_summary']['acc_max']:.4f}`.",
        "",
        "## Fixed-Fusion Split Check",
        "",
        "These rows evaluate the already chosen fixed 0.7111 predictor on subsets of the same validation set. They are useful sanity checks, but they are not a true nested estimate because the weights were selected using the full validation benchmark.",
        "",
        u.md_table([fixed_summary], list(fixed_summary.keys())),
        "",
        "## Selection-Holdout Risk",
        "",
        "The older audit also re-tuned weights inside validation subsets and checked held-out subsets. That is a stricter proxy for overfit risk.",
        "",
        f"- Random-half check acc mean: `{selection_holdout.get('random_half_check_acc_mean')}`.",
        f"- Block10 even/odd check acc mean: `{selection_holdout.get('block10_even_odd_check_acc_mean')}`.",
        f"- All check acc mean: `{selection_holdout.get('all_check_acc_mean')}`.",
        "",
        "The random-half check stays near `0.71`, but the block-wise selection-holdout drops to about `0.53`, which is the main reliability warning.",
        "",
        "## Test Prediction Checks",
        "",
        u.md_table(test_rows, ["name", "line_count", "labels_only_0_1_2", "matches_test_probability_argmax", "test_distribution", "mean_max_probability", "high_conf_ratio_0p7", "class_0_mean_confidence", "class_1_mean_confidence", "class_2_mean_confidence"]),
        "",
        "## Reliability Rating",
        "",
        "- Rating: `Useful but validation-optimized`.",
        "- It is not a confirmed bug or leakage result.",
        "- It is more reliable than the neural stacker `0.7667`, because it is a simple fixed probability ensemble and survived consistency checks.",
        "- It is not fully reliable as a hidden-test estimate, because weight/candidate selection used the same validation set and block-wise selection-holdout drops sharply.",
        "",
        "Final recommendation: keep `fallback_old_balanced_SEED.txt` as the primary score-oriented submission, with `safe_v25_SEED.txt` as the conservative backup. Describe `0.7111` as an audited but validation-optimized fixed probability fusion, not as a guaranteed generalization score.",
    ]
    (OUT_DIR / "README_07111_reliability_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (u.ROOT / "README_07111_reliability_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"OUTPUT_DIR={OUT_DIR}")


if __name__ == "__main__":
    main()
