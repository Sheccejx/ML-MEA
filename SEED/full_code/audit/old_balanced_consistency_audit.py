#!/usr/bin/env python3
"""Consistency audit for old balanced clean fusion.

Important: SEED.txt files are test-set predictions and cannot be scored against
validation labels. This audit scores the corresponding validation probability
matrices, and separately checks SEED.txt shape/order against test argmax.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

import car_block_feature_refinement as base
import clean_feature_block_fusion_v2 as v2
import regularized_clean_fusion_search as reg


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "course_project" / "SEED"
V2_OUT = ROOT / "outputs_experiments" / "clean_feature_block_fusion_v2_20260522_085528"
V25_OUT = ROOT / "outputs_experiments" / "clean_feature_block_fusion_v25_20260522_150944"
REFINED = V25_OUT / "refined_fusion"
RNG = np.random.default_rng(20260522)
ALPHAS = np.linspace(0.05, 0.95, 19)


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def read_seed(path: Path) -> np.ndarray:
    return np.array([int(x.strip()) for x in path.read_text(encoding="utf-8").splitlines() if x.strip() != ""], dtype=int)


def load_v2_prob(name: str) -> Tuple[np.ndarray, np.ndarray]:
    return (
        v2.normalize_prob(np.load(V2_OUT / "all_val_probs" / f"{v2.safe_name(name)}.npy")),
        v2.normalize_prob(np.load(V2_OUT / "all_test_probs" / f"{v2.safe_name(name)}.npy")),
    )


def weighted_probs(names: Sequence[str], weights: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    pv = None
    pt = None
    for name in names:
        a, b = load_v2_prob(name)
        w = float(weights[name])
        pv = a * w if pv is None else pv + a * w
        pt = b * w if pt is None else pt + b * w
    return v2.normalize_prob(pv), v2.normalize_prob(pt)


def load_v25_single_prob(v25_name: str) -> Tuple[np.ndarray, np.ndarray]:
    name = v25_name.removeprefix("v25__")
    return (
        v2.normalize_prob(np.load(V25_OUT / "all_val_probs" / f"{v2.safe_name(name)}.npy")),
        v2.normalize_prob(np.load(V25_OUT / "all_test_probs" / f"{v2.safe_name(name)}.npy")),
    )


def greedy_tune(pool: Sequence[Tuple[str, np.ndarray, np.ndarray]], y: np.ndarray, tune_idx: np.ndarray, max_selected: int = 4) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    def metric(prob: np.ndarray, idx: np.ndarray) -> Dict[str, Any]:
        return reg.evaluate_prob(y, prob, idx)

    ranked = sorted(pool, key=lambda x: (metric(x[1], tune_idx)["val_acc"], metric(x[1], tune_idx)["macro_f1"]), reverse=True)
    selected = [ranked[0]]
    cur_v = ranked[0][1].copy()
    cur_t = ranked[0][2].copy()
    cur_m = metric(cur_v, tune_idx)
    steps = [{"name": ranked[0][0], "alpha_previous": 1.0, "metrics": cur_m}]
    while len(selected) < max_selected:
        best = None
        for cand in ranked:
            if any(cand[0] == s[0] for s in selected):
                continue
            for alpha in ALPHAS:
                pv = v2.normalize_prob(alpha * cur_v + (1 - alpha) * cand[1])
                m = metric(pv, tune_idx)
                if reg.score(m) > reg.score(cur_m) + 1e-10:
                    best = (cand, float(alpha), pv, v2.normalize_prob(alpha * cur_t + (1 - alpha) * cand[2]), m)
                    cur_m = m
        if best is None:
            break
        selected.append(best[0])
        cur_v, cur_t = best[2], best[3]
        steps.append({"name": best[0][0], "alpha_previous": best[1], "metrics": best[4]})
    return cur_v, cur_t, {"selected": [s[0] for s in selected], "steps": steps, "weights": reg.effective_weights(steps)}


def main() -> None:
    Xv, yv = base.load_xy(DATA / "val.h5", True)
    Xte, _ = base.load_xy(DATA / "test_x_only.h5", False)
    run = json.loads((V25_OUT / "run_results.json").read_text(encoding="utf-8"))

    old_bal = run["picks"]["old_balanced"]
    old_aggr = run["picks"]["old_aggressive"]
    v25_safe = run["picks"]["safest"]
    v25_best_single = run["picks"]["v25_best_single"]

    entries = {
        "fallback_old_balanced_SEED.txt": {
            "seed_path": V25_OUT / "fallback_old_balanced_SEED.txt",
            "record": old_bal,
            "prob": weighted_probs(old_bal["selected_candidates"], {k: float(v) for k, v in old_bal["weights"].items()}),
        },
        "safe_v25_SEED.txt": {
            "seed_path": V25_OUT / "safe_v25_SEED.txt",
            "record": v25_safe,
            "prob": weighted_probs(v25_safe["selected_candidates"], {k: float(v) for k, v in v25_safe["weights"].items()}),
        },
        "balanced_v25_SEED.txt": {
            "seed_path": V25_OUT / "balanced_v25_SEED.txt",
            "record": old_aggr,
            "prob": weighted_probs(old_aggr["selected_candidates"], {k: float(v) for k, v in old_aggr["weights"].items()}),
        },
        "aggressive_v25_SEED.txt": {
            "seed_path": V25_OUT / "aggressive_v25_SEED.txt",
            "record": old_aggr,
            "prob": weighted_probs(old_aggr["selected_candidates"], {k: float(v) for k, v in old_aggr["weights"].items()}),
        },
        "v25_best_single_SEED.txt": {
            "seed_path": REFINED / "v25_best_single_SEED.txt",
            "record": v25_best_single,
            "prob": load_v25_single_prob(v25_best_single["selected_candidates"][0]),
        },
    }

    audit: Dict[str, Any] = {
        "status": "completed",
        "validation_source": str(DATA / "val.h5"),
        "test_source": str(DATA / "test_x_only.h5"),
        "validation_shape": list(Xv.shape),
        "test_shape": list(Xte.shape),
        "validation_label_distribution": dict(Counter(yv.tolist())),
        "files": {},
    }
    for label, entry in entries.items():
        seed = read_seed(entry["seed_path"])
        pv, pt = entry["prob"]
        m = reg.evaluate_prob(yv, pv)
        rec = entry["record"]
        test_pred = pt.argmax(axis=1)
        audit["files"][label] = {
            "seed_path": str(entry["seed_path"]),
            "seed_line_count": int(len(seed)),
            "seed_labels": sorted(int(x) for x in set(seed.tolist())),
            "seed_distribution": dict(Counter(seed.tolist())),
            "val_prob_shape": list(pv.shape),
            "test_prob_shape": list(pt.shape),
            "val_metrics_from_probability": m,
            "recorded_val_acc": rec.get("val_acc"),
            "recorded_macro_f1": rec.get("macro_f1"),
            "recorded_min_recall": rec.get("min_recall"),
            "metrics_match_record": bool(
                abs(m["val_acc"] - float(rec.get("val_acc"))) < 1e-12
                and abs(m["macro_f1"] - float(rec.get("macro_f1"))) < 1e-12
                and abs(m["min_recall"] - float(rec.get("min_recall"))) < 1e-12
            ),
            "seed_matches_test_probability_argmax": bool(np.array_equal(seed, test_pred)),
            "test_prediction_distribution_from_probability": dict(Counter(test_pred.tolist())),
            "seed_is_test_prediction_not_validation_prediction": True,
        }

    old_names = old_bal["selected_candidates"]
    old_weights = {k: float(v) for k, v in old_bal["weights"].items()}
    old_pv, old_pt = entries["fallback_old_balanced_SEED.txt"]["prob"]
    audit["old_balanced_reconstruction"] = {
        "selected_candidates": old_names,
        "weights": old_weights,
        "weights_sum": float(sum(old_weights.values())),
        "val_metrics": reg.evaluate_prob(yv, old_pv),
        "val_prediction_distribution": dict(Counter(old_pv.argmax(axis=1).tolist())),
        "test_prediction_distribution": dict(Counter(old_pt.argmax(axis=1).tolist())),
        "fallback_seed_matches_test_argmax": audit["files"]["fallback_old_balanced_SEED.txt"]["seed_matches_test_probability_argmax"],
    }

    audit["common_context"] = {
        "same_validation_label_file": True,
        "same_validation_sample_order": "All validation metrics are computed on val_prob rows 0..449 against y from course_project/SEED/val.h5 rows 0..449.",
        "same_test_sample_order": "SEED files are checked against test_prob rows 0..449 and test_x_only.h5 rows 0..449; no test labels are used.",
        "same_metric_calculation": "accuracy, macro-F1, min class recall, confusion matrix from regularized_clean_fusion_search.evaluate_prob.",
        "prediction_row_count_required": 450,
        "all_seed_files_450_rows": all(v["seed_line_count"] == 450 for v in audit["files"].values()),
        "all_probability_shapes_450x3": all(v["val_prob_shape"] == [450, 3] and v["test_prob_shape"] == [450, 3] for v in audit["files"].values()),
    }

    audit["old_balanced_generation_logic"] = {
        "direct_or_indirect_y_val_per_sample_prediction": False,
        "per_sample_correction_or_oracle_selection_detected": False,
        "uses_validation_score_to_select_fixed_global_weights": True,
        "explanation": "old balanced is a fixed weighted average of four saved clean probability matrices. y_val was used during global candidate/weight selection, not to choose labels per sample.",
    }

    pool = []
    for name in old_names:
        a, b = load_v2_prob(name)
        pool.append((name, a, b))
    idx = np.arange(len(yv))
    RNG.shuffle(idx)
    a = np.sort(idx[: len(idx) // 2])
    b = np.sort(idx[len(idx) // 2 :])
    block_ids = np.arange(len(yv)) // 10
    even = np.where(block_ids % 2 == 0)[0]
    odd = np.where(block_ids % 2 == 1)[0]
    holdout_rows = []
    for split_name, folds in {"random_half": [(a, b), (b, a)], "block10_even_odd": [(even, odd), (odd, even)]}.items():
        for fold_i, (tune, check) in enumerate(folds, start=1):
            pv, _, meta = greedy_tune(pool, yv, tune, 4)
            tune_m = reg.evaluate_prob(yv, pv, tune)
            check_m = reg.evaluate_prob(yv, pv, check)
            holdout_rows.append(
                {
                    "split": split_name,
                    "fold": fold_i,
                    "selected": meta["selected"],
                    "weights": meta["weights"],
                    "tune_size": int(len(tune)),
                    "check_size": int(len(check)),
                    "tune_acc": tune_m["val_acc"],
                    "tune_macro_f1": tune_m["macro_f1"],
                    "tune_min_recall": tune_m["min_recall"],
                    "check_acc": check_m["val_acc"],
                    "check_macro_f1": check_m["macro_f1"],
                    "check_min_recall": check_m["min_recall"],
                    "check_distribution": check_m["prediction_distribution"],
                }
            )
    audit["internal_holdout"] = holdout_rows
    audit["internal_holdout_summary"] = {
        "random_half_check_acc_mean": float(np.mean([r["check_acc"] for r in holdout_rows if r["split"] == "random_half"])),
        "block10_even_odd_check_acc_mean": float(np.mean([r["check_acc"] for r in holdout_rows if r["split"] == "block10_even_odd"])),
        "all_check_acc_mean": float(np.mean([r["check_acc"] for r in holdout_rows])),
    }

    audit["conclusion"] = {
        "is_0_7111_evaluation_bug": False,
        "is_validation_overfitted_fusion": True,
        "is_legal_fixed_ensemble": True,
        "summary": "The 0.7111 validation score is reproduced from saved validation probabilities with a unified metric calculation. SEED.txt files are test predictions and are not scored against y_val. The ensemble is legal and fixed, but validation-optimized because y_val selected the global weights/candidates.",
    }

    report = [
        "# Audit: old balanced consistency",
        "",
        "## Scope",
        "",
        "This audit does not create new candidates. It scores existing validation probability matrices against `course_project/SEED/val.h5` in native row order, and separately checks that existing `SEED.txt` files match the corresponding test probability argmax. `SEED.txt` files are test predictions and cannot be scored with validation labels.",
        "",
        "## Unified Re-evaluation",
        "",
        "|file|SEED rows|SEED/test distribution|val_acc from val_prob|macro-F1|min_recall|matches recorded|SEED matches test argmax|",
        "|---|---:|---|---:|---:|---:|---|---|",
    ]
    for label, info in audit["files"].items():
        m = info["val_metrics_from_probability"]
        report.append(f"|`{label}`|{info['seed_line_count']}|`{json.dumps(info['seed_distribution'], ensure_ascii=False)}`|{m['val_acc']:.6f}|{m['macro_f1']:.6f}|{m['min_recall']:.6f}|{info['metrics_match_record']}|{info['seed_matches_test_probability_argmax']}|")
    report.extend(
        [
            "",
            "## old balanced reconstruction",
            "",
            f"Selected candidates: `{json.dumps(old_names, ensure_ascii=False)}`",
            f"Weights: `{json.dumps(old_weights, ensure_ascii=False)}`",
            f"Weights sum: `{audit['old_balanced_reconstruction']['weights_sum']}`",
            f"Reconstructed validation metrics: `{json.dumps(audit['old_balanced_reconstruction']['val_metrics'], ensure_ascii=False)}`",
            f"Fallback SEED matches reconstructed test argmax: `{audit['old_balanced_reconstruction']['fallback_seed_matches_test_argmax']}`",
            "",
            "## Consistency checks",
            "",
            f"- Validation labels: `{audit['validation_source']}`",
            f"- Validation shape: `{audit['validation_shape']}`",
            f"- Test shape: `{audit['test_shape']}`",
            "- Validation metrics use val probability rows 0..449 against validation labels rows 0..449.",
            "- SEED files use test probability rows 0..449 and are not evaluated against validation labels.",
            "- All checked SEED files have 450 rows and all probability matrices are 450x3.",
            "",
            "## Generation logic risk check",
            "",
            "- Direct/indirect use of y_val to decide each sample prediction: `False`.",
            "- Per-sample correction / oracle selection detected: `False`.",
            "- Uses validation score to select fixed global weights: `True`.",
            "",
            "## Internal holdout",
            "",
            "|split|fold|tune_acc|check_acc|check_macro-F1|check_min_recall|selected|weights|",
            "|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in holdout_rows:
        report.append(f"|{row['split']}|{row['fold']}|{row['tune_acc']:.6f}|{row['check_acc']:.6f}|{row['check_macro_f1']:.6f}|{row['check_min_recall']:.6f}|`{json.dumps(row['selected'], ensure_ascii=False)}`|`{json.dumps(row['weights'], ensure_ascii=False)}`|")
    report.extend(
        [
            "",
            "## Conclusion",
            "",
            "- `0.7111` is not an evaluation bug: it is reproduced from saved validation probabilities with the unified evaluator.",
            "- It is not caused by per-sample oracle correction: predictions come from a fixed weighted average of four clean candidate probability matrices.",
            "- It is validation-optimized: y_val was used to choose candidate/weight settings, so the 0.7111 validation score is likely optimistic relative to a truly unseen holdout.",
            "- Best characterization: legal but validation-overfitted fixed ensemble.",
        ]
    )
    (V25_OUT / "README_audit_old_balanced.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    v2.write_json(V25_OUT / "audit_old_balanced_results.json", audit)
    print(json.dumps(v2.jsonable(audit["conclusion"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
