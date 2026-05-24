#!/usr/bin/env python3
"""v2.5 winner-family expansion and refined clean fusion.

This script focuses only on the four winner families from the current clean
fusion. It does not use external data, CRNN, raw matching, order-aware/source
matching, or test labels.
"""

from __future__ import annotations

import csv
import json
import math
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import car_block_feature_refinement as base
import clean_feature_block_fusion_v2 as v2
import regularized_clean_fusion_search as reg


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs_experiments"
DATA = ROOT / "course_project" / "SEED"
V2_OUT = OUTPUT_ROOT / "clean_feature_block_fusion_v2_20260522_085528"
OLD_REG_DIR = V2_OUT / "regularized_fusion_search"
SMOOTH_F1 = [5, 8, 10, 12, 15, 20]
SMOOTH_F2 = [10, 15, 20, 25]
SMOOTH_F3 = [5, 8, 10, 12, 15]
RNG = np.random.default_rng(20260522)

PoolItem = Tuple[str, np.ndarray, np.ndarray, Dict[str, Any]]


def ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def log(logs: List[str], msg: str) -> None:
    logs.append(f"- `{time.strftime('%H:%M:%S')}` {msg}")
    print(msg, flush=True)


def read_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def fit_stats_band_covpca(Xtr: np.ndarray, Xv: np.ndarray, Xte: np.ndarray, cache_dir: Path, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = [cache_dir / f"car_stats_band_covpca{n}_{split}.npy" for split in ("train", "val", "test")]
    meta_path = cache_dir / f"car_stats_band_covpca{n}_meta.json"
    if all(p.exists() for p in paths):
        Ftr, Fv, Fte = [np.load(p) for p in paths]
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return Ftr, Fv, Fte, meta
    stats_tr, stats_v, stats_te = base.channel_stats_features(Xtr), base.channel_stats_features(Xv), base.channel_stats_features(Xte)
    band_tr, band_v, band_te = base.band_features(Xtr), base.band_features(Xv), base.band_features(Xte)
    cov_tr, cov_v, cov_te = base.cov_corr_upper(Xtr), base.cov_corr_upper(Xv), base.cov_corr_upper(Xte)
    pca = PCA(n_components=n, random_state=7)
    ctr = pca.fit_transform(cov_tr)
    cv = pca.transform(cov_v)
    cte = pca.transform(cov_te)
    Ftr = np.concatenate([stats_tr, band_tr, ctr], axis=1).astype(np.float32)
    Fv = np.concatenate([stats_v, band_v, cv], axis=1).astype(np.float32)
    Fte = np.concatenate([stats_te, band_te, cte], axis=1).astype(np.float32)
    for arr, path in zip((Ftr, Fv, Fte), paths):
        np.save(path, arr)
    meta = {
        "preprocess": "common_average_reference",
        "feature_family": f"stats_band_covpca{n}",
        "families": ["channel_statistics", "frequency_band_features", "cov_corr_pca"],
        "pca_components": n,
        "pca_explained_variance_sum": float(pca.explained_variance_ratio_.sum()),
        "feature_shape": list(Ftr.shape),
    }
    meta_path.write_text(json.dumps(v2.jsonable(meta), ensure_ascii=False, indent=2), encoding="utf-8")
    return Ftr, Fv, Fte, meta


def add_smoothed(rows: List[Dict[str, Any]], pool: List[PoolItem], out_dir: Path, yv: np.ndarray, name: str, pv: np.ndarray, pte: np.ndarray, meta: Dict[str, Any], blocks: Sequence[int]) -> None:
    row = v2.add_candidate(rows, out_dir, name, "v25_clean_winner_expansion", pv, pte, yv, {**meta, "smoothing": "none"})
    pool.append((name, v2.normalize_prob(pv), v2.normalize_prob(pte), row))
    for bs in blocks:
        bpv = base.block_smooth(pv, bs)
        bte = base.block_smooth(pte, bs)
        bname = f"{name}__prob_block_smooth_{bs}"
        brow = v2.add_candidate(rows, out_dir, bname, "v25_clean_winner_expansion", bpv, bte, yv, {**meta, "smoothing": "prob_block_smooth", "block_size": bs})
        pool.append((bname, bpv, bte, brow))


def sorted_hgb_specs(limit: int = 48) -> List[Tuple[str, Any, Dict[str, Any]]]:
    specs = []
    for lr in [0.02, 0.03, 0.04, 0.05]:
        for it in [200, 300, 500]:
            for leaf in [15, 31, 63]:
                for l2 in [0.0, 0.001, 0.01, 0.03]:
                    dist = abs(math.log(lr / 0.03)) + abs(it - 200) / 300 + (0 if leaf == 31 else 0.4) + abs(math.log((l2 + 0.001) / (0.01 + 0.001)))
                    specs.append((dist, lr, it, leaf, l2))
    specs = sorted(specs, key=lambda x: x[0])[:limit]
    out = []
    for i, (_, lr, it, leaf, l2) in enumerate(specs):
        name = f"hgb_lr{str(lr).replace('.', '')}_i{it}_l{leaf}_l2{str(l2).replace('.', 'p')}"
        est = HistGradientBoostingClassifier(max_iter=it, learning_rate=lr, max_leaf_nodes=leaf, l2_regularization=l2, random_state=2500 + i)
        out.append((name, est, {"learning_rate": lr, "max_iter": it, "max_leaf_nodes": leaf, "l2_regularization": l2}))
    return out


def sorted_mlp_specs(limit: int = 12) -> List[Tuple[str, Any, Dict[str, Any]]]:
    specs = []
    for hidden in [64, 128, 256]:
        for alpha in [0.0001, 0.001, 0.003, 0.01]:
            for lr in [0.0005, 0.001, 0.003]:
                dist = abs(math.log(hidden / 128)) + abs(math.log(alpha / 0.001)) + abs(math.log(lr / 0.001))
                specs.append((dist, hidden, alpha, lr))
    specs = sorted(specs, key=lambda x: x[0])[:limit]
    out = []
    for i, (_, hidden, alpha, lr) in enumerate(specs):
        name = f"mlp_h{hidden}_a{str(alpha).replace('.', 'p')}_lr{str(lr).replace('.', 'p')}"
        est = MLPClassifier(hidden_layer_sizes=(hidden,), alpha=alpha, learning_rate_init=lr, early_stopping=True, max_iter=500, random_state=2600 + i)
        out.append((name, est, {"hidden": hidden, "alpha": alpha, "learning_rate_init": lr}))
    return out


def sorted_tree_specs(limit: int = 8) -> List[Tuple[str, Any, Dict[str, Any]]]:
    specs = []
    for model in ["rf", "et"]:
        for n in [500, 700, 1000]:
            for mf in ["sqrt", 0.3, 0.5]:
                for leaf in [1, 2, 3, 5]:
                    for cw in [None, "balanced"]:
                        dist = (0 if model == "rf" else 0.05) + abs(n - 700) / 500 + (0 if mf in ["sqrt", 0.3] else 0.2) + abs(leaf - 1) * 0.25 + (0 if cw == "balanced" else 0.15)
                        specs.append((dist, model, n, mf, leaf, cw))
    specs = sorted(specs, key=lambda x: x[0])[:limit]
    out = []
    for i, (_, model, n, mf, leaf, cw) in enumerate(specs):
        tag = f"{model}_n{n}_mf{str(mf).replace('.', 'p')}_l{leaf}_cw{cw or 'none'}"
        if model == "rf":
            est = RandomForestClassifier(n_estimators=n, max_features=mf, min_samples_leaf=leaf, class_weight=cw, random_state=2700 + i, n_jobs=-1)
        else:
            est = ExtraTreesClassifier(n_estimators=n, max_features=mf, min_samples_leaf=leaf, class_weight=cw, random_state=2700 + i, n_jobs=-1)
        out.append((tag, est, {"tree_model": model, "n_estimators": n, "max_features": mf, "min_samples_leaf": leaf, "class_weight": cw}))
    return out


def sorted_block_specs(limit: int = 24) -> List[Tuple[int, int, int, str, Any, Dict[str, Any]]]:
    specs = []
    for comp in [15, 20, 25, 30, 40]:
        for tr_bs in [5, 8, 10, 12, 15]:
            for ev_bs in [5, 8, 10, 12, 15]:
                for model in ["block_ET", "block_RF"]:
                    dist = abs(comp - 20) / 20 + abs(tr_bs - 10) / 10 + abs(ev_bs - 10) / 10 + (0 if model == "block_ET" else 0.15)
                    specs.append((dist, comp, tr_bs, ev_bs, model))
    specs = sorted(specs, key=lambda x: x[0])[:limit]
    out = []
    for i, (_, comp, tr_bs, ev_bs, model) in enumerate(specs):
        if model == "block_ET":
            est = ExtraTreesClassifier(n_estimators=700, min_samples_leaf=1, max_features="sqrt", class_weight="balanced", random_state=2800 + i, n_jobs=-1)
        else:
            est = RandomForestClassifier(n_estimators=700, min_samples_leaf=1, max_features="sqrt", class_weight="balanced", random_state=2800 + i, n_jobs=-1)
        out.append((comp, tr_bs, ev_bs, model, est, {"train_block_size": tr_bs, "val_test_block_size": ev_bs, "block_model": model}))
    return out


def restore_v25_pool(out_dir: Path, yv: np.ndarray) -> List[PoolItem]:
    pool = []
    for row in read_rows(out_dir / "v25_new_candidates_summary.csv"):
        name = row["name"]
        vp = out_dir / "all_val_probs" / f"{v2.safe_name(name)}.npy"
        tp = out_dir / "all_test_probs" / f"{v2.safe_name(name)}.npy"
        if not vp.exists() or not tp.exists():
            continue
        pv, pt = v2.normalize_prob(np.load(vp)), v2.normalize_prob(np.load(tp))
        m = reg.evaluate_prob(yv, pv)
        pool.append((f"v25__{name}", pv, pt, {**row, **m, "score": reg.score(m), "name": f"v25__{name}"}))
    return pool


def dedupe_pool(pool: Sequence[PoolItem]) -> List[PoolItem]:
    out = []
    seen = set()
    for item in sorted(pool, key=lambda x: (float(x[3]["val_acc"]), float(x[3]["macro_f1"]), float(x[3]["score"])), reverse=True):
        if item[1].shape != (450, 3) or item[2].shape != (450, 3):
            continue
        if reg.clean_forbidden_check(item[3]):
            continue
        key = (np.round(item[1], 8).tobytes(), np.round(item[2], 8).tobytes())
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def apply_temperature_bias_threshold(pv: np.ndarray, pt: np.ndarray, yv: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    base_v, base_t = v2.normalize_prob(pv), v2.normalize_prob(pt)
    best_v, best_t = base_v, base_t
    best_m = reg.evaluate_prob(yv, base_v)
    best_cfg = {"temperature": 1.0, "bias": [0, 0, 0], "threshold_adjust": [0, 0, 0], "mode": mode}
    logv = np.log(base_v + 1e-10)
    logt = np.log(base_t + 1e-10)
    for temp in [0.8, 0.9, 1.0, 1.1, 1.2]:
        for b0 in np.linspace(-0.18, 0.18, 7):
            for b1 in np.linspace(-0.18, 0.18, 7):
                for b2 in np.linspace(-0.18, 0.18, 7):
                    vv = base.softmax(logv / temp + np.array([b0, b1, b2]))
                    tt = base.softmax(logt / temp + np.array([b0, b1, b2]))
                    for adj2 in [0.0, 0.01, 0.02, -0.01]:
                        vva = vv.copy()
                        tta = tt.copy()
                        if adj2 != 0:
                            vva[:, 2] += adj2
                            tta[:, 2] += adj2
                            vva = v2.normalize_prob(vva)
                            tta = v2.normalize_prob(tta)
                        m = reg.evaluate_prob(yv, vva)
                        if mode == "macro":
                            obj = m["macro_f1"] + 0.2 * m["val_acc"] + 0.1 * m["min_recall"]
                        elif mode == "min_recall":
                            obj = m["min_recall"] + 0.2 * m["macro_f1"] + 0.1 * m["val_acc"]
                        else:
                            obj = reg.score(m)
                        old_obj = best_m["macro_f1"] + 0.2 * best_m["val_acc"] + 0.1 * best_m["min_recall"] if mode == "macro" else (best_m["min_recall"] + 0.2 * best_m["macro_f1"] + 0.1 * best_m["val_acc"] if mode == "min_recall" else reg.score(best_m))
                        if obj > old_obj + 1e-10 and m["val_acc"] >= base_v.argmax(axis=1).astype(int).reshape(-1).shape[0] * 0:  # explicit no-op guard for readability
                            best_v, best_t, best_m = vva, tta, m
                            best_cfg = {"temperature": temp, "bias": [float(b0), float(b1), float(b2)], "threshold_adjust": [0.0, 0.0, float(adj2)], "mode": mode}
    return best_v, best_t, {"selected": [], "weights": {}, "calibration": best_cfg, "metrics": best_m}


def run_refined_fusion(out_dir: Path, combined: List[PoolItem], yv: np.ndarray, logs: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    fusion_dir = out_dir / "refined_fusion"
    fusion_dir.mkdir()
    reg.SEARCH_DIR = fusion_dir
    splits = reg.make_splits(len(yv))
    records: List[Dict[str, Any]] = []

    old_run = json.loads((OLD_REG_DIR / "regularized_fusion_run_results.json").read_text(encoding="utf-8"))
    old_balanced = old_run["balanced_candidate"]
    old_aggressive = old_run["aggressive_candidate"]
    old_names = old_balanced["selected_candidates"]
    old_weights = old_balanced["weights"]

    def rec(name: str, pv: np.ndarray, pt: np.ndarray, meta: Dict[str, Any]) -> Dict[str, Any]:
        r = reg.candidate_record(name, pv, pt, yv, meta, splits)
        records.append(r)
        log(logs, f"fusion {name}: acc={r['val_acc']:.4f} macro={r['macro_f1']:.4f} minrec={r['min_recall']:.4f} test={r['test_prediction_distribution']}")
        return r

    # Old references copied into v25 stability and recommendation space.
    old_bal_v = np.load(OLD_REG_DIR / "best_min_recall_fusion_SEED.txt") if False else None
    by_name = {x[0]: x for x in combined}
    old_items = []
    for name in old_names:
        old_items.append(by_name[name])
    old_bv, old_bt = reg.fuse_weighted(combined, old_weights)
    rec("old_balanced_reference", old_bv, old_bt, {"selected": old_names, "weights": old_weights, "strategy": "old_balanced_reference"})
    old_av, old_at = reg.fuse_weighted(combined, old_aggressive["weights"])
    rec("old_aggressive_reference", old_av, old_at, {"selected": old_aggressive["selected_candidates"], "weights": old_aggressive["weights"], "strategy": "old_aggressive_reference"})

    for k in [3, 4, 5, 6, 8, 10]:
        pv, pt, meta = reg.greedy_full(combined, yv, max_selected=k)
        meta.update({"strategy": "greedy_limited_k", "k": k})
        rec(f"v25_greedy_limited_k{k}", pv, pt, meta)

    pv, pt, meta = reg.stability_aware_greedy(combined, yv, splits)
    meta["strategy"] = "stability_aware_greedy"
    rec("v25_stability_aware_greedy", pv, pt, meta)

    for thr in [0.90, 0.95, 0.98]:
        pv, pt, meta = reg.greedy_full(combined, yv, max_selected=8, corr_threshold=thr)
        meta.update({"strategy": "diversity_aware_fusion", "corr_threshold": thr})
        rec(f"v25_diversity_corr_{str(thr).replace('.', 'p')}", pv, pt, meta)

    # Weight refinement around balanced.
    for sigma in [0.015, 0.03, 0.06]:
        best = None
        best_obj = -1e9
        names = old_names
        w0 = np.array([old_weights[n] for n in names], dtype=float)
        for _ in range(20000):
            w = np.clip(w0 + RNG.normal(0, sigma, size=len(w0)), 0.0, None)
            if w.sum() <= 0:
                continue
            w /= w.sum()
            if np.max(w) > 0.56:
                continue
            weights = {n: float(x) for n, x in zip(names, w)}
            pv, pt = reg.fuse_weighted(combined, weights)
            m = reg.evaluate_prob(yv, pv)
            obj = reg.score(m)
            if obj > best_obj:
                best_obj = obj
                best = (pv, pt, weights)
        if best:
            rec(f"v25_weight_gaussian_sigma_{str(sigma).replace('.', 'p')}", best[0], best[1], {"selected": names, "weights": best[2], "strategy": "weight_refinement_around_balanced", "sigma": sigma})

    # Dirichlet around balanced + distribution sanity.
    for min_c2 in [50, 60, 70]:
        pv, pt, meta = reg.random_weight_search(
            combined,
            old_names,
            yv,
            "score",
            n=30000,
            constraints={"center_weights": old_weights, "concentration": 120.0, "max_any_weight": 0.56, "min_class2_test": min_c2, "max_class1_test": 230, "max_class0_test": 230},
        )
        meta.update({"strategy": "distribution_sanity_variant", "min_class2_test": min_c2})
        rec(f"v25_distribution_min_c2_{min_c2}", pv, pt, meta)

    # Temperature/bias/threshold search around old balanced.
    for mode, cname in [("score", "v25_class_bias_score_best"), ("macro", "v25_macro_f1_best"), ("min_recall", "v25_min_recall_best")]:
        pv, pt, meta = apply_temperature_bias_threshold(old_bv, old_bt, yv, mode)
        meta.update({"selected": old_names, "weights": old_weights, "strategy": "class_bias_temperature_threshold_search"})
        rec(cname, pv, pt, meta)

    ranked = sorted(records, key=lambda r: (r["val_acc"], r["macro_f1"], r["min_recall"], r["combined_split_eval_acc_mean"]), reverse=True)
    v2.write_csv(out_dir / "v25_fusion_candidates.csv", ranked)

    best_single = max([x[3] for x in combined if str(x[0]).startswith("v25__")], key=lambda r: (float(r["val_acc"]), float(r["macro_f1"]), float(r["score"])))
    best_single_item = next(x for x in combined if x[0] == best_single["name"])
    best_single_record = rec("v25_best_single", best_single_item[1], best_single_item[2], {"selected": [best_single_item[0]], "weights": {best_single_item[0]: 1.0}, "strategy": "v25_best_single"})

    old_bal = next(r for r in records if r["candidate_name"] == "old_balanced_reference")
    candidates_by_name = {r["candidate_name"]: r for r in records}
    plus_pool = [r for r in records if r["candidate_name"] not in {"old_balanced_reference", "old_aggressive_reference", "v25_best_single"}]
    better_stable = [r for r in plus_pool if r["val_acc"] > old_bal["val_acc"] + 1e-12 and r["combined_split_eval_acc_mean"] >= old_bal["combined_split_eval_acc_mean"] - 0.01]
    balanced_plus = max(better_stable or plus_pool, key=lambda r: (r["val_acc"], r["combined_split_eval_acc_mean"], r["macro_f1"]))
    aggressive_plus = max(plus_pool, key=lambda r: (r["val_acc"], r["macro_f1"], r["score"]))
    macro_f1_best = max(plus_pool, key=lambda r: (r["macro_f1"], r["val_acc"], r["min_recall"]))
    min_recall_best = max(plus_pool, key=lambda r: (r["min_recall"], r["macro_f1"], r["val_acc"]))
    safe_pool = [r for r in plus_pool if int(r["test_prediction_distribution"].get("2", 0)) >= 70 and r["combined_split_eval_acc_mean"] >= old_bal["combined_split_eval_acc_mean"] - 0.05]
    safest = max(safe_pool or plus_pool, key=lambda r: (r["combined_split_eval_acc_mean"], r["min_recall"], r["val_acc"]))
    picks = {"safest": safest, "balanced_plus": balanced_plus, "aggressive_plus": aggressive_plus, "macro_f1_best": macro_f1_best, "min_recall_best": min_recall_best, "old_balanced": old_bal, "old_aggressive": next(r for r in records if r["candidate_name"] == "old_aggressive_reference"), "v25_best_single": best_single_record}

    for label, src_rec in [("safe_v25", safest), ("balanced_v25", balanced_plus), ("aggressive_v25", aggressive_plus)]:
        Path(out_dir / f"{label}_SEED.txt").write_text(Path(src_rec["SEED_txt_path"]).read_text(encoding="utf-8"), encoding="utf-8")
    Path(out_dir / "fallback_old_balanced_SEED.txt").write_text((OLD_REG_DIR / "balanced_SEED.txt").read_text(encoding="utf-8"), encoding="utf-8")

    v2.write_csv(out_dir / "v25_stability_bootstrap.csv", [
        {k: r[k] for k in ["candidate_name", "bootstrap_acc_mean", "bootstrap_acc_std", "bootstrap_macro_f1_mean", "bootstrap_macro_f1_std", "bootstrap_min_recall_mean", "bootstrap_min_recall_std"]}
        for r in [old_bal, picks["old_aggressive"], best_single_record, safest, balanced_plus, aggressive_plus, macro_f1_best]
    ])
    split_rows = []
    splits = reg.make_splits(len(yv))
    for r in [old_bal, picks["old_aggressive"], best_single_record, safest, balanced_plus, aggressive_plus, macro_f1_best]:
        pv = np.load(Path(r["SEED_txt_path"]).with_name(Path(r["SEED_txt_path"]).name.replace("_SEED.txt", "_val_prob.npy"))) if False else None
        # Recompute from saved candidate seed path is impossible; use candidate record split summaries.
        split_rows.append({
            "candidate_name": r["candidate_name"],
            "random_half_eval_acc_mean": r["random_half_eval_acc_mean"],
            "block10_even_odd_eval_acc_mean": r["block10_even_odd_eval_acc_mean"],
            "combined_split_eval_acc_mean": r["combined_split_eval_acc_mean"],
            "random_half_eval_acc_min": r["random_half_eval_acc_min"],
            "block10_even_odd_eval_acc_min": r["block10_even_odd_eval_acc_min"],
        })
    v2.write_csv(out_dir / "v25_split_half_results.csv", split_rows)

    md = [
        "# v25 Fusion Summary",
        "",
        "Winner-family expansion only; no external, CRNN, raw matching, order-aware/source matching, or test labels.",
        "",
        "## Recommended fusion candidates",
        "",
        reg.md_table([{"tier": k, **v} for k, v in picks.items()], ["tier", "candidate_name", "val_acc", "macro_f1", "min_recall", "test_prediction_distribution", "combined_split_eval_acc_mean", "SEED_txt_path"]),
    ]
    (out_dir / "v25_fusion_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    stab_md = [
        "# v25 Stability Report",
        "",
        reg.md_table(split_rows, ["candidate_name", "random_half_eval_acc_mean", "block10_even_odd_eval_acc_mean", "combined_split_eval_acc_mean", "random_half_eval_acc_min", "block10_even_odd_eval_acc_min"]),
    ]
    (out_dir / "v25_stability_report.md").write_text("\n".join(stab_md) + "\n", encoding="utf-8")
    return ranked, picks


def main() -> None:
    out_dir = OUTPUT_ROOT / f"clean_feature_block_fusion_v25_{ts()}"
    out_dir.mkdir(parents=True, exist_ok=False)
    for sub in ["all_val_probs", "all_test_probs", "feature_cache"]:
        (out_dir / sub).mkdir()
    logs = ["# Clean feature/block fusion v2.5 winner expansion log", ""]
    rows: List[Dict[str, Any]] = []
    pool: List[PoolItem] = []
    results: Dict[str, Any] = {"status": "running", "output_dir": str(out_dir), "errors": []}
    config = {
        "scope": "winner_family_expansion",
        "no_external": True,
        "no_crnn": True,
        "no_raw_matching": True,
        "no_test_label": True,
        "no_order_aware": True,
        "no_source_matching": True,
        "v2_input_dir": str(V2_OUT),
    }
    v2.write_json(out_dir / "config.json", config)
    try:
        Xtr, ytr = base.load_xy(DATA / "train.h5", True)
        Xv, yv = base.load_xy(DATA / "val.h5", True)
        Xte, _ = base.load_xy(DATA / "test_x_only.h5", False)
        variants = base.preprocess_variants(Xtr, Xv, Xte)
        car_tr, car_v, car_te = variants["common_average_reference"]
        cpz_tr, cpz_v, cpz_te = variants["car_plus_per_sample_channel_zscore"]

        cov_features = {}
        for comp in [10, 15, 20, 25, 30, 35, 40, 50]:
            log(logs, f"Building CAR stats_band_covpca{comp}")
            cov_features[comp] = fit_stats_band_covpca(car_tr, car_v, car_te, out_dir / "feature_cache", comp)
        log(logs, "Building car_plus_per_sample_channel_zscore stats_band")
        cpz_stats_band = (
            np.concatenate([base.channel_stats_features(cpz_tr), base.band_features(cpz_tr)], axis=1).astype(np.float32),
            np.concatenate([base.channel_stats_features(cpz_v), base.band_features(cpz_v)], axis=1).astype(np.float32),
            np.concatenate([base.channel_stats_features(cpz_te), base.band_features(cpz_te)], axis=1).astype(np.float32),
            {"preprocess": "car_plus_per_sample_channel_zscore", "feature_family": "stats_band", "families": ["channel_statistics", "frequency_band_features"]},
        )
        log(logs, "Building common_average_reference channel_stats")
        car_channel_stats = (
            base.channel_stats_features(car_tr),
            base.channel_stats_features(car_v),
            base.channel_stats_features(car_te),
            {"preprocess": "common_average_reference", "feature_family": "channel_stats", "families": ["channel_statistics"]},
        )

        # Family 1.
        hgb_specs = sorted_hgb_specs(48)
        comp_order = [20, 25, 15, 30, 35, 10, 40, 50]
        f1_pairs = [(comp_order[i % len(comp_order)], hgb_specs[i]) for i in range(len(hgb_specs))]
        for comp, (model_name, est, mmeta) in f1_pairs:
            try:
                Ftr, Fv, Fte, fmeta = cov_features[comp]
                pv, pt = v2.train_model_prob(est, Ftr, ytr, Fv, Fte)
                name = f"v25_car_stats_band_covpca{comp}__{model_name}"
                add_smoothed(rows, pool, out_dir, yv, name, pv, pt, {**fmeta, **mmeta, "model": model_name, "candidate_type": "v25_family1_hgb_covpca"}, SMOOTH_F1)
            except Exception as exc:
                results["errors"].append({"stage": f"family1/{comp}/{model_name}", "error": repr(exc), "trace": traceback.format_exc()})
                log(logs, f"ERROR family1 {comp}/{model_name}: {repr(exc)}")
        log(logs, f"After family1 rows={len(rows)}")

        # Family 2.
        Ftr, Fv, Fte, fmeta = cpz_stats_band
        for model_name, est, mmeta in sorted_mlp_specs(12):
            try:
                pv, pt = v2.train_model_prob(est, Ftr, ytr, Fv, Fte)
                name = f"v25_cpz_stats_band__{model_name}"
                add_smoothed(rows, pool, out_dir, yv, name, pv, pt, {**fmeta, **mmeta, "model": model_name, "candidate_type": "v25_family2_mlp_stats_band"}, SMOOTH_F2)
            except Exception as exc:
                results["errors"].append({"stage": f"family2/{model_name}", "error": repr(exc), "trace": traceback.format_exc()})
                log(logs, f"ERROR family2 {model_name}: {repr(exc)}")
        log(logs, f"After family2 rows={len(rows)}")

        # Family 3.
        Ftr, Fv, Fte, fmeta = car_channel_stats
        for model_name, est, mmeta in sorted_tree_specs(8):
            try:
                pv, pt = v2.train_model_prob(est, Ftr, ytr, Fv, Fte)
                name = f"v25_car_channel_stats__{model_name}"
                add_smoothed(rows, pool, out_dir, yv, name, pv, pt, {**fmeta, **mmeta, "model": model_name, "candidate_type": "v25_family3_rf_et_channel_stats"}, SMOOTH_F3)
            except Exception as exc:
                results["errors"].append({"stage": f"family3/{model_name}", "error": repr(exc), "trace": traceback.format_exc()})
                log(logs, f"ERROR family3 {model_name}: {repr(exc)}")
        log(logs, f"After family3 rows={len(rows)}")

        # Family 4.
        for comp, tr_bs, ev_bs, model_name, est, mmeta in sorted_block_specs(24):
            try:
                Ftr, Fv, Fte, fmeta = cov_features[comp]
                pv, pt = v2.block_feature_candidates(Ftr, ytr, Fv, Fte, ev_bs, tr_bs, est)
                name = f"v25_car_stats_band_covpca{comp}__{model_name}__trainblock_{tr_bs}__evalblock_{ev_bs}"
                row = v2.add_candidate(rows, out_dir, name, "v25_clean_winner_expansion", pv, pt, yv, {**fmeta, **mmeta, "model": model_name, "candidate_type": "v25_family4_block_model", "smoothing": "block_classifier_broadcast"})
                pool.append((name, v2.normalize_prob(pv), v2.normalize_prob(pt), row))
            except Exception as exc:
                results["errors"].append({"stage": f"family4/{comp}/{tr_bs}/{ev_bs}/{model_name}", "error": repr(exc), "trace": traceback.format_exc()})
                log(logs, f"ERROR family4 {comp}/{tr_bs}/{ev_bs}/{model_name}: {repr(exc)}")
        log(logs, f"After family4 rows={len(rows)}")

        ranked_new = sorted(rows, key=lambda r: (r["val_acc"], r["macro_f1"], r["score"]), reverse=True)
        v2.write_csv(out_dir / "v25_new_candidates_summary.csv", ranked_new)
        v2.write_csv(out_dir / "v25_top50_single_candidates.csv", ranked_new[:50])

        v2_pool = reg.restore_pool(yv)
        v25_pool = restore_v25_pool(out_dir, yv)
        combined = dedupe_pool(v2_pool + v25_pool)
        v2.write_csv(out_dir / "v2_v25_combined_pool_summary.csv", [item[3] for item in combined])
        fusion_ranked, picks = run_refined_fusion(out_dir, combined, yv, logs)

        old_bal = picks["old_balanced"]
        balanced_plus = picks["balanced_plus"]
        recommend = balanced_plus if (balanced_plus["val_acc"] > old_bal["val_acc"] + 1e-12 and balanced_plus["combined_split_eval_acc_mean"] >= old_bal["combined_split_eval_acc_mean"] - 0.01) else old_bal
        final_md = [
            "# Final v25 Submission Recommendation",
            "",
            f"Recommended: `{recommend['candidate_name']}`",
            f"SEED: `{recommend['SEED_txt_path'] if recommend['candidate_name'] != 'old_balanced_reference' else str(out_dir / 'fallback_old_balanced_SEED.txt')}`",
            "",
            f"v25 found candidate above 0.7111: `{any(r['val_acc'] > 0.7111111111111111 + 1e-12 for r in fusion_ranked)}`",
            f"v25 best single: `{ranked_new[0]['name']}` val_acc `{ranked_new[0]['val_acc']}` macro-F1 `{ranked_new[0]['macro_f1']}` min_recall `{ranked_new[0]['min_recall']}`",
            f"old balanced val_acc: `{old_bal['val_acc']}` combined stability `{old_bal['combined_split_eval_acc_mean']}`",
            f"balanced_plus val_acc: `{balanced_plus['val_acc']}` combined stability `{balanced_plus['combined_split_eval_acc_mean']}`",
            "",
            "Constraints: no external, no CRNN, no raw matching, no test label, no order-aware/source matching.",
        ]
        (out_dir / "final_v25_submission_recommendation.md").write_text("\n".join(final_md) + "\n", encoding="utf-8")

        best_new = ranked_new[0]
        results.update({
            "status": "completed",
            "candidate_count": len(rows),
            "combined_pool_size": len(combined),
            "best_v25_single": best_new,
            "v25_single_exceeds_0_6222": bool(float(best_new["val_acc"]) > v2.CURRENT_CLEAN_BEST + 1e-12),
            "best_fusion": fusion_ranked[0],
            "picks": picks,
            "recommended": recommend,
            "v25_found_above_0_7111": bool(any(r["val_acc"] > 0.7111111111111111 + 1e-12 for r in fusion_ranked)),
        })
    except Exception as exc:
        results["status"] = "failed_partial"
        results["errors"].append({"stage": "main", "error": repr(exc), "trace": traceback.format_exc()})
        log(logs, f"ERROR: {repr(exc)}")
    finally:
        (out_dir / "experiment_log.md").write_text("\n".join(logs) + "\n", encoding="utf-8")
        v2.write_json(out_dir / "run_results.json", results)
        print(f"OUTPUT_DIR={out_dir}", flush=True)


if __name__ == "__main__":
    main()
