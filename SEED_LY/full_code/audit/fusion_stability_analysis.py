#!/usr/bin/env python3
"""Fusion stability checks for clean_feature_block_fusion_v2 recovered outputs.

No feature models are retrained. No external data, order-aware/source matching,
or test labels are used.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

import car_block_feature_refinement as base
import clean_feature_block_fusion_v2 as v2


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "course_project" / "SEED"
OUT_DIR = ROOT / "outputs_experiments" / "clean_feature_block_fusion_v2_20260522_085528"
RNG = np.random.default_rng(20260522)
ALPHAS = np.linspace(0.05, 0.95, 19)


PoolItem = Tuple[str, np.ndarray, np.ndarray, Dict[str, Any]]


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def falseish(v: Any) -> bool:
    if v in (False, None, "", 0):
        return True
    return str(v).strip().lower() in {"false", "0", "none", "null"}


def parse_float(v: Any, default: float = float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


def clean_forbidden_check(row: Dict[str, Any]) -> str | None:
    payload = {
        "name": row.get("name"),
        "source": row.get("source"),
        "preprocess": row.get("preprocess"),
        "feature_family": row.get("feature_family"),
        "model": row.get("model"),
        "smoothing": row.get("smoothing"),
        "candidate_type": row.get("candidate_type"),
        "origin": row.get("origin"),
    }
    return v2.contains_forbidden(payload)


def evaluate_pred(y: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    pred = np.asarray(pred, dtype=int)
    acc = float(np.mean(pred == y))
    recalls = []
    f1s = []
    cm = np.zeros((3, 3), dtype=int)
    for true, got in zip(y, pred):
        if 0 <= true < 3 and 0 <= got < 3:
            cm[int(true), int(got)] += 1
    for c in range(3):
        tp = cm[c, c]
        fn = cm[c].sum() - tp
        fp = cm[:, c].sum() - tp
        rec = tp / (tp + fn) if tp + fn else 0.0
        prec = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        recalls.append(rec)
        f1s.append(f1)
    return {
        "val_acc": acc,
        "macro_f1": float(np.mean(f1s)),
        "min_recall": float(np.min(recalls)),
        "per_class_recall": {str(i): float(v) for i, v in enumerate(recalls)},
        "confusion_matrix": cm.tolist(),
        "prediction_distribution": {str(k): int(v) for k, v in Counter(pred.tolist()).items()},
    }


def metric_score(m: Dict[str, Any]) -> float:
    return float(m["val_acc"] + 0.2 * m["macro_f1"] + 0.1 * m["min_recall"])


def evaluate_prob_on_indices(y: np.ndarray, prob: np.ndarray, idx: np.ndarray | None = None) -> Dict[str, Any]:
    if idx is None:
        return evaluate_pred(y, v2.normalize_prob(prob).argmax(axis=1))
    return evaluate_pred(y[idx], v2.normalize_prob(prob[idx]).argmax(axis=1))


def row_sort_key(item: PoolItem, idx: np.ndarray | None = None, y: np.ndarray | None = None) -> Tuple[float, float, float]:
    if idx is None:
        row = item[3]
        return (parse_float(row.get("val_acc")), parse_float(row.get("macro_f1")), parse_float(row.get("score")))
    assert y is not None
    m = evaluate_prob_on_indices(y, item[1], idx)
    return (m["val_acc"], m["macro_f1"], metric_score(m))


def restore_pool(yv: np.ndarray) -> List[PoolItem]:
    rows = read_csv_rows(OUT_DIR / "all_clean_candidates_summary.csv")
    restored: List[PoolItem] = []
    for row in rows:
        name = row.get("name", "")
        if row.get("risk_level") != "clean-low":
            continue
        if not falseish(row.get("uses_order_prior")) or not falseish(row.get("uses_source_matching")):
            continue
        if clean_forbidden_check(row):
            continue
        vp = OUT_DIR / "all_val_probs" / f"{v2.safe_name(name)}.npy"
        tp = OUT_DIR / "all_test_probs" / f"{v2.safe_name(name)}.npy"
        if not vp.exists() or not tp.exists():
            continue
        pv = v2.normalize_prob(np.load(vp))
        pte = v2.normalize_prob(np.load(tp))
        metrics = evaluate_prob_on_indices(yv, pv)
        meta = {**row, **metrics, "score": metric_score(metrics)}
        restored.append((name, pv, pte, meta))

    deduped: List[PoolItem] = []
    seen = set()
    for item in sorted(restored, key=lambda x: row_sort_key(x), reverse=True):
        key = (np.round(item[1], 8).tobytes(), np.round(item[2], 8).tobytes())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def greedy_select(
    pool: Sequence[PoolItem],
    y: np.ndarray,
    idx_select: np.ndarray | None = None,
    search_top_n: int | None = None,
    max_selected: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if not pool:
        raise ValueError("greedy_select requires non-empty pool")
    ranked = sorted(pool, key=lambda x: row_sort_key(x, idx_select, y), reverse=True)
    if search_top_n is not None:
        ranked = ranked[:search_top_n]
    selected = [ranked[0]]
    cur_v = ranked[0][1].copy()
    cur_t = ranked[0][2].copy()
    cur_m = evaluate_prob_on_indices(y, cur_v, idx_select)
    steps = [{"name": ranked[0][0], "alpha_previous": 1.0, "metrics": cur_m}]

    improved = True
    while improved and (max_selected is None or len(selected) < max_selected):
        improved = False
        best = None
        for cand in ranked:
            if any(cand[0] == s[0] for s in selected):
                continue
            for alpha in ALPHAS:
                pv = v2.normalize_prob(alpha * cur_v + (1 - alpha) * cand[1])
                m = evaluate_prob_on_indices(y, pv, idx_select)
                if metric_score(m) > metric_score(cur_m) + 1e-10:
                    best = (cand, float(alpha), pv, v2.normalize_prob(alpha * cur_t + (1 - alpha) * cand[2]), m)
                    cur_m = m
                    improved = True
        if best is not None:
            selected.append(best[0])
            cur_v, cur_t = best[2], best[3]
            steps.append({"name": best[0][0], "alpha_previous": best[1], "metrics": best[4]})
    return cur_v, cur_t, {
        "selected": [x[0] for x in selected],
        "steps": steps,
        "search_top_n": search_top_n,
        "max_selected": max_selected,
    }


def effective_weights(steps: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for i, step in enumerate(steps):
        name = step["name"]
        alpha = float(step.get("alpha_previous", 1.0))
        if i == 0:
            weights[name] = 1.0
        else:
            for key in list(weights):
                weights[key] *= alpha
            weights[name] = 1.0 - alpha
    return weights


def write_seed_for_candidate(name: str, pte: np.ndarray) -> str:
    path = OUT_DIR / f"{name}_SEED.txt"
    v2.write_seed(path, pte.argmax(axis=1))
    return str(path)


def candidate_summary(name: str, pv: np.ndarray, pte: np.ndarray, yv: np.ndarray, meta: Dict[str, Any]) -> Dict[str, Any]:
    m = evaluate_prob_on_indices(yv, pv)
    test_dist = {str(k): int(v) for k, v in Counter(pte.argmax(axis=1).tolist()).items()}
    seed_path = write_seed_for_candidate(name, pte)
    return {
        "candidate": name,
        "val_acc": m["val_acc"],
        "macro_f1": m["macro_f1"],
        "min_recall": m["min_recall"],
        "confusion_matrix": m["confusion_matrix"],
        "val_prediction_distribution": m["prediction_distribution"],
        "test_prediction_distribution": test_dist,
        "seed_path": seed_path,
        **meta,
    }


def bootstrap_rows(candidates: Dict[str, Tuple[np.ndarray, np.ndarray]], yv: np.ndarray, n: int = 100) -> List[Dict[str, Any]]:
    rows = []
    for cand_name, (pv, _) in candidates.items():
        accs, f1s, mins = [], [], []
        for b in range(n):
            idx = RNG.integers(0, len(yv), size=len(yv))
            m = evaluate_prob_on_indices(yv, pv, idx)
            accs.append(m["val_acc"])
            f1s.append(m["macro_f1"])
            mins.append(m["min_recall"])
        rows.append(
            {
                "candidate": cand_name,
                "bootstrap_n": n,
                "acc_mean": float(np.mean(accs)),
                "acc_std": float(np.std(accs, ddof=1)),
                "macro_f1_mean": float(np.mean(f1s)),
                "macro_f1_std": float(np.std(f1s, ddof=1)),
                "min_recall_mean": float(np.mean(mins)),
                "min_recall_std": float(np.std(mins, ddof=1)),
            }
        )
    return rows


def method_fit(
    method: str,
    pool: Sequence[PoolItem],
    yv: np.ndarray,
    idx_select: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if method == "best_single":
        best = max(pool, key=lambda x: row_sort_key(x, idx_select, yv))
        return best[1], {"selected": [best[0]], "steps": [{"name": best[0], "alpha_previous": 1.0, "metrics": evaluate_prob_on_indices(yv, best[1], idx_select)}]}
    if method == "greedy_fusion_top5":
        pv, _, meta = greedy_select(pool, yv, idx_select=idx_select, search_top_n=5)
        return pv, meta
    if method == "greedy_fusion_top10":
        pv, _, meta = greedy_select(pool, yv, idx_select=idx_select, search_top_n=10)
        return pv, meta
    if method == "greedy_fusion_unlimited":
        pv, _, meta = greedy_select(pool, yv, idx_select=idx_select, search_top_n=None)
        return pv, meta
    raise ValueError(method)


def split_half_rows(pool: Sequence[PoolItem], yv: np.ndarray, split_name: str, a: np.ndarray, b: np.ndarray) -> List[Dict[str, Any]]:
    rows = []
    methods = ["best_single", "greedy_fusion_top5", "greedy_fusion_top10", "greedy_fusion_unlimited"]
    for fold, (sel, ev) in enumerate([(a, b), (b, a)], start=1):
        for method in methods:
            pv, meta = method_fit(method, pool, yv, sel)
            m_sel = evaluate_prob_on_indices(yv, pv, sel)
            m_ev = evaluate_prob_on_indices(yv, pv, ev)
            rows.append(
                {
                    "split": split_name,
                    "fold": fold,
                    "method": method,
                    "select_size": int(len(sel)),
                    "eval_size": int(len(ev)),
                    "selected_count": len(meta.get("selected", [])),
                    "selected_names": meta.get("selected", []),
                    "select_acc": m_sel["val_acc"],
                    "select_macro_f1": m_sel["macro_f1"],
                    "select_min_recall": m_sel["min_recall"],
                    "eval_acc": m_ev["val_acc"],
                    "eval_macro_f1": m_ev["macro_f1"],
                    "eval_min_recall": m_ev["min_recall"],
                    "eval_confusion_matrix": m_ev["confusion_matrix"],
                    "eval_prediction_distribution": m_ev["prediction_distribution"],
                }
            )
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    v2.write_csv(path, rows)


def markdown_table(rows: Sequence[Dict[str, Any]], cols: Sequence[str]) -> str:
    out = ["|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        vals = []
        for c in cols:
            v = row.get(c, "")
            if isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v).replace("\n", " "))
        out.append("|" + "|".join(vals) + "|")
    return "\n".join(out)


def main() -> None:
    _, yv = base.load_xy(DATA / "val.h5", True)
    pool = restore_pool(yv)
    if not pool:
        raise RuntimeError("No clean pool restored from saved probabilities.")
    fixed = json.loads((OUT_DIR / "fixed_run_results.json").read_text(encoding="utf-8"))
    fixed_best = fixed["fixed_best_clean"]

    by_name = {name: item for item in pool for name in [item[0]]}
    selected_detail_rows = []
    for i, step in enumerate(fixed_best["steps"], start=1):
        name = step["name"]
        item = by_name[name]
        row = item[3]
        selected_detail_rows.append(
            {
                "order": i,
                "name": name,
                "alpha_previous": step.get("alpha_previous"),
                "effective_weight": effective_weights(fixed_best["steps"]).get(name),
                "val_acc": step["metrics"]["val_acc"],
                "macro_f1": step["metrics"]["macro_f1"],
                "min_recall": step["metrics"]["min_recall"],
                "feature_family": row.get("feature_family"),
                "model": row.get("model"),
                "smoothing": row.get("smoothing"),
                "block_size": row.get("block_size"),
                "train_block_size": row.get("train_block_size"),
                "val_test_block_size": row.get("val_test_block_size"),
            }
        )

    # Conservative candidates.
    best_single = pool[0]
    best_macro = max(pool, key=lambda x: (parse_float(x[3].get("macro_f1")), parse_float(x[3].get("val_acc")), parse_float(x[3].get("score"))))
    eligible_balanced = [x for x in pool if parse_float(x[3].get("val_acc")) >= parse_float(best_single[3].get("val_acc")) - 0.03]
    balanced = min(
        eligible_balanced,
        key=lambda x: (
            sum(abs(Counter(x[2].argmax(axis=1).tolist()).get(c, 0) - 150) for c in range(3)),
            -parse_float(x[3].get("val_acc")),
            -parse_float(x[3].get("macro_f1")),
        ),
    )

    candidates: Dict[str, Tuple[np.ndarray, np.ndarray, Dict[str, Any]]] = {
        "best_single": (best_single[1], best_single[2], {"kind": "single", "source_name": best_single[0]}),
        "best_macro_f1_candidate": (best_macro[1], best_macro[2], {"kind": "single", "source_name": best_macro[0]}),
        "balanced_test_distribution_candidate": (balanced[1], balanced[2], {"kind": "single", "source_name": balanced[0]}),
    }
    for label, top_n in [
        ("greedy_fusion_unlimited", None),
        ("greedy_fusion_top3", 3),
        ("greedy_fusion_top5", 5),
        ("greedy_fusion_top10", 10),
        ("greedy_fusion_top20", 20),
    ]:
        pv, pte, meta = greedy_select(pool, yv, search_top_n=top_n)
        candidates[label] = (pv, pte, {"kind": "fusion", **meta, "effective_weights": effective_weights(meta["steps"])})

    comparison_rows = []
    for name in ["best_single", "greedy_fusion_unlimited", "greedy_fusion_top3", "greedy_fusion_top5", "greedy_fusion_top10", "greedy_fusion_top20", "best_macro_f1_candidate", "balanced_test_distribution_candidate"]:
        pv, pte, meta = candidates[name]
        comparison_rows.append(candidate_summary(name, pv, pte, yv, meta))
    write_csv(OUT_DIR / "fusion_candidate_comparison.csv", comparison_rows)

    stability_candidates = {
        name: (candidates[name][0], candidates[name][1])
        for name in ["best_single", "greedy_fusion_top5", "greedy_fusion_top10", "greedy_fusion_unlimited"]
    }
    bootstrap = bootstrap_rows(stability_candidates, yv, 100)
    write_csv(OUT_DIR / "fusion_stability_bootstrap.csv", bootstrap)

    shuffled = np.arange(len(yv))
    RNG.shuffle(shuffled)
    half_a, half_b = np.sort(shuffled[: len(yv) // 2]), np.sort(shuffled[len(yv) // 2 :])
    split_rows = split_half_rows(pool, yv, "random_half", half_a, half_b)
    block_ids = np.arange(len(yv)) // 10
    even_blocks = np.where(block_ids % 2 == 0)[0]
    odd_blocks = np.where(block_ids % 2 == 1)[0]
    split_rows.extend(split_half_rows(pool, yv, "block10_even_odd", even_blocks, odd_blocks))
    write_csv(OUT_DIR / "fusion_split_half_results.csv", split_rows)

    split_eval = {}
    for method in ["best_single", "greedy_fusion_top5", "greedy_fusion_top10", "greedy_fusion_unlimited"]:
        vals = [r["eval_acc"] for r in split_rows if r["method"] == method]
        split_eval[method] = float(np.mean(vals))

    def mean_eval(method: str, split: str) -> float:
        vals = [r["eval_acc"] for r in split_rows if r["method"] == method and r["split"] == split]
        return float(np.mean(vals))

    random_eval = {m: mean_eval(m, "random_half") for m in split_eval}
    block_eval = {m: mean_eval(m, "block10_even_odd") for m in split_eval}
    unlimited_stable = (
        random_eval["greedy_fusion_unlimited"] > random_eval["best_single"] + 1e-12
        and block_eval["greedy_fusion_unlimited"] > block_eval["best_single"] + 1e-12
    )
    if unlimited_stable:
        recommended = "fixed_best_clean_low_risk_SEED.txt"
        rationale = "unlimited greedy 在 random half 与 block split 的平均评估 acc 均高于 best_single，因此推荐当前 fixed best clean fusion。"
    elif max(split_eval["greedy_fusion_top5"], split_eval["greedy_fusion_top10"]) > split_eval["best_single"] + 1e-12:
        recommended_method = "greedy_fusion_top5" if split_eval["greedy_fusion_top5"] >= split_eval["greedy_fusion_top10"] else "greedy_fusion_top10"
        recommended = f"{recommended_method}_SEED.txt"
        rationale = f"unlimited greedy 的稳定性不足，但 {recommended_method} 的 split-half 平均评估优于 best_single。"
    else:
        recommended = "best_single_SEED.txt"
        rationale = "fusion 在 split-half 中不稳定优于 best_single，因此推荐 0.6222 best_single。"

    summary_md = [
        "# Fusion Candidate Summary",
        "",
        "所有候选均只使用当前 output_dir 中已有的 `all_val_probs` / `all_test_probs`，未重新训练模型，未使用 external、order-aware/source matching 或 test label。",
        "",
        "## fixed_clean_fusion_greedy_forward 解析",
        "",
        f"- selected candidate 数量：{len(selected_detail_rows)}",
        f"- 最终 confusion matrix：`{json.dumps(fixed_best['confusion_matrix'], ensure_ascii=False)}`",
        f"- validation prediction distribution：`{json.dumps(fixed_best['prediction_distribution'], ensure_ascii=False)}`",
        f"- test prediction distribution：`{json.dumps(fixed_best['test_prediction_distribution'], ensure_ascii=False)}`",
        "",
        markdown_table(selected_detail_rows, ["order", "name", "alpha_previous", "effective_weight", "val_acc", "macro_f1", "min_recall", "feature_family", "model", "smoothing", "block_size"]),
        "",
        "## 保守候选对比",
        "",
        markdown_table(comparison_rows, ["candidate", "val_acc", "macro_f1", "min_recall", "test_prediction_distribution", "seed_path"]),
    ]
    (OUT_DIR / "fusion_candidate_summary.md").write_text("\n".join(summary_md) + "\n", encoding="utf-8")

    stability_md = [
        "# Fusion Stability Report",
        "",
        "## Bootstrap Validation",
        "",
        markdown_table(bootstrap, ["candidate", "acc_mean", "acc_std", "macro_f1_mean", "macro_f1_std", "min_recall_mean", "min_recall_std"]),
        "",
        "## Split-Half / Block Split",
        "",
        markdown_table(split_rows, ["split", "fold", "method", "selected_count", "select_acc", "eval_acc", "eval_macro_f1", "eval_min_recall", "selected_names"]),
        "",
        "## 平均 eval acc",
        "",
        markdown_table([{"method": k, "mean_eval_acc": v} for k, v in split_eval.items()], ["method", "mean_eval_acc"]),
    ]
    (OUT_DIR / "fusion_stability_report.md").write_text("\n".join(stability_md) + "\n", encoding="utf-8")

    recommendation_md = [
        "# Final Submission Recommendation",
        "",
        f"推荐提交：`{recommended}`",
        "",
        rationale,
        "",
        "关键结果：",
        f"- 修复后 fixed best clean val_acc：{fixed_best['val_acc']}",
        f"- 修复后 fixed best clean macro-F1：{fixed_best['macro_f1']}",
        f"- 修复后 fixed best clean min_recall：{fixed_best['min_recall']}",
        f"- best_single val_acc：{comparison_rows[0]['val_acc']}",
        f"- greedy_fusion_unlimited val_acc：{next(r for r in comparison_rows if r['candidate'] == 'greedy_fusion_unlimited')['val_acc']}",
        f"- split-half 平均 eval acc：`{json.dumps(split_eval, ensure_ascii=False)}`",
        "",
        "约束确认：未重新训练 feature models，未引入 external，未使用 order-aware/source matching，未使用 test label。",
    ]
    (OUT_DIR / "final_submission_recommendation.md").write_text("\n".join(recommendation_md) + "\n", encoding="utf-8")

    print(json.dumps({
        "pool_size": len(pool),
        "selected_detail_rows": selected_detail_rows,
        "comparison": comparison_rows,
        "bootstrap": bootstrap,
        "split_eval": split_eval,
        "recommended": recommended,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
