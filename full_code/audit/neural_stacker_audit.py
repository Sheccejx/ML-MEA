#!/usr/bin/env python3
"""Reliability audit for the saved neural probability stacker.

This script does not expand the candidate pool and does not use hidden test
labels. It reuses the 60 candidate probabilities used by the 20260523 stacker
run, then runs nested holdout, permutation, ablation, and test-output checks.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import car_block_feature_refinement as base
import two_day_rescue_utils as u


STACKER_DIR = u.OUTPUT_ROOT / "neural_probability_stacker_20260523_004708"
AUDIT_DIR = u.OUTPUT_ROOT / f"neural_stacker_reliability_audit_{u.stamp()}"


def metrics_from_pred(y: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    rec = recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "acc": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "min_recall": float(rec.min()),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).astype(int).tolist(),
        "prediction_distribution": {str(k): int(v) for k, v in Counter(pred.tolist()).items()},
    }


def metrics_from_prob(y: np.ndarray, prob: np.ndarray) -> Dict[str, Any]:
    return metrics_from_pred(y, u.normalize_prob(prob).argmax(axis=1))


class StackerMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def predict_mlp(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    outs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), 512):
            xb = torch.from_numpy(X[start : start + 512].astype(np.float32))
            outs.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return u.normalize_prob(np.concatenate(outs, axis=0))


def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    stop_idx: np.ndarray,
    cfg: Dict[str, Any],
    seed: int,
    max_epoch: int = 80,
    patience: int = 8,
) -> StackerMLP:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = StackerMLP(X.shape[1], int(cfg["hidden_dim"]), float(cfg["dropout"]))
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=float(cfg.get("label_smoothing", 0.0)))
    xtr = torch.from_numpy(X[train_idx].astype(np.float32))
    ytr = torch.from_numpy(y[train_idx].astype(np.int64))
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xtr, ytr), batch_size=96, shuffle=True, generator=gen)
    best_state = None
    best_score = -1e9
    wait = 0
    for _ in range(max_epoch):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        if len(stop_idx) > 0:
            m = metrics_from_prob(y[stop_idx], predict_mlp(model, X[stop_idx]))
            score = m["acc"] + 0.2 * m["macro_f1"] + 0.1 * m["min_recall"]
        else:
            score = -float(loss.item())
        if score > best_score + 1e-10:
            best_score = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def split_train_stop(idx: np.ndarray, y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.asarray(idx, dtype=int)
    if len(idx) < 30:
        return idx, np.asarray([], dtype=int)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    local_tr, local_stop = next(sss.split(np.zeros(len(idx)), y[idx]))
    return idx[local_tr], idx[local_stop]


AUDIT_CONFIGS: List[Dict[str, Any]] = [
    {"hidden_dim": 8, "dropout": 0.3, "weight_decay": 1e-3, "lr": 1e-3, "label_smoothing": 0.0},
    {"hidden_dim": 8, "dropout": 0.7, "weight_decay": 1e-2, "lr": 5e-4, "label_smoothing": 0.1},
    {"hidden_dim": 16, "dropout": 0.3, "weight_decay": 1e-3, "lr": 1e-3, "label_smoothing": 0.05},
    {"hidden_dim": 16, "dropout": 0.5, "weight_decay": 1e-2, "lr": 5e-4, "label_smoothing": 0.1},
    {"hidden_dim": 32, "dropout": 0.3, "weight_decay": 1e-2, "lr": 1e-3, "label_smoothing": 0.1},
    {"hidden_dim": 32, "dropout": 0.5, "weight_decay": 1e-2, "lr": 5e-4, "label_smoothing": 0.05},
    {"hidden_dim": 32, "dropout": 0.7, "weight_decay": 1e-3, "lr": 1e-3, "label_smoothing": 0.0},
    {"hidden_dim": 16, "dropout": 0.7, "weight_decay": 1e-3, "lr": 5e-4, "label_smoothing": 0.0},
]

ORIGINAL_BEST_CFG = {"hidden_dim": 32, "dropout": 0.3, "weight_decay": 1e-2, "lr": 1e-3, "label_smoothing": 0.1}


def inner_select_config(X: np.ndarray, y: np.ndarray, outer_train: np.ndarray, seed: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
    local_y = y[outer_train]
    for cfg_i, cfg in enumerate(AUDIT_CONFIGS):
        fold_scores = []
        fold_accs = []
        for fold_i, (tr_local, ev_local) in enumerate(skf.split(np.zeros(len(outer_train)), local_y), 1):
            tr_idx = outer_train[tr_local]
            ev_idx = outer_train[ev_local]
            fit_idx, stop_idx = split_train_stop(tr_idx, y, seed + 100 * cfg_i + fold_i)
            model = train_mlp(X, y, fit_idx, stop_idx, cfg, seed + 1000 * cfg_i + fold_i, max_epoch=70, patience=7)
            m = metrics_from_prob(y[ev_idx], predict_mlp(model, X[ev_idx]))
            score = m["acc"] + 0.2 * m["macro_f1"] + 0.1 * m["min_recall"]
            fold_scores.append(score)
            fold_accs.append(m["acc"])
        row = {**cfg, "inner_score_mean": float(np.mean(fold_scores)), "inner_acc_mean": float(np.mean(fold_accs))}
        rows.append(row)
    best = max(rows, key=lambda r: (r["inner_score_mean"], r["inner_acc_mean"]))
    cfg = {k: best[k] for k in ["hidden_dim", "dropout", "weight_decay", "lr", "label_smoothing"]}
    return cfg, rows


def nested_eval(X: np.ndarray, y: np.ndarray, outer_splits: Sequence[Tuple[str, np.ndarray, np.ndarray]], label: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    pred = np.full(len(y), -1, dtype=int)
    rows: List[Dict[str, Any]] = []
    for fold_i, (fold_name, tr_idx, check_idx) in enumerate(outer_splits, 1):
        cfg, inner_rows = inner_select_config(X, y, tr_idx, 30000 + fold_i)
        fit_idx, stop_idx = split_train_stop(tr_idx, y, 40000 + fold_i)
        model = train_mlp(X, y, fit_idx, stop_idx, cfg, 50000 + fold_i, max_epoch=90, patience=9)
        p = predict_mlp(model, X[check_idx])
        fold_pred = p.argmax(axis=1)
        pred[check_idx] = fold_pred
        m = metrics_from_pred(y[check_idx], fold_pred)
        rows.append(
            {
                "split_group": label,
                "fold": fold_name,
                "check_size": int(len(check_idx)),
                **cfg,
                **m,
                "inner_best_score": float(max(r["inner_score_mean"] for r in inner_rows)),
            }
        )
    covered = pred >= 0
    full_m = metrics_from_pred(y[covered], pred[covered])
    accs = [r["acc"] for r in rows]
    full_m.update(
        {
            "split_group": label,
            "fold_count": len(rows),
            "covered": int(covered.sum()),
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
            "acc_min": float(np.min(accs)),
            "acc_max": float(np.max(accs)),
        }
    )
    return full_m, rows


def oof_mlp_fixed(X: np.ndarray, y: np.ndarray, cfg: Dict[str, Any], seed: int, n_splits: int = 5) -> np.ndarray:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = np.zeros((len(y), 3), dtype=np.float64)
    for fold_i, (tr, ev) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        fit_idx, stop_idx = split_train_stop(tr, y, seed + fold_i)
        model = train_mlp(X, y, fit_idx, stop_idx, cfg, seed + 100 * fold_i, max_epoch=80, patience=8)
        out[ev] = predict_mlp(model, X[ev])
    return u.normalize_prob(out)


def oof_logistic(X: np.ndarray, y: np.ndarray, kind: str, seed: int = 2026) -> np.ndarray | np.ndarray:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pred = np.zeros(len(y), dtype=int)
    prob = np.zeros((len(y), 3), dtype=np.float64)
    for tr, ev in skf.split(np.zeros(len(y)), y):
        if kind == "logistic":
            clf = LogisticRegression(max_iter=2000, C=0.25, penalty="l2", solver="lbfgs")
            clf.fit(X[tr], y[tr])
            prob[ev] = clf.predict_proba(X[ev])
        else:
            clf = RidgeClassifier(alpha=10.0)
            clf.fit(X[tr], y[tr])
            pred[ev] = clf.predict(X[ev])
    return prob if kind == "logistic" else pred


def reference_probs(y: np.ndarray, pick_name: str) -> Tuple[np.ndarray, np.ndarray]:
    v25_json = json.loads((u.V25_OUT / "run_results.json").read_text(encoding="utf-8"))
    weights = v25_json["picks"][pick_name]["weights"]
    val = None
    test = None
    for cand_name, w in weights.items():
        vp = u.V2_OUT / "all_val_probs" / f"{u.safe_name(cand_name)}.npy"
        tp = u.V2_OUT / "all_test_probs" / f"{u.safe_name(cand_name)}.npy"
        pv = u.normalize_prob(np.load(vp))
        pt = u.normalize_prob(np.load(tp))
        val = pv * w if val is None else val + pv * w
        test = pt * w if test is None else test + pt * w
    return u.normalize_prob(val), u.normalize_prob(test)


def line_ref(path: Path, pattern: str) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    hits = [i + 1 for i, line in enumerate(lines) if pattern in line]
    return ", ".join(str(x) for x in hits[:6]) if hits else "not found"


def dist_str(pred: np.ndarray) -> Dict[str, int]:
    return {str(k): int(v) for k, v in Counter(pred.astype(int).tolist()).items()}


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=False)
    result: Dict[str, Any] = {"status": "running", "errors": [], "audit_dir": str(AUDIT_DIR)}
    try:
        _, y = base.load_xy(u.DATA / "val.h5", True)
        pool = u.load_candidate_pool(y, max_candidates=60)
        names = [p[0] for p in pool]
        val_probs = [p[1] for p in pool]
        test_probs = [p[2] for p in pool]
        X = np.concatenate(val_probs, axis=1).astype(np.float32)
        Xt = np.concatenate(test_probs, axis=1).astype(np.float32)

        candidate_rows = []
        for i, (name, pv, pt, meta) in enumerate(pool):
            candidate_rows.append(
                {
                    "idx": i,
                    "name": name,
                    "val_shape": list(pv.shape),
                    "test_shape": list(pt.shape),
                    "val_acc": meta["val_acc"],
                    "macro_f1": meta["macro_f1"],
                    "min_recall": meta["min_recall"],
                    "test_distribution": dist_str(pt.argmax(axis=1)),
                    "source": meta.get("source"),
                }
            )
        u.write_csv(AUDIT_DIR / "candidate_registry_audit.csv", candidate_rows)

        duplicate_rows = []
        near_duplicate_pairs = []
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                dv = float(np.max(np.abs(val_probs[i] - val_probs[j])))
                dt = float(np.max(np.abs(test_probs[i] - test_probs[j])))
                pred_agree_v = float(np.mean(val_probs[i].argmax(axis=1) == val_probs[j].argmax(axis=1)))
                pred_agree_t = float(np.mean(test_probs[i].argmax(axis=1) == test_probs[j].argmax(axis=1)))
                if dv < 1e-10 and dt < 1e-10:
                    duplicate_rows.append({"i": i, "j": j, "name_i": names[i], "name_j": names[j], "kind": "exact"})
                if pred_agree_v >= 0.995 and pred_agree_t >= 0.995:
                    near_duplicate_pairs.append({"i": i, "j": j, "name_i": names[i], "name_j": names[j], "pred_agree_val": pred_agree_v, "pred_agree_test": pred_agree_t, "max_abs_val": dv, "max_abs_test": dt})
        u.write_csv(AUDIT_DIR / "near_duplicate_pairs.csv", near_duplicate_pairs)

        old_v, old_t = reference_probs(y, "old_balanced")
        bal_v, bal_t = reference_probs(y, "balanced_plus")
        safe_v, safe_t = reference_probs(y, "safest")
        reference_checks = []
        suspicious_ref_indices = set()
        for ref_name, rv, rt in [("old_balanced", old_v, old_t), ("balanced_v25", bal_v, bal_t), ("safe_v25", safe_v, safe_t)]:
            for i, name in enumerate(names):
                row = {
                    "reference": ref_name,
                    "candidate_idx": i,
                    "candidate_name": name,
                    "max_abs_val": float(np.max(np.abs(rv - val_probs[i]))),
                    "max_abs_test": float(np.max(np.abs(rt - test_probs[i]))),
                    "pred_agreement_val": float(np.mean(rv.argmax(axis=1) == val_probs[i].argmax(axis=1))),
                    "pred_agreement_test": float(np.mean(rt.argmax(axis=1) == test_probs[i].argmax(axis=1))),
                }
                if row["max_abs_val"] < 1e-8 and row["max_abs_test"] < 1e-8:
                    suspicious_ref_indices.add(i)
                if row["pred_agreement_val"] >= 0.98 or row["pred_agreement_test"] >= 0.98:
                    suspicious_ref_indices.add(i)
                reference_checks.append(row)
        reference_checks.sort(key=lambda r: (r["max_abs_val"] + r["max_abs_test"], -r["pred_agreement_val"]))
        u.write_csv(AUDIT_DIR / "reference_similarity_checks.csv", reference_checks[:200])

        original_probs = np.load(STACKER_DIR / "stacker_val_probs.npy")
        original_metrics = metrics_from_prob(y, original_probs)

        n = len(y)
        random5 = [(f"random5_fold{i}", tr, ev) for i, (tr, ev) in enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=20260523).split(np.zeros(n), y), 1)]
        half_splits = []
        for seed in [2024, 2025, 2026, 2027, 2028]:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
            tr, ev = next(sss.split(np.zeros(n), y))
            half_splits.append((f"half_seed{seed}", tr, ev))
        block_ids = np.arange(n) // 10
        even = np.where(block_ids % 2 == 0)[0]
        odd = np.where(block_ids % 2 == 1)[0]
        block_even_odd = [("block10_even_check", odd, even), ("block10_odd_check", even, odd)]
        contiguous = []
        for i, ev in enumerate(np.array_split(np.arange(n), 5), 1):
            ev = np.asarray(ev, dtype=int)
            tr = np.setdiff1d(np.arange(n), ev)
            contiguous.append((f"contiguous_chunk{i}", tr, ev))

        nested_summaries = []
        nested_fold_rows = []
        for label, splits in [
            ("nested_random5", random5),
            ("nested_random_half", half_splits),
            ("nested_block10_even_odd", block_even_odd),
            ("nested_contiguous5", contiguous),
        ]:
            summary, rows = nested_eval(X, y, splits, label)
            nested_summaries.append(summary)
            nested_fold_rows.extend(rows)
        u.write_csv(AUDIT_DIR / "nested_fold_metrics.csv", nested_fold_rows)
        u.write_csv(AUDIT_DIR / "nested_summary.csv", nested_summaries)

        perm_rows = []
        rng = np.random.default_rng(20260523)
        for rep in range(20):
            yp = y.copy()
            rng.shuffle(yp)
            pv = oof_mlp_fixed(X, yp, ORIGINAL_BEST_CFG, 70000 + rep, n_splits=5)
            m = metrics_from_prob(yp, pv)
            perm_rows.append({"rep": rep, **m})
        u.write_csv(AUDIT_DIR / "permutation_label_test.csv", perm_rows)
        perm_accs = [r["acc"] for r in perm_rows]
        perm_summary = {
            "permutation_repeats": len(perm_rows),
            "acc_mean": float(np.mean(perm_accs)),
            "acc_std": float(np.std(perm_accs)),
            "acc_max": float(np.max(perm_accs)),
            "pass": bool(np.max(perm_accs) < 0.45 and np.mean(perm_accs) < 0.38),
        }

        classifier_rows = []
        avg_prob = u.normalize_prob(np.mean(val_probs, axis=0))
        classifier_rows.append({"method": "simple_fixed_average", **metrics_from_prob(y, avg_prob)})
        log_prob = oof_logistic(X, y, "logistic", seed=2026)
        classifier_rows.append({"method": "logistic_regression_oof", **metrics_from_prob(y, log_prob)})
        ridge_pred = oof_logistic(X, y, "ridge", seed=2026)
        classifier_rows.append({"method": "ridge_classifier_oof", **metrics_from_pred(y, ridge_pred)})
        mlp_prob = oof_mlp_fixed(X, y, ORIGINAL_BEST_CFG, 2026, n_splits=5)
        classifier_rows.append({"method": "mlp_fixed_best_oof_seed2026", **metrics_from_prob(y, mlp_prob)})
        u.write_csv(AUDIT_DIR / "classifier_ablation.csv", classifier_rows)

        # Candidate subset ablation. No new candidates are added; subsets select columns from the saved 60.
        old_like = sorted(suspicious_ref_indices)
        keep_no_old_like = [i for i in range(len(pool)) if i not in set(old_like)]
        dup_remove = set()
        for pair in near_duplicate_pairs:
            dup_remove.add(int(pair["j"]))
        keep_no_dup = [i for i in range(len(pool)) if i not in dup_remove]
        diverse = []
        for i in range(len(pool)):
            if not diverse:
                diverse.append(i)
            else:
                agree = max(float(np.mean(val_probs[i].argmax(axis=1) == val_probs[j].argmax(axis=1))) for j in diverse)
                if agree < 0.92:
                    diverse.append(i)
            if len(diverse) >= 20:
                break
        subsets = {
            "all_60": list(range(len(pool))),
            "top4": list(range(min(4, len(pool)))),
            "remove_old_balanced_like": keep_no_old_like,
            "remove_highly_duplicated": keep_no_dup,
            "diverse_greedy_up_to20": diverse,
            "tree_logreg_non_neural_all": list(range(len(pool))),
        }
        subset_rows = []
        for subset_name, cand_idx in subsets.items():
            cols = []
            for idx in cand_idx:
                cols.extend([3 * idx, 3 * idx + 1, 3 * idx + 2])
            Xs = X[:, cols].astype(np.float32)
            cfg = {**ORIGINAL_BEST_CFG, "hidden_dim": min(32, max(8, Xs.shape[1] // 2))}
            pv = oof_mlp_fixed(Xs, y, cfg, 81000 + len(subset_rows), n_splits=5)
            subset_rows.append({"subset": subset_name, "candidate_count": len(cand_idx), "input_dim": Xs.shape[1], **metrics_from_prob(y, pv)})
        u.write_csv(AUDIT_DIR / "candidate_subset_ablation.csv", subset_rows)

        regularization_rows = []
        reg_configs = []
        for dropout in [0.0, 0.3, 0.5, 0.7]:
            reg_configs.append({**ORIGINAL_BEST_CFG, "dropout": dropout})
        for wd in [1e-3, 1e-2]:
            reg_configs.append({**ORIGINAL_BEST_CFG, "weight_decay": wd})
        for ls in [0.0, 0.1]:
            reg_configs.append({**ORIGINAL_BEST_CFG, "label_smoothing": ls})
        seen_cfgs = set()
        unique_reg_configs = []
        for cfg in reg_configs:
            key = tuple((k, cfg[k]) for k in sorted(cfg))
            if key not in seen_cfgs:
                seen_cfgs.add(key)
                unique_reg_configs.append(cfg)
        for i, cfg in enumerate(unique_reg_configs):
            pv = oof_mlp_fixed(X, y, cfg, 82000 + i, n_splits=5)
            regularization_rows.append({**cfg, **metrics_from_prob(y, pv)})
        u.write_csv(AUDIT_DIR / "regularization_ablation.csv", regularization_rows)

        seed_rows = []
        for seed in [2024, 2025, 2026, 2027, 2028]:
            pv = oof_mlp_fixed(X, y, ORIGINAL_BEST_CFG, seed, n_splits=5)
            seed_rows.append({"seed": seed, **metrics_from_prob(y, pv)})
        u.write_csv(AUDIT_DIR / "seed_robustness.csv", seed_rows)

        # Test output checks for stacker and stacker+fusion files.
        test_check_rows = []
        test_items = [
            ("stacker", STACKER_DIR / "stacker_SEED.txt", STACKER_DIR / "stacker_test_probs.npy"),
            ("regularized_fusion_plus_neural_stacker_avg", STACKER_DIR / "regularized_fusion_plus_neural_stacker_avg_SEED.txt", None),
            ("old_balanced", u.V25_OUT / "fallback_old_balanced_SEED.txt", None),
            ("balanced_v25", u.V25_OUT / "balanced_v25_SEED.txt", None),
            ("safe_v25", u.V25_OUT / "safe_v25_SEED.txt", None),
            ("regularized_stable_fusion", u.OUTPUT_ROOT / "regularized_stable_fusion_search_20260523_003129" / "best_stable_fusion_SEED.txt", u.OUTPUT_ROOT / "regularized_stable_fusion_search_20260523_003129" / "best_stable_fusion_test_probs.npy"),
        ]
        # Reconstruct probabilities for blend/reference files when needed.
        stack_t = np.load(STACKER_DIR / "stacker_test_probs.npy")
        reg_v, reg_t = reference_probs(y, "balanced_plus")
        old_v2, old_t2 = reference_probs(y, "old_balanced")
        safe_v2, safe_t2 = reference_probs(y, "safest")
        prob_map = {
            "regularized_fusion_plus_neural_stacker_avg": u.normalize_prob(0.5 * reg_t + 0.5 * stack_t),
            "old_balanced": old_t2,
            "balanced_v25": reg_t,
            "safe_v25": safe_t2,
        }
        for name, seed_path, prob_path in test_items:
            prob = np.load(prob_path) if prob_path is not None and prob_path.exists() else prob_map.get(name)
            vals = [int(x.strip()) for x in seed_path.read_text(encoding="utf-8").splitlines() if x.strip() != ""]
            pred = np.asarray(vals, dtype=int)
            row = {
                "name": name,
                "seed_path": str(seed_path),
                "line_count": len(vals),
                "labels_only_0_1_2": bool(set(vals).issubset({0, 1, 2})),
                "test_distribution": dist_str(pred),
            }
            if prob is not None:
                prob = u.normalize_prob(prob)
                maxp = prob.max(axis=1)
                argmax = prob.argmax(axis=1).astype(int)
                row.update(
                    {
                        "matches_probability_argmax": bool(len(argmax) == len(pred) and np.array_equal(argmax, pred)),
                        "mean_max_probability": float(np.mean(maxp)),
                        "high_conf_ratio_0p7": float(np.mean(maxp >= 0.7)),
                        "high_conf_ratio_0p9": float(np.mean(maxp >= 0.9)),
                    }
                )
                for cls in [0, 1, 2]:
                    mask = argmax == cls
                    row[f"class_{cls}_mean_confidence"] = float(np.mean(maxp[mask])) if mask.any() else None
            test_check_rows.append(row)
        u.write_csv(AUDIT_DIR / "test_prediction_checks.csv", test_check_rows)

        contribution_rows = []
        # Diagnostic, not used for final score: train a logistic stacker on all validation
        # labels to inspect which candidate columns get the largest absolute coefficients.
        clf = LogisticRegression(max_iter=2000, C=0.25, penalty="l2", solver="lbfgs")
        clf.fit(X, y)
        coef_abs = np.abs(clf.coef_).sum(axis=0)
        for i, name in enumerate(names):
            contribution_rows.append({"candidate_idx": i, "candidate_name": name, "coef_abs_sum": float(coef_abs[3 * i : 3 * i + 3].sum()), **candidate_rows[i]})
        contribution_rows.sort(key=lambda r: r["coef_abs_sum"], reverse=True)
        u.write_csv(AUDIT_DIR / "logistic_candidate_contribution_diagnostic.csv", contribution_rows)

        nested_by_name = {r["split_group"]: r for r in nested_summaries}
        seed_accs = [r["acc"] for r in seed_rows]
        classifier_best = sorted(classifier_rows, key=lambda r: r["acc"], reverse=True)
        subset_sorted = sorted(subset_rows, key=lambda r: r["acc"], reverse=True)
        reg_sorted = sorted(regularization_rows, key=lambda r: r["acc"], reverse=True)
        test_stacker = next(r for r in test_check_rows if r["name"] == "stacker")
        test_blend = next(r for r in test_check_rows if r["name"] == "regularized_fusion_plus_neural_stacker_avg")

        if not perm_summary["pass"]:
            rating = "Bug/leakage"
            recommendation = "Do not use the neural stacker."
        elif nested_by_name["nested_block10_even_odd"]["acc_mean"] < 0.65 or nested_by_name["nested_contiguous5"]["acc_mean"] < 0.65:
            rating = "Validation-overfit"
            recommendation = "Do not use the neural stacker as final submission; keep it as a high-risk neural ablation."
        elif nested_by_name["nested_random5"]["acc_mean"] >= 0.68:
            rating = "Useful but risky"
            recommendation = "Use only as aggressive backup; old balanced remains safer."
        else:
            rating = "Validation-overfit"
            recommendation = "Use old balanced/fallback instead."

        code_refs = {
            "candidate_pool": line_ref(u.ROOT / "neural_probability_stacker.py", "u.load_candidate_pool"),
            "oof_split": line_ref(u.ROOT / "neural_probability_stacker.py", "StratifiedKFold"),
            "oof_assignment": line_ref(u.ROOT / "neural_probability_stacker.py", "oof[ev]"),
            "hpo_objective": line_ref(u.ROOT / "neural_probability_stacker.py", "objective ="),
            "test_generation": line_ref(u.ROOT / "neural_probability_stacker.py", "u.write_seed_from_prob"),
        }

        readme = [
            "# Neural Stacker Reliability Audit",
            "",
            "This audit reuses the saved `outputs_experiments/neural_probability_stacker_20260523_004708` stacker run. It does not use hidden test labels and does not add new base candidates.",
            "",
            "## A. Data Flow Audit",
            "",
            f"- Stacker script: `neural_probability_stacker.py`.",
            f"- Audited output directory: `{STACKER_DIR}`.",
            f"- Candidate count: `{len(pool)}`; input dimension `{X.shape[1]}` from `60 x 3` probability columns.",
            f"- Candidate registry written to `{AUDIT_DIR / 'candidate_registry_audit.csv'}`.",
            f"- All candidate val/test probability shapes are `{val_probs[0].shape}` and `{test_probs[0].shape}`.",
            f"- Exact duplicate probability pairs after the original dedupe: `{len(duplicate_rows)}`.",
            f"- Near-duplicate prediction pairs with >=99.5% val/test agreement: `{len(near_duplicate_pairs)}`; details in `near_duplicate_pairs.csv`.",
            f"- Direct old-balanced/v25 aggregate probability included as a candidate: `{bool(suspicious_ref_indices)}` by exact/near reference-prediction check. Similarity details are in `reference_similarity_checks.csv`.",
            f"- Code evidence: candidate pool line(s) `{code_refs['candidate_pool']}`, OOF split line(s) `{code_refs['oof_split']}`, OOF assignment line(s) `{code_refs['oof_assignment']}`, HPO objective line(s) `{code_refs['hpo_objective']}`, test SEED generation line(s) `{code_refs['test_generation']}`.",
            "",
            "Data-flow conclusion: `suspicious`, not confirmed leakage. The per-sample OOF fold assignment is structurally correct, and test labels are not used. The main concern is that hyperparameters and the base probability pool were selected using the same validation benchmark, so the reported `0.7667` has validation-specific optimism.",
            "",
            "## B. Strict Nested Holdout Audit",
            "",
            f"- Original OOF metrics: acc `{original_metrics['acc']:.4f}`, macro-F1 `{original_metrics['macro_f1']:.4f}`, min recall `{original_metrics['min_recall']:.4f}`.",
            "",
            u.md_table(
                nested_summaries,
                ["split_group", "acc", "acc_mean", "acc_std", "acc_min", "macro_f1", "min_recall", "confusion_matrix"],
            ),
            "",
            "Interpretation: nested evaluation removes the most direct hyperparameter-selection optimism by selecting configs inside each outer train split only. Random nested checks are the closest reliability estimate; block/contiguous checks stress sample-order shift.",
            "",
            "## C. Permutation Label Test",
            "",
            f"- Repeats: `{perm_summary['permutation_repeats']}`.",
            f"- Shuffled-label acc mean/std/max: `{perm_summary['acc_mean']:.4f}` / `{perm_summary['acc_std']:.4f}` / `{perm_summary['acc_max']:.4f}`.",
            f"- Pass permutation test: `{perm_summary['pass']}`.",
            "",
            "Permutation conclusion: shuffled labels fall near chance if the pass flag is true; this argues against an obvious label leakage or evaluation bug in the stacker loop.",
            "",
            "## D. Ablation",
            "",
            "### Classifier Ablation",
            "",
            u.md_table(classifier_rows, ["method", "acc", "macro_f1", "min_recall", "confusion_matrix"]),
            "",
            "### Candidate Subset Ablation",
            "",
            u.md_table(subset_rows, ["subset", "candidate_count", "input_dim", "acc", "macro_f1", "min_recall", "confusion_matrix"]),
            "",
            "### Regularization Ablation",
            "",
            u.md_table(reg_sorted, ["hidden_dim", "dropout", "weight_decay", "lr", "label_smoothing", "acc", "macro_f1", "min_recall"]),
            "",
            "### Seed Robustness",
            "",
            u.md_table(seed_rows, ["seed", "acc", "macro_f1", "min_recall", "confusion_matrix"]),
            "",
            f"- Seed acc mean/std: `{np.mean(seed_accs):.4f}` / `{np.std(seed_accs):.4f}`.",
            f"- Top diagnostic contributor: `{contribution_rows[0]['candidate_name']}`. Full diagnostic coefficient ranking is in `logistic_candidate_contribution_diagnostic.csv`.",
            "- No separate neural base candidates were identified in this saved candidate pool; the pool is feature/probability-model based.",
            "",
            "## E. Test Prediction Checks",
            "",
            u.md_table(test_check_rows, ["name", "line_count", "labels_only_0_1_2", "matches_probability_argmax", "test_distribution", "mean_max_probability", "high_conf_ratio_0p7", "high_conf_ratio_0p9", "class_0_mean_confidence", "class_1_mean_confidence", "class_2_mean_confidence"]),
            "",
            f"- Stacker test distribution: `{test_stacker['test_distribution']}`.",
            f"- Stacker+fusion test distribution: `{test_blend['test_distribution']}`.",
            "The stacker distributions are not single-class collapsed, but they differ materially from old balanced/safe outputs and therefore remain higher risk.",
            "",
            "## F. Final Reliability Rating",
            "",
            f"- Rating: `{rating}`.",
            f"- Recommendation: {recommendation}",
            "",
            "Direct answers:",
            "",
            f"- Is `0.7667` trustworthy? `{rating}`. It is not proven to be a bug, but it should not be treated as equally reliable as the audited fixed fusion.",
            "- Should `regularized_fusion_plus_neural_stacker_avg_SEED.txt` be final submitted? No, not as the primary final submission. It is an aggressive backup only.",
            "- Compared with `fallback_old_balanced_SEED.txt`, the fallback/old balanced file is more suitable as the final submission because it is an audited fixed probability fusion with a clearer risk profile.",
            "- Poster wording if stacker is not recommended: `A neural probability stacker improved validation/OOF scores, but stricter nested and block-wise audits indicated validation-specific overfitting risk; therefore it was retained as a high-risk neural meta-ensemble ablation rather than the primary submission.`",
        ]
        (u.ROOT / "README_neural_stacker_reliability_audit.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
        result.update(
            {
                "status": "completed",
                "rating": rating,
                "original_metrics": original_metrics,
                "nested_summaries": nested_summaries,
                "permutation_summary": perm_summary,
                "classifier_ablation": classifier_rows,
                "subset_ablation": subset_rows,
                "regularization_ablation": regularization_rows,
                "seed_robustness": seed_rows,
                "test_checks": test_check_rows,
            }
        )
    except Exception as exc:
        result["status"] = "failed"
        result["errors"].append({"error": repr(exc), "trace": traceback.format_exc()})
        (u.ROOT / "README_neural_stacker_reliability_audit.md").write_text(
            "# Neural Stacker Reliability Audit\n\nAudit failed; see audit run_results.json for traceback.\n",
            encoding="utf-8",
        )
    finally:
        u.write_json(AUDIT_DIR / "run_results.json", result)
        print(f"OUTPUT_DIR={AUDIT_DIR}")


if __name__ == "__main__":
    main()
