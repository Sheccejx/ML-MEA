#!/usr/bin/env python3
"""SEED clean-only feature/block fusion v2.

No historical order-aware/source-match probabilities enter the clean pool.
The script starts from official H5 files, reproduces the known clean 0.6222
pipeline, then trains/fuses only clean tabular feature candidates.
"""

from __future__ import annotations

import csv
import json
import math
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import car_block_feature_refinement as base


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs_experiments"
DATA = ROOT / "course_project" / "SEED"
PREV_CAR_DIR = OUTPUT_ROOT / "car_block_feature_refinement_20260521_003247"
PREV_SPLIT_DIR = OUTPUT_ROOT / "seed_split_forensics_and_reprocessing_20260520_193716"
CURRENT_CLEAN_BEST = 0.6222222222222222
FS = 200.0
EPS = 1e-10
RNG = np.random.default_rng(20260521)

FORBIDDEN_TERMS = [
    "order_aware",
    "soft_order",
    "hard_order",
    "order_prior",
    "prior_peak",
    "offset",
    "source_match",
    "source_matching",
    "high_risk",
    "medium_risk",
    "block_smoothing_soft_order_prior",
    "recommendable=false",
    "final_seed_score_maximization_20260519_233236",
    "order_aware_model_fusion_20260519_231547",
]
EXCLUDED_DIR_MARKERS = [
    "outputs_experiments/final_seed_score_maximization_20260519_233236",
    "outputs_experiments\\final_seed_score_maximization_20260519_233236",
    "outputs_experiments/order_aware_model_fusion_20260519_231547",
    "outputs_experiments\\order_aware_model_fusion_20260519_231547",
]
FUSION_MARKERS = {"fusion", "probability_pool", "bias_calibration", "greedy_forward", "dirichlet_random", "previous_05333_replay_low_risk"}

BANDS = [
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta_low", 13.0, 20.0),
    ("beta_high", 20.0, 30.0),
    ("gamma_low", 30.0, 45.0),
    ("gamma_high", 45.0, 75.0),
]


def ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(jsonable(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(jsonable(row.get(k)), ensure_ascii=False) if isinstance(row.get(k), (dict, list, tuple, np.ndarray)) else jsonable(row.get(k)) for k in keys})


def log(logs: List[str], msg: str) -> None:
    logs.append(f"- `{time.strftime('%H:%M:%S')}` {msg}")
    print(msg, flush=True)


def contains_forbidden(text: Any) -> Optional[str]:
    s = json.dumps(jsonable(text), ensure_ascii=False).lower().replace("/", "\\")
    for marker in EXCLUDED_DIR_MARKERS:
        if marker.lower().replace("/", "\\") in s:
            return f"excluded_dir:{marker}"
    for term in FORBIDDEN_TERMS:
        if term.lower() in s:
            return f"forbidden_term:{term}"
    return None


def normalize_prob(p: np.ndarray) -> np.ndarray:
    return base.normalize_prob(p)


def evaluate(y: np.ndarray, prob: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(prob)
    pred = normalize_prob(arr).argmax(axis=1) if arr.ndim == 2 else arr.astype(int).reshape(-1)
    rec = recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "val_acc": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "min_recall": float(rec.min()),
        "per_class_recall": {str(i): float(v) for i, v in enumerate(rec)},
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).astype(int).tolist(),
        "prediction_distribution": {str(k): int(v) for k, v in Counter(pred.tolist()).items()},
    }


def score(m: Dict[str, Any]) -> float:
    return float(m["val_acc"] + 0.2 * m["macro_f1"] + 0.1 * m["min_recall"])


def safe_name(name: str) -> str:
    return base.safe_name(name)


def validate_seed(path: Path) -> Dict[str, Any]:
    vals = []
    ok = True
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            v = int(line.strip())
            vals.append(v)
            ok = ok and v in {0, 1, 2}
        except Exception:
            ok = False
    return {"path": str(path), "line_count": len(vals), "exactly_450_lines": len(vals) == 450, "labels_only_0_1_2": bool(ok), "distribution": dict(Counter(vals))}


def write_seed(path: Path, pred: np.ndarray) -> Dict[str, Any]:
    pred = np.asarray(pred, dtype=int).reshape(-1)
    with path.open("w", encoding="utf-8", newline="\n") as fp:
        for v in pred:
            fp.write(f"{int(v)}\n")
    return validate_seed(path)


def write_submission(path: Path, pred: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["id", "label"])
        for i, v in enumerate(np.asarray(pred, dtype=int).reshape(-1)):
            writer.writerow([i, int(v)])


def save_prob(out_dir: Path, name: str, pv: np.ndarray, pte: np.ndarray) -> None:
    np.save(out_dir / "all_val_probs" / f"{safe_name(name)}.npy", normalize_prob(pv))
    np.save(out_dir / "all_test_probs" / f"{safe_name(name)}.npy", normalize_prob(pte))


def add_candidate(rows: List[Dict[str, Any]], out_dir: Path, name: str, source: str, pv: np.ndarray, pte: np.ndarray, y: np.ndarray, meta: Dict[str, Any]) -> Dict[str, Any]:
    forbidden = contains_forbidden({"name": name, "source": source, "meta": meta})
    if forbidden:
        raise ValueError(f"Refusing to add contaminated clean candidate {name}: {forbidden}")
    pv, pte = normalize_prob(pv), normalize_prob(pte)
    m = evaluate(y, pv)
    pred_test = pte.argmax(axis=1)
    row = {
        "name": name,
        "source": source,
        "risk_level": "clean-low",
        "uses_order_prior": False,
        "uses_source_matching": False,
        "score": score(m),
        "test_prediction_distribution": {str(k): int(v) for k, v in Counter(pred_test.tolist()).items()},
        **m,
        **meta,
    }
    rows.append(row)
    save_prob(out_dir, name, pv, pte)
    return row


def load_previous_clean_candidates(out_dir: Path, y: np.ndarray, registry: List[Dict[str, Any]], excluded: List[Dict[str, Any]]) -> List[Tuple[str, np.ndarray, np.ndarray, Dict[str, Any]]]:
    pool = []
    summary = PREV_CAR_DIR / "all_candidates_summary.csv"
    if not summary.exists():
        return pool
    with summary.open(encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            name = row.get("name", "")
            full_text = dict(row)
            reason = contains_forbidden(full_text)
            if not reason:
                if row.get("risk_level") != "low":
                    reason = f"risk_not_low:{row.get('risk_level')}"
                elif row.get("selected") not in (None, "", "[]"):
                    reason = "fusion_selected_field_present"
                elif (row.get("preprocess") in FUSION_MARKERS or row.get("feature_family") in FUSION_MARKERS or row.get("model") in FUSION_MARKERS or name in FUSION_MARKERS):
                    reason = "previous_fusion_or_bias_candidate_not_atomic_feature"
                elif row.get("uses_order_prior") not in ("False", "false", False, "", None):
                    reason = "uses_order_prior_not_false"
                elif row.get("uses_source_matching") not in ("False", "false", False, "", None):
                    reason = "uses_source_matching_not_false"
            val_path = PREV_CAR_DIR / "all_val_probs" / f"{safe_name(name)}.npy"
            test_path = PREV_CAR_DIR / "all_test_probs" / f"{safe_name(name)}.npy"
            if not reason and (not val_path.exists() or not test_path.exists()):
                reason = "probability_file_missing"
            reg = {"candidate": name, "candidate_source": str(summary), "included": reason is None, "reason": reason or "included_clean_atomic_feature"}
            registry.append(reg)
            if reason:
                excluded.append(reg)
                continue
            pv, pte = normalize_prob(np.load(val_path)), normalize_prob(np.load(test_path))
            meta = {
                "origin": "previous_car_clean_atomic",
                "preprocess": row.get("preprocess"),
                "feature_family": row.get("feature_family"),
                "model": row.get("model"),
                "smoothing": row.get("smoothing"),
                "block_size": row.get("block_size"),
            }
            pool.append((f"prev_clean__{name}", pv, pte, {**evaluate(y, pv), **meta}))
    # Add the previously confirmed clean split-forensics candidate explicitly.
    for stem in ["low_risk_model_candidate", "medium_low_risk_model_candidate"]:
        vp = PREV_SPLIT_DIR / f"{stem}_val_prob.npy"
        tp = PREV_SPLIT_DIR / f"{stem}_test_prob.npy"
        name = f"prev_split_clean__{stem}"
        reason = contains_forbidden({"path": str(vp), "name": name})
        if not reason and vp.exists() and tp.exists():
            pv, pte = normalize_prob(np.load(vp)), normalize_prob(np.load(tp))
            meta = {"origin": "previous_split_clean", "preprocess": "clean_previous", "feature_family": "legacy_clean", "model": stem, "smoothing": "prob_block_smooth_10"}
            pool.append((name, pv, pte, {**evaluate(y, pv), **meta}))
            registry.append({"candidate": name, "candidate_source": str(vp), "included": True, "reason": "included_previous_confirmed_clean"})
        else:
            reg = {"candidate": name, "candidate_source": str(vp), "included": False, "reason": reason or "missing_previous_split_prob"}
            registry.append(reg)
            excluded.append(reg)
    return pool


def model_specs() -> List[Tuple[str, Any]]:
    return [
        ("hgb_lr003_i200_l31_l2_001", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.03, max_leaf_nodes=31, l2_regularization=0.01, random_state=21)),
        ("hgb_lr005_i200_l31_l2_001", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=0.01, random_state=22)),
        ("hgb_lr005_i300_l63_l2_0", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=0.0, random_state=23)),
        ("hgb_lr003_i400_l15_l2_01", HistGradientBoostingClassifier(max_iter=400, learning_rate=0.03, max_leaf_nodes=15, l2_regularization=0.1, random_state=24)),
        ("et_500_sqrt_l1", ExtraTreesClassifier(n_estimators=500, min_samples_leaf=1, max_features="sqrt", class_weight="balanced", random_state=25, n_jobs=-1)),
        ("et_700_03_l2", ExtraTreesClassifier(n_estimators=700, min_samples_leaf=2, max_features=0.3, class_weight="balanced", random_state=26, n_jobs=-1)),
        ("rf_500_sqrt_l1", RandomForestClassifier(n_estimators=500, min_samples_leaf=1, max_features="sqrt", class_weight="balanced", random_state=27, n_jobs=-1)),
        ("rf_700_03_l2", RandomForestClassifier(n_estimators=700, min_samples_leaf=2, max_features=0.3, class_weight="balanced", random_state=28, n_jobs=-1)),
        ("logreg_C01_bal", LogisticRegression(max_iter=4000, C=0.1, class_weight="balanced", random_state=29)),
        ("logreg_C1_bal", LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced", random_state=30)),
        ("linear_svm_C02", LinearSVC(C=0.2, class_weight="balanced", random_state=31, max_iter=7000)),
        ("mlp_128_a001", MLPClassifier(hidden_layer_sizes=(128,), alpha=1e-3, early_stopping=True, max_iter=500, random_state=32)),
    ]


def predict_proba(est: Any, X: np.ndarray) -> np.ndarray:
    return base.predict_proba(est, X)


def train_model_prob(estimator: Any, Ftr: np.ndarray, ytr: np.ndarray, Fv: np.ndarray, Fte: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    model = make_pipeline(StandardScaler(), clone(estimator))
    model.fit(Ftr, ytr)
    return predict_proba(model, Fv), predict_proba(model, Fte)


def add_smoothed_variants(pool: List[Tuple[str, np.ndarray, np.ndarray, Dict[str, Any]]], rows: List[Dict[str, Any]], out_dir: Path, y: np.ndarray, name: str, source: str, pv: np.ndarray, pte: np.ndarray, meta: Dict[str, Any]) -> None:
    base_row = add_candidate(rows, out_dir, name, source, pv, pte, y, {**meta, "smoothing": "none"})
    pool.append((name, normalize_prob(pv), normalize_prob(pte), base_row))
    for bs in [5, 10, 15, 20, 25, 30]:
        bpv, bpte = base.block_smooth(pv, bs), base.block_smooth(pte, bs)
        bname = f"{name}__prob_block_smooth_{bs}"
        brow = add_candidate(rows, out_dir, bname, source, bpv, bpte, y, {**meta, "smoothing": "prob_block_smooth", "block_size": bs, "hybrid_alpha": None})
        pool.append((bname, bpv, bpte, brow))
        for alpha in [0.0, 0.2, 0.4, 0.6, 0.8]:
            hpv = normalize_prob(alpha * normalize_prob(pv) + (1 - alpha) * bpv)
            hpte = normalize_prob(alpha * normalize_prob(pte) + (1 - alpha) * bpte)
            hname = f"{name}__hybrid_block_{bs}_alpha_{alpha}"
            hrow = add_candidate(rows, out_dir, hname, source, hpv, hpte, y, {**meta, "smoothing": "hybrid_sample_block_prob", "block_size": bs, "hybrid_alpha": alpha})
            pool.append((hname, hpv, hpte, hrow))


def make_target_feature_sets(Xtr: np.ndarray, Xv: np.ndarray, Xte: np.ndarray, cache_dir: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]]:
    variants = base.preprocess_variants(Xtr, Xv, Xte)
    selected: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]] = {}
    for prep in ["common_average_reference", "per_sample_channel_zscore", "raw", "car_plus_per_sample_channel_zscore"]:
        feat_sets = base.fit_feature_sets(*variants[prep], cache_dir, prep)
        wanted = []
        if prep == "common_average_reference":
            wanted = ["channel_stats", "band", "stats_band", "stats_band_eig", "stats_band_covpca20", "stats_band_covpca50", "stats_band_covpca100"]
        elif prep == "per_sample_channel_zscore":
            wanted = ["channel_stats", "stats_band", "legacy"]
        elif prep == "raw":
            wanted = ["channel_stats", "stats_band"]
        elif prep == "car_plus_per_sample_channel_zscore":
            wanted = ["channel_stats", "legacy", "stats_band"]
        for feat in wanted:
            if feat in feat_sets:
                a, b, c, meta = feat_sets[feat]
                selected[f"{prep}__{feat}"] = (a, b, c, {"preprocess": prep, "feature_family": feat, **meta})
        if prep in {"common_average_reference", "raw", "car_plus_per_sample_channel_zscore"}:
            Xp_tr, Xp_v, Xp_te = variants[prep]
            Etr, Ev, Ete, emeta = fit_v2_enhanced_features(Xp_tr, Xp_v, Xp_te, cache_dir, prep)
            selected[f"{prep}__v2_enhanced_stats_spectral"] = (
                Etr,
                Ev,
                Ete,
                {"preprocess": prep, "feature_family": "v2_enhanced_stats_spectral", **emeta},
            )
    return selected


def _safe_corr_upper(X: np.ndarray) -> np.ndarray:
    out = []
    iu = np.triu_indices(X.shape[1], k=1)
    for sample in X:
        c = np.corrcoef(sample)
        c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
        out.append(c[iu])
    return np.asarray(out, dtype=np.float32)


def _safe_slope_features(X: np.ndarray) -> np.ndarray:
    # Compact temporal-shape descriptor: first/second half means and linear trend per channel.
    t = np.linspace(-1.0, 1.0, X.shape[-1], dtype=np.float32)
    t_norm = float(np.sum(t * t)) + EPS
    first = X[:, :, : X.shape[-1] // 2].mean(axis=2)
    second = X[:, :, X.shape[-1] // 2 :].mean(axis=2)
    slope = np.einsum("nct,t->nc", X, t) / t_norm
    return np.concatenate([first, second, second - first, slope], axis=1).astype(np.float32)


def _spectral_moment_features(X: np.ndarray) -> np.ndarray:
    freqs = np.fft.rfftfreq(X.shape[-1], d=1.0 / FS)
    power = np.abs(np.fft.rfft(X, axis=-1)) ** 2
    total = power.sum(axis=-1) + EPS
    centroid = (power * freqs[None, None, :]).sum(axis=-1) / total
    spread = np.sqrt((power * (freqs[None, None, :] - centroid[:, :, None]) ** 2).sum(axis=-1) / total)
    band_parts = []
    for _, lo, hi in BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        bp = np.log(power[:, :, mask].mean(axis=-1) + EPS)
        rel = power[:, :, mask].sum(axis=-1) / total
        band_parts.extend([bp, rel])
    return np.concatenate([centroid, spread, *band_parts], axis=1).astype(np.float32)


def _robust_channel_features(X: np.ndarray) -> np.ndarray:
    q = np.quantile(X, [0.05, 0.25, 0.5, 0.75, 0.95], axis=2).transpose(1, 2, 0).reshape(len(X), -1)
    sk = stats.skew(X, axis=2, bias=False, nan_policy="omit")
    ku = stats.kurtosis(X, axis=2, fisher=True, bias=False, nan_policy="omit")
    rms = np.sqrt(np.mean(X * X, axis=2) + EPS)
    zcr = np.mean(np.diff(np.signbit(X), axis=2), axis=2)
    return np.nan_to_num(np.concatenate([q, sk, ku, rms, zcr], axis=1), nan=0.0).astype(np.float32)


def fit_v2_enhanced_features(
    Xtr: np.ndarray, Xv: np.ndarray, Xte: np.ndarray, cache_dir: Path, prefix: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = [cache_dir / f"{prefix}_v2_enhanced_{split}.npy" for split in ("train", "val", "test")]
    if all(p.exists() for p in paths):
        Ftr, Fv, Fte = [np.load(p) for p in paths]
    else:
        def build(X: np.ndarray) -> np.ndarray:
            parts = [
                base.channel_stats_features(X),
                base.band_features(X),
                base.covariance_eig_features(X, topk=40),
                _robust_channel_features(X),
                _safe_slope_features(X),
                _spectral_moment_features(X),
                _safe_corr_upper(X),
            ]
            return np.nan_to_num(np.concatenate(parts, axis=1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        Ftr, Fv, Fte = build(Xtr), build(Xv), build(Xte)
        for arr, path in zip((Ftr, Fv, Fte), paths):
            np.save(path, arr)
    return Ftr, Fv, Fte, {
        "families": [
            "channel_statistics",
            "frequency_band_features",
            "covariance_eigenvalues",
            "robust_quantile_moments",
            "temporal_slope_halves",
            "spectral_moments",
            "correlation_upper_triangle",
        ],
        "feature_shape": list(Ftr.shape),
    }


def block_feature_candidates(Ftr: np.ndarray, ytr: np.ndarray, Fv: np.ndarray, Fte: np.ndarray, eval_bs: int, train_bs: int, model: Any) -> Tuple[np.ndarray, np.ndarray]:
    Btr = base.block_feature_matrix(Ftr, train_bs)
    Bv = base.block_feature_matrix(Fv, eval_bs)
    Bte = base.block_feature_matrix(Fte, eval_bs)
    yb = base.train_block_labels(ytr, train_bs)
    pipe = make_pipeline(StandardScaler(), clone(model))
    pipe.fit(Btr, yb)
    pv_block = predict_proba(pipe, Bv)
    pte_block = predict_proba(pipe, Bte)
    return np.repeat(pv_block, eval_bs, axis=0)[: len(Fv)], np.repeat(pte_block, eval_bs, axis=0)[: len(Fte)]


def greedy_fusion(pool: List[Tuple[str, np.ndarray, np.ndarray, Dict[str, Any]]], y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    pool = sorted(pool, key=lambda x: (x[3]["val_acc"], x[3]["macro_f1"]), reverse=True)
    selected = [pool[0]]
    cur_v, cur_t = pool[0][1], pool[0][2]
    cur_m = evaluate(y, cur_v)
    steps = [{"name": pool[0][0], "alpha_previous": 1.0, "metrics": cur_m}]
    improved = True
    while improved and len(selected) < 10:
        improved = False
        best = None
        for cand in pool:
            if any(cand[0] == s[0] for s in selected):
                continue
            for alpha in np.linspace(0.05, 0.95, 19):
                pv = normalize_prob(alpha * cur_v + (1 - alpha) * cand[1])
                m = evaluate(y, pv)
                if score(m) > score(cur_m) + 1e-10:
                    best = (cand, alpha, pv, normalize_prob(alpha * cur_t + (1 - alpha) * cand[2]), m)
                    cur_m = m
                    improved = True
        if best is not None:
            selected.append(best[0])
            cur_v, cur_t = best[2], best[3]
            steps.append({"name": best[0][0], "alpha_previous": float(best[1]), "metrics": best[4]})
    return cur_v, cur_t, {"fusion_method": "greedy_forward_clean", "selected": [s[0] for s in selected], "steps": steps}


def dirichlet_fusion(pool: List[Tuple[str, np.ndarray, np.ndarray, Dict[str, Any]]], y: np.ndarray, trials: int = 100000) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    top = sorted(pool, key=lambda x: (x[3]["val_acc"], x[3]["macro_f1"]), reverse=True)[:32]
    P = np.stack([x[1] for x in top])
    T = np.stack([x[2] for x in top])
    best_w = np.zeros(len(top))
    best_v = top[0][1]
    best_m = evaluate(y, best_v)
    done = 0
    while done < trials:
        b = min(2000, trials - done)
        W = RNG.dirichlet(np.ones(len(top)) * 0.6, size=b)
        fused = np.einsum("bn,nsc->bsc", W, P)
        pred = fused.argmax(axis=2)
        for i in range(b):
            m = evaluate(y, pred[i])
            if score(m) > score(best_m):
                best_m = m
                best_w = W[i].copy()
                best_v = normalize_prob(fused[i])
        done += b
    best_t = normalize_prob(np.einsum("n,nsc->sc", best_w, T))
    weights = [{"name": n, "weight": float(w)} for n, w in sorted(zip([x[0] for x in top], best_w), key=lambda x: -x[1]) if w > 1e-4]
    return best_v, best_t, {"fusion_method": "dirichlet_clean", "trials": trials, "weights": weights, "metrics": best_m}


def classwise_fusion(pool: List[Tuple[str, np.ndarray, np.ndarray, Dict[str, Any]]], y: np.ndarray, trials: int = 100000) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    top = sorted(pool, key=lambda x: (x[3]["val_acc"], x[3]["macro_f1"]), reverse=True)[:18]
    P = np.stack([x[1] for x in top])
    T = np.stack([x[2] for x in top])
    best_W = np.zeros((3, len(top)))
    best_v = top[0][1]
    best_m = evaluate(y, best_v)
    for _ in range(trials):
        W = RNG.dirichlet(np.ones(len(top)) * 0.7, size=3)
        fv = np.zeros_like(P[0])
        ft = np.zeros_like(T[0])
        for c in range(3):
            fv[:, c] = np.einsum("n,ns->s", W[c], P[:, :, c])
            ft[:, c] = np.einsum("n,ns->s", W[c], T[:, :, c])
        fv = normalize_prob(fv)
        m = evaluate(y, fv)
        if score(m) > score(best_m):
            best_m = m
            best_W = W.copy()
            best_v = fv
            best_t = normalize_prob(ft)
    if np.all(best_W == 0):
        best_t = top[0][2]
    weights = [{"class": int(c), "weights": [{"name": top[i][0], "weight": float(w)} for i, w in enumerate(best_W[c]) if w > 1e-4]} for c in range(3)]
    return best_v, best_t, {"fusion_method": "classwise_dirichlet_clean", "trials": trials, "classwise_weights": weights, "metrics": best_m}


def bias_calibrate(pv: np.ndarray, pte: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    best_m = evaluate(y, pv)
    best_b = np.zeros(3)
    best_v, best_t = pv, pte
    logv = np.log(normalize_prob(pv) + 1e-10)
    logt = np.log(normalize_prob(pte) + 1e-10)
    for b0 in np.linspace(-0.5, 0.5, 21):
        for b1 in np.linspace(-0.5, 0.5, 21):
            for b2 in np.linspace(-0.5, 0.5, 21):
                b = np.array([b0, b1, b2])
                vv = base.softmax(logv + b)
                m = evaluate(y, vv)
                if score(m) > score(best_m):
                    best_m = m
                    best_b = b.copy()
                    best_v = vv
                    best_t = base.softmax(logt + b)
    return best_v, best_t, {"fusion_method": "bias_calibration_clean", "bias": best_b.tolist(), "metrics": best_m}


def confidence_binned_calibration(pv: np.ndarray, pte: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    best_m = evaluate(y, pv)
    best_v, best_t = pv, pte
    best_cfg = {"threshold": None, "blend": None, "target": None}
    targets = {
        "uniform": np.ones(3) / 3,
        "val_prior": np.bincount(y, minlength=3) / len(y),
    }
    for target_name, target in targets.items():
        for thr in np.linspace(0.34, 0.75, 15):
            for blend in np.linspace(0.05, 0.5, 10):
                vv = normalize_prob(pv.copy())
                tt = normalize_prob(pte.copy())
                mv = vv.max(axis=1) < thr
                mt = tt.max(axis=1) < thr
                vv[mv] = normalize_prob((1 - blend) * vv[mv] + blend * target)
                tt[mt] = normalize_prob((1 - blend) * tt[mt] + blend * target)
                m = evaluate(y, vv)
                if score(m) > score(best_m):
                    best_m = m
                    best_v, best_t = vv, tt
                    best_cfg = {"threshold": float(thr), "blend": float(blend), "target": target_name}
    return best_v, best_t, {"fusion_method": "confidence_binned_clean", "config": best_cfg, "metrics": best_m}


def main() -> None:
    out_dir = OUTPUT_ROOT / f"clean_feature_block_fusion_v2_{ts()}"
    out_dir.mkdir(parents=True, exist_ok=False)
    for sub in ["all_val_probs", "all_test_probs", "feature_cache"]:
        (out_dir / sub).mkdir()
    logs = ["# Clean feature/block fusion v2 log", ""]
    registry: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    results: Dict[str, Any] = {"status": "running", "output_dir": str(out_dir), "errors": []}
    config = {
        "current_clean_best_reference": CURRENT_CLEAN_BEST,
        "forbidden_terms": FORBIDDEN_TERMS,
        "explicitly_excluded_dirs": EXCLUDED_DIR_MARKERS,
        "clean_methods_only": [
            "official H5 train/val/test only",
            "CAR/raw/zscore preprocessing",
            "channel/band/covariance features",
            "v2 enhanced robust/spectral/correlation features",
            "RF/ET/HGB/Logistic/SVM/MLP tabular models",
            "probability block averaging",
            "clean-only greedy/Dirichlet/classwise fusion",
        ],
        "no_test_labels": True,
        "no_order_prior": True,
        "no_historical_probability_pool": True,
        "fusion_trials": {"dirichlet": 150000, "classwise": 150000},
    }
    write_json(out_dir / "config.json", config)
    try:
        log(logs, "Loading official train/val/test")
        Xtr, ytr = base.load_xy(DATA / "train.h5", True)
        Xv, yv = base.load_xy(DATA / "val.h5", True)
        Xte, _ = base.load_xy(DATA / "test_x_only.h5", False)

        log(logs, "Starting v2 from official H5 only; historical probability pools are intentionally skipped")
        pool: List[Tuple[str, np.ndarray, np.ndarray, Dict[str, Any]]] = []

        log(logs, "Extracting targeted clean feature sets")
        feature_sets = make_target_feature_sets(Xtr, Xv, Xte, out_dir / "feature_cache")

        log(logs, "Training targeted clean feature models")
        specs = model_specs()
        for feat_name, (Ftr, Fv, Fte, fmeta) in feature_sets.items():
            # Keep expensive models on targeted requested combinations.
            for model_name, estimator in specs:
                if "covpca100" in feat_name and model_name.startswith(("rf_", "mlp", "linear")):
                    continue
                if "v2_enhanced" in feat_name and model_name.startswith(("linear_svm", "mlp")):
                    continue
                if "band" == fmeta.get("feature_family") and not model_name.startswith(("hgb", "et", "logreg")):
                    continue
                try:
                    pv, pte = train_model_prob(estimator, Ftr, ytr, Fv, Fte)
                    name = f"{feat_name}__{model_name}"
                    meta = {**fmeta, "model": model_name, "model_family": model_name.split("_")[0], "candidate_type": "sample_feature_model"}
                    add_smoothed_variants(pool, rows, out_dir, yv, name, "new_clean_feature_model", pv, pte, meta)
                except Exception as exc:
                    results["errors"].append({"stage": f"{feat_name}/{model_name}", "error": repr(exc), "trace": traceback.format_exc()})

        log(logs, "Training clean block-level feature models")
        block_models = [
            ("block_hgb", HistGradientBoostingClassifier(max_iter=220, learning_rate=0.04, max_leaf_nodes=31, l2_regularization=0.01, random_state=41)),
            ("block_et", ExtraTreesClassifier(n_estimators=600, min_samples_leaf=1, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)),
            ("block_rf", RandomForestClassifier(n_estimators=500, min_samples_leaf=1, max_features="sqrt", class_weight="balanced", random_state=43, n_jobs=-1)),
        ]
        for feat_name in [k for k in feature_sets if k.startswith("common_average_reference") and any(x in k for x in ["channel_stats", "stats_band_covpca20", "stats_band_covpca50", "stats_band_eig", "v2_enhanced"])]:
            Ftr, Fv, Fte, fmeta = feature_sets[feat_name]
            for train_bs, eval_bs in [(20, 10), (10, 10), (20, 5), (20, 15), (20, 20), (20, 30)]:
                for model_name, estimator in block_models:
                    try:
                        pv, pte = block_feature_candidates(Ftr, ytr, Fv, Fte, eval_bs, train_bs, estimator)
                        name = f"{feat_name}__{model_name}__trainblock_{train_bs}__evalblock_{eval_bs}"
                        meta = {**fmeta, "model": model_name, "candidate_type": "block_level_classifier", "train_block_size": train_bs, "val_test_block_size": eval_bs, "smoothing": "block_classifier_broadcast"}
                        row = add_candidate(rows, out_dir, name, "new_clean_block_model", pv, pte, yv, meta)
                        pool.append((name, normalize_prob(pv), normalize_prob(pte), row))
                    except Exception as exc:
                        results["errors"].append({"stage": f"block/{feat_name}/{train_bs}/{eval_bs}/{model_name}", "error": repr(exc), "trace": traceback.format_exc()})

        log(logs, "Running clean-only fusions and calibrations")
        # Deduplicate the pool by rounded probabilities.
        clean_pool = []
        seen = set()
        for item in sorted(pool, key=lambda x: (x[3]["val_acc"], x[3]["macro_f1"]), reverse=True):
            reason = contains_forbidden({"name": item[0], "meta": item[3]})
            if reason:
                excluded.append({"candidate": item[0], "included": False, "reason": f"late_pool_exclusion:{reason}"})
                continue
            key = (np.round(item[1], 8).tobytes(), np.round(item[2], 8).tobytes())
            if key in seen:
                continue
            seen.add(key)
            clean_pool.append(item)
        results["clean_pool_size"] = len(clean_pool)

        gv, gt, gmeta = greedy_fusion(clean_pool, yv)
        grow = add_candidate(rows, out_dir, "clean_fusion_greedy_forward", "clean_fusion", gv, gt, yv, {"candidate_type": "clean_fusion", **gmeta})
        clean_pool.append(("clean_fusion_greedy_forward", gv, gt, grow))

        dv, dt, dmeta = dirichlet_fusion(clean_pool, yv, 150000)
        drow = add_candidate(rows, out_dir, "clean_fusion_dirichlet150k", "clean_fusion", dv, dt, yv, {"candidate_type": "clean_fusion", **dmeta})
        clean_pool.append(("clean_fusion_dirichlet150k", dv, dt, drow))

        cv, ct, cmeta = classwise_fusion(clean_pool, yv, 150000)
        crow = add_candidate(rows, out_dir, "clean_fusion_classwise150k", "clean_fusion", cv, ct, yv, {"candidate_type": "clean_fusion", **cmeta})
        clean_pool.append(("clean_fusion_classwise150k", cv, ct, crow))

        for base_name, pv, pte in [("best_single", clean_pool[0][1], clean_pool[0][2]), ("greedy", gv, gt), ("dirichlet", dv, dt), ("classwise", cv, ct)]:
            bv, bt, bmeta = bias_calibrate(pv, pte, yv)
            brow = add_candidate(rows, out_dir, f"clean_{base_name}_bias_calibrated", "clean_calibration", bv, bt, yv, {"candidate_type": "clean_bias_calibration", "base": base_name, **bmeta})
            clean_pool.append((f"clean_{base_name}_bias_calibrated", bv, bt, brow))
            kv, kt, kmeta = confidence_binned_calibration(bv, bt, yv)
            krow = add_candidate(rows, out_dir, f"clean_{base_name}_confidence_binned", "clean_calibration", kv, kt, yv, {"candidate_type": "clean_confidence_binned_calibration", "base": base_name, **kmeta})
            clean_pool.append((f"clean_{base_name}_confidence_binned", kv, kt, krow))

        best = max(rows, key=lambda r: (r["val_acc"], r["macro_f1"], r["score"]))
        best_v = normalize_prob(np.load(out_dir / "all_val_probs" / f"{safe_name(best['name'])}.npy"))
        best_t = normalize_prob(np.load(out_dir / "all_test_probs" / f"{safe_name(best['name'])}.npy"))
        write_seed(out_dir / "best_clean_low_risk_SEED.txt", best_t.argmax(axis=1))
        write_submission(out_dir / "best_submission.csv", best_t.argmax(axis=1))
        np.save(out_dir / "best_clean_val_prob.npy", best_v)
        np.save(out_dir / "best_clean_test_prob.npy", best_t)

        write_csv(out_dir / "risk_registry.csv", registry)
        write_csv(out_dir / "excluded_candidates.csv", excluded)
        write_csv(out_dir / "all_clean_candidates_summary.csv", sorted(rows, key=lambda r: (r["val_acc"], r["macro_f1"], r["score"]), reverse=True))
        write_csv(out_dir / "top20_clean_low_risk.csv", sorted(rows, key=lambda r: (r["val_acc"], r["macro_f1"], r["score"]), reverse=True)[:20])
        single_rows = [r for r in rows if r.get("candidate_type") not in {"clean_fusion", "clean_bias_calibration", "clean_confidence_binned_calibration"}]
        single_best = max(single_rows, key=lambda r: (r["val_acc"], r["macro_f1"], r["score"]))
        results.update(
            {
                "status": "completed",
                "best_clean": best,
                "best_clean_single": single_best,
                "clean_single_exceeds_0_6222": bool(single_best["val_acc"] > CURRENT_CLEAN_BEST + 1e-12),
                "clean_fusion_exceeds_0_6222": bool(best["val_acc"] > CURRENT_CLEAN_BEST + 1e-12),
                "clean_reaches_0_65": bool(best["val_acc"] >= 0.65),
                "excluded_count": len(excluded),
                "risk_registry_count": len(registry),
                "seed_validation": validate_seed(out_dir / "best_clean_low_risk_SEED.txt"),
                "best_prob_shapes": {"val": list(best_v.shape), "test": list(best_t.shape)},
            }
        )
    except Exception as exc:
        results["status"] = "failed_partial"
        results["errors"].append({"stage": "main", "error": repr(exc), "trace": traceback.format_exc()})
        log(logs, f"ERROR: {repr(exc)}")
    finally:
        (out_dir / "experiment_log.md").write_text("\n".join(logs) + "\n", encoding="utf-8")
        write_json(out_dir / "run_results.json", results)
        print(f"OUTPUT_DIR={out_dir}", flush=True)


if __name__ == "__main__":
    main()
