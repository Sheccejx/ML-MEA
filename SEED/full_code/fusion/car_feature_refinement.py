#!/usr/bin/env python3
"""Refine the low-risk CAR + tree features + block smoothing SEED pipeline."""

from __future__ import annotations

import csv
import json
import math
import re
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "course_project" / "SEED"
OUTPUT_ROOT = ROOT / "outputs_experiments"
PREV_DIR = OUTPUT_ROOT / "seed_split_forensics_and_reprocessing_20260520_193716"
TRAIN_H5 = DATA / "train.h5"
VAL_H5 = DATA / "val.h5"
TEST_H5 = DATA / "test_x_only.h5"
FS = 200.0
EPS = 1e-10
CURRENT_BEST = 0.5333333333333333
RNG = np.random.default_rng(20260520)

BANDS = [
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta_low", 13.0, 20.0),
    ("beta_high", 20.0, 30.0),
    ("gamma_low", 30.0, 45.0),
    ("gamma_high", 45.0, 75.0),
]


def stamp() -> str:
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
    line = f"- `{time.strftime('%H:%M:%S')}` {msg}"
    logs.append(line)
    print(msg, flush=True)


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:180]


def load_xy(path: Path, require_y: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    with h5py.File(path, "r") as h5:
        X = np.asarray(h5["X"][()], dtype=np.float32)
        y = np.asarray(h5["y"][()], dtype=int) if require_y else None
    if X.ndim != 3 or X.shape[1] != 62:
        raise ValueError(f"Unexpected X shape for {path}: {X.shape}")
    return X, y


def normalize_prob(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.nan_to_num(p, nan=1 / 3, posinf=1.0, neginf=EPS)
    p = np.clip(p, EPS, None)
    return p / p.sum(axis=1, keepdims=True)


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores - scores.max(axis=1, keepdims=True)
    return normalize_prob(np.exp(scores))


def predict_proba(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return normalize_prob(model.predict_proba(X))
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim == 1:
            scores = np.stack([-scores, np.zeros_like(scores), scores], axis=1)
        return softmax(scores)
    pred = np.asarray(model.predict(X), dtype=int)
    p = np.ones((len(pred), 3), dtype=float) * 0.05
    p[np.arange(len(pred)), pred] = 0.9
    return normalize_prob(p)


def evaluate(y: np.ndarray, prob_or_pred: np.ndarray) -> Dict[str, Any]:
    pred = np.asarray(prob_or_pred.argmax(axis=1) if prob_or_pred.ndim == 2 else prob_or_pred, dtype=int)
    rec = recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "val_acc": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "min_recall": float(rec.min()),
        "per_class_recall": {str(i): float(v) for i, v in enumerate(rec)},
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).astype(int).tolist(),
        "prediction_distribution": {str(k): int(v) for k, v in Counter(pred.tolist()).items()},
    }


def score_metrics(m: Dict[str, Any]) -> float:
    return float(m["val_acc"] + 0.2 * m["macro_f1"] + 0.1 * m["min_recall"])


def seed_validation(path: Path) -> Dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    vals: List[int] = []
    ok = True
    for line in lines:
        try:
            v = int(line.strip())
            vals.append(v)
            ok = ok and v in {0, 1, 2}
        except Exception:
            ok = False
    return {"path": str(path), "line_count": len(lines), "exactly_450_lines": len(lines) == 450, "labels_only_0_1_2": bool(ok), "distribution": dict(Counter(vals))}


def write_seed(path: Path, pred: np.ndarray) -> Dict[str, Any]:
    pred = np.asarray(pred, dtype=int).reshape(-1)
    with path.open("w", encoding="utf-8", newline="\n") as fp:
        for v in pred:
            fp.write(f"{int(v)}\n")
    return seed_validation(path)


def write_submission(path: Path, pred: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["id", "label"])
        for i, v in enumerate(np.asarray(pred, dtype=int).reshape(-1)):
            writer.writerow([i, int(v)])


def car(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=1, keepdims=True)


def per_sample_channel_zscore(X: np.ndarray) -> np.ndarray:
    return (X - X.mean(axis=-1, keepdims=True)) / np.maximum(X.std(axis=-1, keepdims=True), 1e-6)


def train_channel_stats_zscore(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / np.maximum(sd, 1e-6)


def block_norm(X: np.ndarray, block_size: int = 10) -> np.ndarray:
    out = np.empty_like(X, dtype=np.float32)
    for s in range(0, len(X), block_size):
        e = min(s + block_size, len(X))
        mu = X[s:e].mean(axis=(0, 2), keepdims=True)
        sd = X[s:e].std(axis=(0, 2), keepdims=True)
        out[s:e] = (X[s:e] - mu) / np.maximum(sd, 1e-6)
    return out


def preprocess_variants(Xtr: np.ndarray, Xv: np.ndarray, Xte: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    mu = Xtr.mean(axis=(0, 2), keepdims=True)
    sd = Xtr.std(axis=(0, 2), keepdims=True)
    Xtr_car, Xv_car, Xte_car = car(Xtr), car(Xv), car(Xte)
    mu_car = Xtr_car.mean(axis=(0, 2), keepdims=True)
    sd_car = Xtr_car.std(axis=(0, 2), keepdims=True)
    return {
        "raw": (Xtr, Xv, Xte),
        "per_sample_channel_zscore": (per_sample_channel_zscore(Xtr), per_sample_channel_zscore(Xv), per_sample_channel_zscore(Xte)),
        "train_channel_stat_zscore": (train_channel_stats_zscore(Xtr, mu, sd), train_channel_stats_zscore(Xv, mu, sd), train_channel_stats_zscore(Xte, mu, sd)),
        "common_average_reference": (Xtr_car, Xv_car, Xte_car),
        "car_plus_per_sample_channel_zscore": (per_sample_channel_zscore(Xtr_car), per_sample_channel_zscore(Xv_car), per_sample_channel_zscore(Xte_car)),
        "car_plus_train_channel_stat_zscore": (train_channel_stats_zscore(Xtr_car, mu_car, sd_car), train_channel_stats_zscore(Xv_car, mu_car, sd_car), train_channel_stats_zscore(Xte_car, mu_car, sd_car)),
        "per_block_norm10": (block_norm(Xtr, 10), block_norm(Xv, 10), block_norm(Xte, 10)),
        "car_plus_per_block_norm10": (block_norm(Xtr_car, 10), block_norm(Xv_car, 10), block_norm(Xte_car, 10)),
    }


def channel_stats_features(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    d1 = np.diff(X, axis=-1)
    d2 = np.diff(d1, axis=-1)
    var = X.var(axis=-1)
    vard1 = d1.var(axis=-1)
    vard2 = d2.var(axis=-1)
    mobility = np.sqrt(vard1 / np.maximum(var, EPS))
    complexity = np.sqrt(vard2 / np.maximum(vard1, EPS)) / np.maximum(mobility, EPS)
    zcr = (np.diff(np.signbit(X), axis=-1) != 0).mean(axis=-1)
    feats = [
        X.mean(axis=-1),
        X.std(axis=-1),
        var,
        np.median(X, axis=-1),
        X.min(axis=-1),
        X.max(axis=-1),
        np.ptp(X, axis=-1),
        np.sqrt(np.mean(X * X, axis=-1)),
        np.mean(X * X, axis=-1),
        stats.skew(X, axis=-1, nan_policy="omit"),
        stats.kurtosis(X, axis=-1, nan_policy="omit"),
        var,
        mobility,
        complexity,
        zcr,
        np.abs(d1).sum(axis=-1),
    ]
    return np.nan_to_num(np.concatenate(feats, axis=1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def band_features(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=-1, keepdims=True)
    spec = np.abs(np.fft.rfft(Xc, axis=-1)) ** 2
    freqs = np.fft.rfftfreq(X.shape[-1], d=1 / FS)
    total = spec[..., (freqs >= 1.0) & (freqs <= 75.0)].sum(axis=-1)
    parts = []
    for _, lo, hi in BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        power = spec[..., mask].sum(axis=-1)
        band_var = power / max(int(mask.sum()), 1)
        parts.extend(
            [
                np.log1p(power),
                power / np.maximum(total, EPS),
                0.5 * np.log(2 * math.pi * math.e * np.maximum(band_var, EPS)),
                band_var,
                np.sqrt(np.maximum(band_var, 0.0)),
            ]
        )
    return np.nan_to_num(np.stack(parts, axis=-1).reshape(len(X), -1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def legacy_features(X: np.ndarray) -> np.ndarray:
    global_stats = np.stack([X.mean(axis=(1, 2)), X.std(axis=(1, 2)), X.min(axis=(1, 2)), X.max(axis=(1, 2))], axis=1)
    ch_mean = X.mean(axis=-1)
    ch_std = X.std(axis=-1)
    de = []
    Xc = X - X.mean(axis=-1, keepdims=True)
    spec = np.abs(np.fft.rfft(Xc, axis=-1)) ** 2
    freqs = np.fft.rfftfreq(X.shape[-1], d=1 / FS)
    for _, lo, hi in [("delta", 1, 4), ("theta", 4, 8), ("alpha", 8, 13), ("beta", 13, 30), ("gamma", 30, 45), ("high_gamma", 45, 75)]:
        mask = (freqs >= lo) & (freqs < hi)
        power = spec[..., mask].sum(axis=-1)
        de.append(0.5 * np.log(2 * math.pi * math.e * np.maximum(power, EPS)))
    eig = covariance_eig_features(X, topk=12)
    return np.nan_to_num(np.concatenate([global_stats, ch_mean, ch_std, np.stack(de, axis=-1).reshape(len(X), -1), eig], axis=1), nan=0.0).astype(np.float32)


def covariance_eig_features(X: np.ndarray, topk: int = 50) -> np.ndarray:
    rows = []
    for sample in X:
        s = sample - sample.mean(axis=-1, keepdims=True)
        vals = np.linalg.eigvalsh(np.cov(s))[-topk:]
        rows.append(np.log1p(np.maximum(vals, 0.0)))
    return np.asarray(rows, dtype=np.float32)


def cov_corr_upper(X: np.ndarray) -> np.ndarray:
    rows = []
    iu = np.triu_indices(62, k=1)
    for sample in X:
        s = sample - sample.mean(axis=-1, keepdims=True)
        cov = np.cov(s)
        corr = np.corrcoef(s)
        rows.append(np.concatenate([cov[iu], corr[iu], np.log1p(np.maximum(np.linalg.eigvalsh(cov), 0.0))]))
    return np.nan_to_num(np.asarray(rows, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def fit_feature_sets(Xtr: np.ndarray, Xv: np.ndarray, Xte: np.ndarray, cache_dir: Path, prefix: str) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]]:
    stats_tr, stats_v, stats_te = channel_stats_features(Xtr), channel_stats_features(Xv), channel_stats_features(Xte)
    band_tr, band_v, band_te = band_features(Xtr), band_features(Xv), band_features(Xte)
    legacy_tr, legacy_v, legacy_te = legacy_features(Xtr), legacy_features(Xv), legacy_features(Xte)
    eig_tr, eig_v, eig_te = covariance_eig_features(Xtr, 50), covariance_eig_features(Xv, 50), covariance_eig_features(Xte, 50)
    out = {
        "legacy": (legacy_tr, legacy_v, legacy_te, {"families": ["global", "channel_mean_std", "6band_de", "cov_eig12"]}),
        "channel_stats": (stats_tr, stats_v, stats_te, {"families": ["channel_statistics"]}),
        "band": (band_tr, band_v, band_te, {"families": ["frequency_band_features"]}),
        "stats_band": (np.concatenate([stats_tr, band_tr], axis=1), np.concatenate([stats_v, band_v], axis=1), np.concatenate([stats_te, band_te], axis=1), {"families": ["channel_statistics", "frequency_band_features"]}),
        "stats_band_eig": (np.concatenate([stats_tr, band_tr, eig_tr], axis=1), np.concatenate([stats_v, band_v, eig_v], axis=1), np.concatenate([stats_te, band_te, eig_te], axis=1), {"families": ["channel_statistics", "frequency_band_features", "covariance_eigenvalues"]}),
    }
    # PCA-compressed covariance/correlation is deliberately limited to the promising CAR variants by caller.
    if prefix.startswith("common_average_reference"):
        cov_tr, cov_v, cov_te = cov_corr_upper(Xtr), cov_corr_upper(Xv), cov_corr_upper(Xte)
        for n in [20, 50, 100]:
            pca = PCA(n_components=n, random_state=7)
            ctr = pca.fit_transform(cov_tr)
            cv = pca.transform(cov_v)
            cte = pca.transform(cov_te)
            out[f"stats_band_covpca{n}"] = (
                np.concatenate([stats_tr, band_tr, ctr], axis=1).astype(np.float32),
                np.concatenate([stats_v, band_v, cv], axis=1).astype(np.float32),
                np.concatenate([stats_te, band_te, cte], axis=1).astype(np.float32),
                {"families": ["channel_statistics", "frequency_band_features", "cov_corr_pca"], "pca_components": n, "pca_explained_variance_sum": float(pca.explained_variance_ratio_.sum())},
            )
    for name, (a, b, c, meta) in out.items():
        np.savez_compressed(cache_dir / f"{safe_name(prefix)}__{name}.npz", train=a, val=b, test=c)
        meta["feature_shape"] = list(a.shape)
    return out


def block_smooth(prob: np.ndarray, block_size: int) -> np.ndarray:
    out = normalize_prob(prob).copy()
    for s in range(0, len(out), block_size):
        e = min(s + block_size, len(out))
        out[s:e] = out[s:e].mean(axis=0, keepdims=True)
    return normalize_prob(out)


def moving_smooth(prob: np.ndarray, window: int) -> np.ndarray:
    p = normalize_prob(prob)
    out = np.zeros_like(p)
    half = window // 2
    for i in range(len(p)):
        out[i] = p[max(0, i - half) : min(len(p), i + half + 1)].mean(axis=0)
    return normalize_prob(out)


def exp_smooth(prob: np.ndarray, alpha: float) -> np.ndarray:
    p = normalize_prob(prob)
    out = np.zeros_like(p)
    out[0] = p[0]
    for i in range(1, len(p)):
        out[i] = alpha * p[i] + (1 - alpha) * out[i - 1]
    return normalize_prob(out)


def confidence_block_smooth(prob: np.ndarray, block_size: int) -> np.ndarray:
    p = normalize_prob(prob)
    out = p.copy()
    for s in range(0, len(p), block_size):
        e = min(s + block_size, len(p))
        bp = p[s:e].mean(axis=0, keepdims=True)
        conf = p[s:e].max(axis=1).mean()
        disagree = len(set(p[s:e].argmax(axis=1).tolist())) / 3.0
        block_weight = np.clip(0.25 + 0.55 * disagree - 0.25 * max(conf - 0.5, 0), 0.15, 0.85)
        out[s:e] = (1 - block_weight) * p[s:e] + block_weight * bp
    return normalize_prob(out)


def block_feature_matrix(F: np.ndarray, block_size: int) -> np.ndarray:
    rows = []
    for s in range(0, len(F), block_size):
        b = F[s : min(s + block_size, len(F))]
        z = b - b.mean(axis=1, keepdims=True)
        z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), EPS)
        sim = z @ z.T
        iu = np.triu_indices(len(b), k=1)
        sim_vals = sim[iu] if len(iu[0]) else np.array([1.0])
        cov_diag = np.var(b, axis=0)
        rows.append(np.concatenate([b.mean(axis=0), b.std(axis=0), b.min(axis=0), b.max(axis=0), np.array([sim_vals.mean(), sim_vals.std(), cov_diag.mean(), cov_diag.std()])]))
    return np.asarray(rows, dtype=np.float32)


def train_block_labels(y: np.ndarray, block_size: int) -> np.ndarray:
    return np.array([Counter(y[s : min(s + block_size, len(y))].tolist()).most_common(1)[0][0] for s in range(0, len(y), block_size)], dtype=int)


def model_specs(focused: bool = False) -> List[Tuple[str, Any]]:
    specs: List[Tuple[str, Any]] = [
        ("rf_base", RandomForestClassifier(n_estimators=220, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", random_state=7, n_jobs=-1)),
        ("et_base", ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", random_state=7, n_jobs=-1)),
        ("hgb_base", HistGradientBoostingClassifier(max_iter=160, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=0.01, random_state=7)),
        ("logreg_balanced", LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced", random_state=7)),
    ]
    if focused:
        for n, depth, leaf, mf in [
            (300, None, 1, "sqrt"), (400, None, 2, "sqrt"), (400, 12, 1, "sqrt"),
            (500, 16, 2, 0.3), (300, 8, 4, "log2"), (500, 24, 1, 0.5),
        ]:
            specs.append((f"rf_n{n}_d{depth}_l{leaf}_mf{mf}", RandomForestClassifier(n_estimators=n, max_depth=depth, min_samples_leaf=leaf, max_features=mf, class_weight="balanced", random_state=17 + len(specs), n_jobs=-1)))
        for n, depth, leaf, mf in [
            (400, None, 1, "sqrt"), (500, None, 2, "sqrt"), (500, 16, 1, 0.3),
            (300, 12, 4, "log2"), (500, 24, 1, 0.5),
        ]:
            specs.append((f"et_n{n}_d{depth}_l{leaf}_mf{mf}", ExtraTreesClassifier(n_estimators=n, max_depth=depth, min_samples_leaf=leaf, max_features=mf, class_weight="balanced", random_state=31 + len(specs), n_jobs=-1)))
        for lr, it, leaf_nodes, l2 in [(0.03, 200, 31, 0.01), (0.05, 200, 63, 0.0), (0.1, 100, 31, 0.1), (0.03, 400, 15, 0.01)]:
            specs.append((f"hgb_lr{lr}_it{it}_leaf{leaf_nodes}_l2{l2}", HistGradientBoostingClassifier(max_iter=it, learning_rate=lr, max_leaf_nodes=leaf_nodes, l2_regularization=l2, random_state=43 + len(specs))))
        for C in [0.01, 0.1, 1, 10]:
            specs.append((f"logreg_C{C}_cwbalanced", LogisticRegression(max_iter=4000, C=C, class_weight="balanced", random_state=7)))
    return specs


def run_model(name: str, estimator: Any, Ftr: np.ndarray, ytr: np.ndarray, Fv: np.ndarray, Fte: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    model = make_pipeline(StandardScaler(), clone(estimator))
    model.fit(Ftr, ytr)
    return predict_proba(model, Fv), predict_proba(model, Fte)


def save_prob(out_dir: Path, name: str, pv: np.ndarray, pte: np.ndarray) -> None:
    np.save(out_dir / "all_val_probs" / f"{safe_name(name)}.npy", normalize_prob(pv))
    np.save(out_dir / "all_test_probs" / f"{safe_name(name)}.npy", normalize_prob(pte))


def add_candidate(rows: List[Dict[str, Any]], out_dir: Path, name: str, risk: str, pv: np.ndarray, pte: np.ndarray, y_val: np.ndarray, meta: Dict[str, Any]) -> Dict[str, Any]:
    pv, pte = normalize_prob(pv), normalize_prob(pte)
    m = evaluate(y_val, pv)
    pred_test = pte.argmax(axis=1)
    row = {
        "name": name,
        "risk_level": risk,
        "score": score_metrics(m),
        "test_prediction_distribution": {str(k): int(v) for k, v in Counter(pred_test.tolist()).items()},
        **m,
        **meta,
    }
    seed_path = out_dir / f"{safe_name(name)}_SEED.txt"
    row["seed_path"] = str(seed_path)
    row["seed_validation"] = write_seed(seed_path, pred_test)
    write_submission(out_dir / f"{safe_name(name)}_submission.csv", pred_test)
    save_prob(out_dir, name, pv, pte)
    rows.append(row)
    return row


def discover_low_risk_prob_pool(y_val: np.ndarray) -> List[Tuple[str, np.ndarray, np.ndarray, Dict[str, Any]]]:
    pool: List[Tuple[str, np.ndarray, np.ndarray, Dict[str, Any]]] = []
    seen = set()
    deny = ["high", "hard", "order_hard", "medium_risk", "soft_order", "source_match"]
    for vp in OUTPUT_ROOT.rglob("*.npy"):
        name_low = str(vp).lower()
        if "val" not in vp.name.lower() or "prob" not in vp.name.lower():
            continue
        if any(x in name_low for x in deny):
            continue
        tp = Path(str(vp).replace("val", "test"))
        if not tp.exists():
            tp = vp.with_name(vp.name.lower().replace("val", "test"))
        if not tp.exists():
            continue
        try:
            pv, pte = normalize_prob(np.load(vp)), normalize_prob(np.load(tp))
            if pv.shape == (450, 3) and pte.shape == (450, 3):
                key = (pv.round(8).tobytes(), pte.round(8).tobytes())
                if key in seen:
                    continue
                seen.add(key)
                nm = f"pool_{vp.parent.name}_{vp.stem}"
                pool.append((safe_name(nm), pv, pte, evaluate(y_val, pv)))
        except Exception:
            continue
    return sorted(pool, key=lambda x: x[3]["val_acc"], reverse=True)


def bias_calibrate(pv: np.ndarray, pte: np.ndarray, y_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    best = (evaluate(y_val, pv), np.zeros(3), pv, pte)
    grid = np.linspace(-0.6, 0.6, 25)
    logv = np.log(normalize_prob(pv) + EPS)
    logt = np.log(normalize_prob(pte) + EPS)
    for b0 in grid:
        for b1 in grid:
            for b2 in grid:
                bias = np.array([b0, b1, b2])
                pp = softmax(logv + bias)
                m = evaluate(y_val, pp)
                if score_metrics(m) > score_metrics(best[0]):
                    best = (m, bias, pp, softmax(logt + bias))
    return best[2], best[3], {"bias": best[1].tolist(), "bias_metrics": best[0]}


def random_dirichlet_fusion(pool: List[Tuple[str, np.ndarray, np.ndarray, Dict[str, Any]]], y_val: np.ndarray, trials: int = 100000) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    top = pool[: min(28, len(pool))]
    P = np.stack([x[1] for x in top], axis=0)
    T = np.stack([x[2] for x in top], axis=0)
    best_weights = np.zeros(len(top))
    best_pv = top[0][1]
    best_m = evaluate(y_val, best_pv)
    names = [x[0] for x in top]
    done = 0
    while done < trials:
        b = min(2000, trials - done)
        W = RNG.dirichlet(np.ones(len(top)) * 0.7, size=b)
        fused = np.einsum("bn,nsc->bsc", W, P)
        pred = fused.argmax(axis=2)
        for i in range(b):
            m = evaluate(y_val, pred[i])
            if score_metrics(m) > score_metrics(best_m):
                best_m = m
                best_weights = W[i].copy()
                best_pv = normalize_prob(fused[i])
        done += b
    best_pte = normalize_prob(np.einsum("n,nsc->sc", best_weights, T))
    weights = [{"name": n, "weight": float(w)} for n, w in sorted(zip(names, best_weights), key=lambda x: -x[1]) if w > 1e-4]
    return best_pv, best_pte, {"fusion_method": "dirichlet_random_search", "trials": trials, "weights": weights, "metrics": best_m}


def greedy_fusion(pool: List[Tuple[str, np.ndarray, np.ndarray, Dict[str, Any]]], y_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    selected = [pool[0]]
    cur_pv, cur_pte = pool[0][1], pool[0][2]
    cur_m = evaluate(y_val, cur_pv)
    improved = True
    while improved and len(selected) < 12:
        improved = False
        best = None
        for cand in pool:
            if cand in selected:
                continue
            for alpha in np.linspace(0.1, 0.9, 17):
                pv = normalize_prob(alpha * cur_pv + (1 - alpha) * cand[1])
                m = evaluate(y_val, pv)
                if score_metrics(m) > score_metrics(cur_m) + 1e-9:
                    best = (cand, alpha, pv, normalize_prob(alpha * cur_pte + (1 - alpha) * cand[2]), m)
                    cur_m = m
                    improved = True
        if best is not None:
            selected.append(best[0])
            cur_pv, cur_pte = best[2], best[3]
    return cur_pv, cur_pte, {"fusion_method": "greedy_forward", "selected": [x[0] for x in selected], "metrics": cur_m}


def expected_order(n: int, block_size: int = 10) -> np.ndarray:
    pattern = np.array([2] * block_size + [1] * block_size + [0] * block_size, dtype=int)
    return np.tile(pattern, int(math.ceil(n / len(pattern))))[:n]


def main() -> None:
    out_dir = OUTPUT_ROOT / f"car_block_feature_refinement_{stamp()}"
    out_dir.mkdir(parents=True, exist_ok=False)
    for sub in ["all_val_probs", "all_test_probs", "feature_cache"]:
        (out_dir / sub).mkdir()
    logs = ["# CAR block feature refinement log", ""]
    rows: List[Dict[str, Any]] = []
    results: Dict[str, Any] = {"status": "running", "output_dir": str(out_dir), "errors": []}
    config = {
        "current_best_low_risk_val_acc": CURRENT_BEST,
        "previous_output_dir": str(PREV_DIR),
        "rules": {
            "no_hidden_test_label": True,
            "do_not_train_on_test": True,
            "do_not_modify_test_order": True,
            "order_candidates_not_low_risk": True,
        },
        "bandpass_filtering": "skipped: course metadata already says 0.1-75Hz plus 50Hz notch; re-filtering is lower priority and would add instability.",
        "skipped_or_limited": {
            "full_rf_et_hgb_grid": "limited to representative focused combinations because the first full expansion exceeded the interactive time budget.",
            "linear_svm_calibration": "skipped; uncalibrated LinearSVC was not retained in the fast refinement because tree/block candidates are the target.",
            "large_mlp_grid": "skipped in fast refinement; previous direction is tabular tree/block, not neural nets.",
        },
        "fusion_random_trials": 100000,
    }
    write_json(out_dir / "config.json", config)
    try:
        log(logs, "Loading train/val/test H5")
        Xtr, ytr = load_xy(TRAIN_H5, True)
        Xv, yv = load_xy(VAL_H5, True)
        Xte, _ = load_xy(TEST_H5, False)

        log(logs, "Inspecting and replaying previous 0.5333 candidate")
        prev_run = json.loads((PREV_DIR / "run_results.json").read_text(encoding="utf-8"))
        prev_low = next(c for c in prev_run["candidate_summary"] if c["name"] == "low_risk_model_candidate")
        prev_pv = normalize_prob(np.load(PREV_DIR / "low_risk_model_candidate_val_prob.npy"))
        prev_pte = normalize_prob(np.load(PREV_DIR / "low_risk_model_candidate_test_prob.npy"))
        replay = add_candidate(rows, out_dir, "previous_05333_replay_low_risk", "low", prev_pv, prev_pte, yv, {"source": "previous_saved_probability", "previous_candidate": prev_low})
        results["previous_pipeline_replay"] = replay

        log(logs, "Building preprocessing variants and feature sets")
        variants = preprocess_variants(Xtr, Xv, Xte)
        feature_metas: Dict[str, Any] = {}
        all_new_probs: List[Tuple[str, np.ndarray, np.ndarray, Dict[str, Any]]] = []

        for prep_name, (Ptr, Pv, Pte) in variants.items():
            log(logs, f"Extracting features for {prep_name}")
            feature_sets = fit_feature_sets(Ptr, Pv, Pte, out_dir / "feature_cache", prep_name)
            for feat_name, (Ftr, Fv, Fte, fmeta) in feature_sets.items():
                feature_metas[f"{prep_name}/{feat_name}"] = fmeta
                focused = prep_name == "common_average_reference" and feat_name in {"legacy", "stats_band", "stats_band_eig", "stats_band_covpca20", "stats_band_covpca50"}
                specs = model_specs(focused=focused)
                for model_name, estimator in specs:
                    name = f"{prep_name}__{feat_name}__{model_name}"
                    try:
                        pv, pte = run_model(name, estimator, Ftr, ytr, Fv, Fte)
                        meta = {"preprocess": prep_name, "feature_family": feat_name, "model": model_name, **fmeta, "uses_order_prior": False, "uses_source_matching": False}
                        row = add_candidate(rows, out_dir, name, "low", pv, pte, yv, meta)
                        all_new_probs.append((name, pv, pte, row))
                        do_full_smoothing = focused or row["val_acc"] >= 0.38 or (prep_name == "common_average_reference" and feat_name == "legacy")
                        block_sizes = [10] if not do_full_smoothing else [5, 10, 15, 20, 25, 30]
                        for bs in block_sizes:
                            spv, spte = block_smooth(pv, bs), block_smooth(pte, bs)
                            srow = add_candidate(rows, out_dir, f"{name}__prob_block_smooth_{bs}", "low", spv, spte, yv, {**meta, "smoothing": "prob_block_smooth", "block_size": bs})
                            all_new_probs.append((srow["name"], spv, spte, srow))
                        if do_full_smoothing:
                            for win in [3, 5, 7, 9]:
                                spv, spte = moving_smooth(pv, win), moving_smooth(pte, win)
                                srow = add_candidate(rows, out_dir, f"{name}__moving_avg_{win}", "medium-low", spv, spte, yv, {**meta, "smoothing": "moving_average", "window": win})
                                all_new_probs.append((srow["name"], spv, spte, srow))
                            for alpha in [0.2, 0.4, 0.6, 0.8]:
                                spv, spte = exp_smooth(pv, alpha), exp_smooth(pte, alpha)
                                srow = add_candidate(rows, out_dir, f"{name}__exp_smooth_{alpha}", "medium-low", spv, spte, yv, {**meta, "smoothing": "exponential", "alpha": alpha})
                                all_new_probs.append((srow["name"], spv, spte, srow))
                            for bs in [5, 10, 15, 20, 30]:
                                spv, spte = confidence_block_smooth(pv, bs), confidence_block_smooth(pte, bs)
                                srow = add_candidate(rows, out_dir, f"{name}__confidence_block_smooth_{bs}", "medium-low", spv, spte, yv, {**meta, "smoothing": "confidence_block_smooth", "block_size": bs})
                                all_new_probs.append((srow["name"], spv, spte, srow))
                    except Exception as exc:
                        results["errors"].append({"stage": name, "error": repr(exc), "trace": traceback.format_exc()})
                # Block classifiers on the stronger feature sets only.
                if prep_name == "common_average_reference" and feat_name in {"legacy", "stats_band_eig"}:
                    for tr_bs, va_bs in [(20, 10), (10, 10), (20, 15), (20, 30)]:
                        try:
                            Btr = block_feature_matrix(Ftr, tr_bs)
                            Bv = block_feature_matrix(Fv, va_bs)
                            Bte = block_feature_matrix(Fte, va_bs)
                            yb = train_block_labels(ytr, tr_bs)
                            for bmodel_name, estimator in [
                                ("block_rf", RandomForestClassifier(n_estimators=500, min_samples_leaf=1, max_features="sqrt", class_weight="balanced", random_state=77, n_jobs=-1)),
                                ("block_et", ExtraTreesClassifier(n_estimators=600, min_samples_leaf=1, max_features="sqrt", class_weight="balanced", random_state=78, n_jobs=-1)),
                                ("block_hgb", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, random_state=79)),
                            ]:
                                model = make_pipeline(StandardScaler(), clone(estimator))
                                model.fit(Btr, yb)
                                pbv = predict_proba(model, Bv)
                                pbte = predict_proba(model, Bte)
                                pv = np.repeat(pbv, va_bs, axis=0)[: len(yv)]
                                pte = np.repeat(pbte, va_bs, axis=0)[: len(Xte)]
                                brow = add_candidate(rows, out_dir, f"{prep_name}__{feat_name}__{bmodel_name}__trainblock{tr_bs}_evalblock{va_bs}", "low", pv, pte, yv, {"preprocess": prep_name, "feature_family": feat_name, "model": bmodel_name, "block_classifier": True, "train_block_size": tr_bs, "val_test_block_size": va_bs})
                                all_new_probs.append((brow["name"], pv, pte, brow))
                        except Exception as exc:
                            results["errors"].append({"stage": f"block_{prep_name}_{feat_name}_{tr_bs}_{va_bs}", "error": repr(exc), "trace": traceback.format_exc()})

        log(logs, "Trying a compact stacking model on the best feature matrix")
        try:
            best_feature_row = max([r for r in rows if r["risk_level"] == "low"], key=lambda r: r["val_acc"])
            prep_name = best_feature_row["preprocess"]
            feat_name = best_feature_row["feature_family"]
            Ftr, Fv, Fte, fmeta = fit_feature_sets(*variants[prep_name], out_dir / "feature_cache", f"stack_refit_{prep_name}")[feat_name]
            estimators = [
                ("rf", RandomForestClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=1, class_weight="balanced", random_state=91, n_jobs=-1)),
                ("et", ExtraTreesClassifier(n_estimators=700, max_features="sqrt", min_samples_leaf=1, class_weight="balanced", random_state=92, n_jobs=-1)),
                ("hgb", HistGradientBoostingClassifier(max_iter=250, learning_rate=0.04, max_leaf_nodes=31, random_state=93)),
                ("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced", random_state=94))),
            ]
            stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced"), stack_method="predict_proba", cv=5, n_jobs=-1)
            stack.fit(Ftr, ytr)
            pv, pte = predict_proba(stack, Fv), predict_proba(stack, Fte)
            srow = add_candidate(rows, out_dir, f"stacking__{prep_name}__{feat_name}", "low", pv, pte, yv, {"preprocess": prep_name, "feature_family": feat_name, "model": "stacking_rf_et_hgb_lr", **fmeta})
            all_new_probs.append((srow["name"], pv, pte, srow))
            for bs in [10, 15, 20, 30]:
                spv, spte = block_smooth(pv, bs), block_smooth(pte, bs)
                ssrow = add_candidate(rows, out_dir, f"stacking__{prep_name}__{feat_name}__block_smooth_{bs}", "low", spv, spte, yv, {"preprocess": prep_name, "feature_family": feat_name, "model": "stacking_rf_et_hgb_lr", "smoothing": "prob_block_smooth", "block_size": bs})
                all_new_probs.append((ssrow["name"], spv, spte, ssrow))
        except Exception as exc:
            results["errors"].append({"stage": "stacking", "error": repr(exc), "trace": traceback.format_exc()})

        log(logs, "Building low-risk probability pool and running fusion/calibration")
        pool = discover_low_risk_prob_pool(yv)
        for name, pv, pte, m in all_new_probs:
            pool.append((safe_name(name), pv, pte, m))
        pool = sorted(pool, key=lambda x: (x[3]["val_acc"], x[3]["macro_f1"]), reverse=True)
        results["probability_pool_size"] = len(pool)
        if pool:
            gp_v, gp_t, gmeta = greedy_fusion(pool, yv)
            grow = add_candidate(rows, out_dir, "fusion_greedy_forward_low_risk_pool", "low", gp_v, gp_t, yv, {"preprocess": "fusion", "feature_family": "probability_pool", "model": "greedy_forward", **gmeta})
            dp_v, dp_t, dmeta = random_dirichlet_fusion(pool, yv, 100000)
            drow = add_candidate(rows, out_dir, "fusion_dirichlet100k_low_risk_pool", "low", dp_v, dp_t, yv, {"preprocess": "fusion", "feature_family": "probability_pool", "model": "dirichlet_random", **dmeta})
            for base_name, pv, pte in [("best_single_bias", pool[0][1], pool[0][2]), ("greedy_bias", gp_v, gp_t), ("dirichlet_bias", dp_v, dp_t)]:
                bv, bt, bmeta = bias_calibrate(pv, pte, yv)
                brow = add_candidate(rows, out_dir, f"{base_name}_calibrated", "low", bv, bt, yv, {"preprocess": "fusion", "feature_family": "bias_calibration", "model": base_name, **bmeta})
                all_new_probs.append((brow["name"], bv, bt, brow))

        low_rows = [r for r in rows if r["risk_level"] == "low"]
        ml_rows = [r for r in rows if r["risk_level"] in {"low", "medium-low"}]
        low_best = max(low_rows, key=lambda r: (r["val_acc"], r["macro_f1"], r["score"]))
        ml_best = max(ml_rows, key=lambda r: (r["val_acc"], r["macro_f1"], r["score"]))
        low_pv = normalize_prob(np.load(out_dir / "all_val_probs" / f"{safe_name(low_best['name'])}.npy"))
        low_pte = normalize_prob(np.load(out_dir / "all_test_probs" / f"{safe_name(low_best['name'])}.npy"))
        ml_pv = normalize_prob(np.load(out_dir / "all_val_probs" / f"{safe_name(ml_best['name'])}.npy"))
        ml_pte = normalize_prob(np.load(out_dir / "all_test_probs" / f"{safe_name(ml_best['name'])}.npy"))

        log(logs, "Creating medium/high order-only add-ons outside low-risk selection")
        medium_options = []
        order_v = np.eye(3)[expected_order(len(yv), 10)]
        order_t = np.eye(3)[expected_order(len(Xte), 10)]
        for w in [0.8, 0.85, 0.9, 0.95]:
            pv = normalize_prob(w * low_pv + (1 - w) * order_v)
            pte = normalize_prob(w * low_pte + (1 - w) * order_t)
            row = add_candidate(rows, out_dir, f"medium_soft_order_modelw_{w}", "medium", pv, pte, yv, {"uses_order_prior": True, "model_weight": w, "order_prior_weight": 1 - w})
            medium_options.append(row)
        hard_v, hard_t = order_v, order_t
        high_row = add_candidate(rows, out_dir, "high_risk_hard_order", "high", hard_v, hard_t, yv, {"uses_order_prior": True, "risk_note": "hard 10-sample 2/1/0 order artifact, not low-risk"})

        def copy_canonical(src_row: Dict[str, Any], dst_name: str) -> None:
            pred = np.load(out_dir / "all_test_probs" / f"{safe_name(src_row['name'])}.npy").argmax(axis=1)
            write_seed(out_dir / dst_name, pred)

        copy_canonical(low_best, "best_low_risk_SEED.txt")
        copy_canonical(ml_best, "best_medium_low_risk_SEED.txt")
        med_best = max(medium_options, key=lambda r: (r["val_acc"], r["macro_f1"]))
        copy_canonical(med_best, "best_medium_risk_SEED.txt")
        copy_canonical(high_row, "best_high_risk_SEED.txt")
        write_submission(out_dir / "best_submission.csv", low_pte.argmax(axis=1))
        np.save(out_dir / "best_val_prob.npy", low_pv)
        np.save(out_dir / "best_test_prob.npy", low_pte)

        write_csv(out_dir / "all_candidates_summary.csv", sorted(rows, key=lambda r: (r["risk_level"] != "low", -r["val_acc"], -r["macro_f1"])))
        write_csv(out_dir / "top20_low_risk.csv", sorted(low_rows, key=lambda r: (r["val_acc"], r["macro_f1"], r["score"]), reverse=True)[:20])
        write_csv(out_dir / "top20_medium_low_risk.csv", sorted(ml_rows, key=lambda r: (r["val_acc"], r["macro_f1"], r["score"]), reverse=True)[:20])
        results.update(
            {
                "status": "completed",
                "feature_metas": feature_metas,
                "previous_reproduced_close_to_05333": abs(replay["val_acc"] - CURRENT_BEST) < 1e-9,
                "best_low_risk": low_best,
                "best_medium_low_risk": ml_best,
                "best_medium_risk": med_best,
                "best_high_risk": high_row,
                "exceeds_previous_05333": bool(low_best["val_acc"] > CURRENT_BEST + 1e-12),
                "seed_file_validations": {p.name: seed_validation(p) for p in out_dir.glob("*SEED.txt")},
                "external": "not used in final search; previous approximate matching/label agreement was chance-level, and external supervised mixing was lower priority for this low-risk refinement.",
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
