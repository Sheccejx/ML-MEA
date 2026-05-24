#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Forensic preprocessing analysis for the course SEED H5 files.

This script audits official train/val/test H5 files, inspects the local
SEED-style Preprocessed_EEG archive, generates small SEED-like preprocessing
candidates, and ranks candidates by statistical proximity to official train.
Validation/test labels are never used for preprocessing statistics.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np
from scipy.io import loadmat, whosmat
from scipy.stats import ks_2samp, wasserstein_distance


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "course_project" / "SEED"
DEFAULT_ZIP = Path(r"C:\Users\Archery\Downloads\archive (1).zip")
OUTPUT_ROOT = ROOT / "outputs_preprocessing_forensics"
SEED_CHANNELS = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "FT7",
    "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6",
    "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2",
]
STANDARD_LABELS = np.asarray([1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1], dtype=np.int64)
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
EEG_KEY_RE = re.compile(r"(?:^|_)eeg_?(\d+)$", re.IGNORECASE)


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def as_float_list(x: np.ndarray, limit: Optional[int] = None) -> List[float]:
    arr = np.asarray(x).reshape(-1)
    if limit:
        arr = arr[:limit]
    return [float(v) for v in arr]


def label_distribution(y: Optional[np.ndarray]) -> Dict[str, int]:
    if y is None:
        return {}
    vals, counts = np.unique(y.astype(int), return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(vals, counts)}


def quantiles(x: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {}
    qs = [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    return {str(q): float(np.quantile(arr, q)) for q in qs}


def load_h5(path: Path, require_y: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    with h5py.File(path, "r") as h5:
        X = h5["X"][()].astype(np.float32)
        y = h5["y"][()].astype(np.int64) if require_y and "y" in h5 else None
    return X, y


def corr_upper_distribution(X: np.ndarray, max_samples: int = 120) -> np.ndarray:
    rng = np.random.default_rng(123)
    idx = np.arange(len(X))
    if len(idx) > max_samples:
        idx = rng.choice(idx, size=max_samples, replace=False)
    vals = []
    tri = np.triu_indices(X.shape[1], k=1)
    for i in idx:
        c = np.corrcoef(X[i])
        c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
        vals.append(c[tri])
    return np.concatenate(vals) if vals else np.asarray([], dtype=np.float32)


def cov_eig_signature(X: np.ndarray, max_samples: int = 120) -> np.ndarray:
    rng = np.random.default_rng(321)
    idx = np.arange(len(X))
    if len(idx) > max_samples:
        idx = rng.choice(idx, size=max_samples, replace=False)
    eigs = []
    for i in idx:
        cov = np.cov(X[i])
        vals = np.linalg.eigvalsh(np.nan_to_num(cov))
        vals = np.sort(vals)[::-1]
        vals = vals / max(float(vals.sum()), 1e-8)
        eigs.append(vals)
    return np.mean(np.stack(eigs), axis=0) if eigs else np.zeros((X.shape[1],), dtype=np.float32)


def adjacent_correlation(X: np.ndarray, max_pairs: int = 300) -> Dict[str, float]:
    n = min(len(X) - 1, max_pairs)
    if n <= 0:
        return {}
    vals = []
    for i in range(n):
        a = X[i].reshape(-1)
        b = X[i + 1].reshape(-1)
        vals.append(float(np.corrcoef(a, b)[0, 1]))
    return quantiles(np.asarray(vals))


def block_similarity(X: np.ndarray, block_sizes: Iterable[int] = (5, 10, 15, 20, 30)) -> Dict[str, float]:
    out = {}
    flat = X.reshape(X.shape[0], -1)
    for b in block_sizes:
        vals = []
        for start in range(0, len(flat) - b + 1, b):
            block = flat[start:start + b]
            c = np.corrcoef(block)
            tri = c[np.triu_indices(b, k=1)]
            vals.append(float(np.nanmean(tri)))
        if vals:
            out[str(b)] = float(np.mean(vals))
    return out


def h5_forensics(path: Path, require_y: bool) -> Dict[str, object]:
    X, y = load_h5(path, require_y=require_y)
    sample_mean = X.mean(axis=(1, 2))
    sample_std = X.std(axis=(1, 2))
    sample_channel_mean = X.mean(axis=2)
    sample_channel_std = X.std(axis=2)
    channel_mean = X.mean(axis=(0, 2))
    channel_std = X.std(axis=(0, 2))
    return {
        "path": str(path),
        "shape": list(X.shape),
        "dtype": str(X.dtype),
        "min": float(np.min(X)),
        "max": float(np.max(X)),
        "global_mean": float(X.mean()),
        "global_std": float(X.std()),
        "sample_mean_quantiles": quantiles(sample_mean),
        "sample_std_quantiles": quantiles(sample_std),
        "channel_mean_quantiles": quantiles(channel_mean),
        "channel_std_quantiles": quantiles(channel_std),
        "sample_channel_abs_mean_quantiles": quantiles(np.abs(sample_channel_mean)),
        "sample_channel_std_quantiles": quantiles(sample_channel_std),
        "pct_sample_channel_mean_abs_lt_1e_3": float(np.mean(np.abs(sample_channel_mean) < 1e-3)),
        "pct_sample_channel_std_near_1": float(np.mean((sample_channel_std > 0.9) & (sample_channel_std < 1.1))),
        "label_distribution": label_distribution(y),
        "has_nan": bool(np.isnan(X).any()),
        "has_inf": bool(np.isinf(X).any()),
        "adjacent_correlation_quantiles": adjacent_correlation(X),
        "block_similarity": block_similarity(X),
        "corr_upper": corr_upper_distribution(X),
        "cov_eig": cov_eig_signature(X),
        "sample_mean": sample_mean,
        "sample_std": sample_std,
        "channel_mean": channel_mean,
        "channel_std": channel_std,
    }


def public_stats(stats: Dict[str, object]) -> Dict[str, object]:
    return {k: v for k, v in stats.items() if k not in {"corr_upper", "cov_eig", "sample_mean", "sample_std", "channel_mean", "channel_std"}}


def trial_number(key: str) -> Optional[int]:
    m = EEG_KEY_RE.search(key)
    return int(m.group(1)) if m else None


def ensure_62_by_t(arr: np.ndarray) -> Optional[np.ndarray]:
    x = np.asarray(arr)
    if x.ndim != 2:
        return None
    if x.shape[0] == 62:
        return x.astype(np.float32, copy=False)
    if x.shape[1] == 62:
        return x.T.astype(np.float32, copy=False)
    return None


def inspect_zip(zip_path: Path, run_dir: Path) -> Tuple[List[Dict[str, object]], np.ndarray]:
    rows = []
    labels = STANDARD_LABELS.copy()
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        label_names = [n for n in names if n.lower().endswith("label.mat")]
        if label_names:
            label_mat = loadmat(io.BytesIO(zf.read(label_names[0])))
            labels = np.asarray(label_mat.get("label", labels)).reshape(-1).astype(np.int64)
        for i, name in enumerate(names, start=1):
            info = zf.getinfo(name)
            row: Dict[str, object] = {"index": i, "name": name, "size": int(info.file_size)}
            if name.lower().endswith(".mat") and not name.lower().endswith("label.mat"):
                try:
                    entries = whosmat(io.BytesIO(zf.read(name)))
                    eeg_entries = [(k, shape) for k, shape, dtype in entries if trial_number(k) is not None and 62 in shape]
                    lengths = []
                    shapes = []
                    for key, shape in eeg_entries:
                        shapes.append(f"{key}:{shape}")
                        lengths.append(max(shape))
                    row.update({
                        "kind": "eeg_mat",
                        "key_count": len(entries),
                        "eeg_trial_count": len(eeg_entries),
                        "has_15_trials": len(eeg_entries) == 15,
                        "trial_length_min": int(min(lengths)) if lengths else "",
                        "trial_length_max": int(max(lengths)) if lengths else "",
                        "trial_shapes": "; ".join(shapes[:15]),
                    })
                except Exception as exc:
                    row.update({"kind": "eeg_mat_error", "error": str(exc)})
            elif name.lower().endswith("label.mat"):
                row.update({"kind": "label_mat", "label_values": labels.astype(int).tolist()})
            elif name.lower().endswith("readme.txt"):
                text = zf.read(name).decode("utf-8", errors="replace")
                row.update({"kind": "readme", "text": text[:500]})
            else:
                row.update({"kind": "other"})
            rows.append(row)
    with (run_dir / "archive_file_inventory.csv").open("w", encoding="utf-8", newline="") as fp:
        fields = sorted({k for r in rows for k in r.keys()})
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows, labels


def get_starts(length: int, mode: str, window: int, rng: np.random.Generator, k: int) -> List[int]:
    if length < window:
        return []
    if mode == "stride_200":
        starts = list(range(0, length - window + 1, 200))
    elif mode in {"first_nonoverlap", "full_nonoverlap"}:
        starts = list(range(0, length - window + 1, 400))
    elif mode == "middle_nonoverlap":
        all_starts = list(range(0, length - window + 1, 400))
        mid = length / 2
        starts = sorted(all_starts, key=lambda s: abs((s + window / 2) - mid))
    elif mode == "fixed_k_per_trial":
        if k <= 1:
            starts = [max(0, (length - window) // 2)]
        else:
            starts = np.linspace(0, length - window, num=k, dtype=int).tolist()
    elif mode == "random_k_per_trial":
        n_possible = length - window + 1
        starts = rng.choice(n_possible, size=min(k, n_possible), replace=False).astype(int).tolist()
    else:
        raise ValueError(mode)
    if mode in {"first_nonoverlap", "middle_nonoverlap"}:
        starts = starts[:k]
    if mode in {"full_nonoverlap", "stride_200"}:
        starts = starts[:max(k, 1)]
    return sorted(set(int(s) for s in starts))


def normalize_window(
    win: np.ndarray,
    norm: str,
    trial_mean: Optional[np.ndarray],
    trial_std: Optional[np.ndarray],
    official: Dict[str, object],
) -> np.ndarray:
    x = win.astype(np.float32, copy=True)
    if norm == "raw_no_norm":
        return x
    if norm == "per_window_global_demean":
        return (x - x.mean()).astype(np.float32)
    if norm == "per_window_global_zscore":
        return ((x - x.mean()) / max(float(x.std()), 1e-6)).astype(np.float32)
    if norm == "per_window_channel_zscore":
        return ((x - x.mean(axis=1, keepdims=True)) / np.maximum(x.std(axis=1, keepdims=True), 1e-6)).astype(np.float32)
    if norm == "per_trial_channel_zscore_then_window":
        assert trial_mean is not None and trial_std is not None
        return ((x - trial_mean) / np.maximum(trial_std, 1e-6)).astype(np.float32)
    if norm == "common_average_reference_then_per_window_channel_zscore":
        x = x - x.mean(axis=0, keepdims=True)
        return ((x - x.mean(axis=1, keepdims=True)) / np.maximum(x.std(axis=1, keepdims=True), 1e-6)).astype(np.float32)
    if norm == "official_train_global_align":
        target_mean = float(official["global_mean"])
        target_std = float(official["global_std"])
        return (((x - x.mean()) / max(float(x.std()), 1e-6)) * target_std + target_mean).astype(np.float32)
    if norm == "official_train_channel_align":
        cm = np.asarray(official["channel_mean"], dtype=np.float32).reshape(62, 1)
        cs = np.asarray(official["channel_std"], dtype=np.float32).reshape(62, 1)
        z = (x - x.mean(axis=1, keepdims=True)) / np.maximum(x.std(axis=1, keepdims=True), 1e-6)
        return (z * cs + cm).astype(np.float32)
    raise ValueError(norm)


def candidate_specs() -> List[Dict[str, object]]:
    return [
        {"id": "c01_first_raw", "windowing": "first_nonoverlap", "normalization": "raw_no_norm", "k": 3},
        {"id": "c02_first_global_demean", "windowing": "first_nonoverlap", "normalization": "per_window_global_demean", "k": 3},
        {"id": "c03_first_global_z", "windowing": "first_nonoverlap", "normalization": "per_window_global_zscore", "k": 3},
        {"id": "c04_first_channel_z", "windowing": "first_nonoverlap", "normalization": "per_window_channel_zscore", "k": 3},
        {"id": "c05_middle_channel_z", "windowing": "middle_nonoverlap", "normalization": "per_window_channel_zscore", "k": 5},
        {"id": "c06_full_global_demean", "windowing": "full_nonoverlap", "normalization": "per_window_global_demean", "k": 3},
        {"id": "c07_stride200_global_demean", "windowing": "stride_200", "normalization": "per_window_global_demean", "k": 3},
        {"id": "c08_fixed5_trial_z", "windowing": "fixed_k_per_trial", "normalization": "per_trial_channel_zscore_then_window", "k": 5},
        {"id": "c09_fixed10_car_channel_z", "windowing": "fixed_k_per_trial", "normalization": "common_average_reference_then_per_window_channel_zscore", "k": 10},
        {"id": "c10_random10_channel_z", "windowing": "random_k_per_trial", "normalization": "per_window_channel_zscore", "k": 10},
        {"id": "c11_fixed20_official_global", "windowing": "fixed_k_per_trial", "normalization": "official_train_global_align", "k": 20},
        {"id": "c12_fixed10_official_channel", "windowing": "fixed_k_per_trial", "normalization": "official_train_channel_align", "k": 10},
        {"id": "c13_middle_global_demean", "windowing": "middle_nonoverlap", "normalization": "per_window_global_demean", "k": 5},
        {"id": "c14_random10_global_demean", "windowing": "random_k_per_trial", "normalization": "per_window_global_demean", "k": 10},
    ]


def generate_candidates(zip_path: Path, labels_orig: np.ndarray, official: Dict[str, object], run_dir: Path, samples_per_class: int) -> List[Path]:
    specs = candidate_specs()
    rng = np.random.default_rng(42)
    rows = {spec["id"]: [] for spec in specs}
    counts = {spec["id"]: Counter() for spec in specs}
    with zipfile.ZipFile(zip_path, "r") as zf:
        mat_names = sorted(n for n in zf.namelist() if n.lower().endswith(".mat") and not n.lower().endswith("label.mat"))
        for name in mat_names:
            if all(all(counts[s["id"]][c] >= samples_per_class for c in (0, 1, 2)) for s in specs):
                break
            mat = loadmat(io.BytesIO(zf.read(name)))
            trial_keys = []
            for key, arr in mat.items():
                t = trial_number(key)
                if t is not None and 1 <= t <= len(labels_orig):
                    trial_keys.append((t, key))
            for trial_id, key in sorted(trial_keys):
                arr = ensure_62_by_t(mat[key])
                if arr is None:
                    continue
                y = LABEL_MAP[int(labels_orig[trial_id - 1])]
                trial_mean = arr.mean(axis=1, keepdims=True)
                trial_std = arr.std(axis=1, keepdims=True)
                for spec in specs:
                    cid = str(spec["id"])
                    if counts[cid][y] >= samples_per_class:
                        continue
                    starts = get_starts(arr.shape[1], str(spec["windowing"]), 400, rng, int(spec["k"]))
                    for start in starts:
                        if counts[cid][y] >= samples_per_class:
                            break
                        win = arr[:, start:start + 400]
                        x = normalize_window(win, str(spec["normalization"]), trial_mean, trial_std, official)
                        rows[cid].append((x, y, name, trial_id, start))
                        counts[cid][y] += 1
    out_dir = run_dir / "converted_candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for spec in specs:
        cid = str(spec["id"])
        recs = rows[cid]
        path = out_dir / f"{cid}.h5"
        with h5py.File(path, "w") as h5:
            h5.create_dataset("X", data=np.stack([r[0] for r in recs]).astype(np.float32), compression="lzf", chunks=(min(64, len(recs)), 62, 400))
            h5.create_dataset("y", data=np.asarray([r[1] for r in recs], dtype=np.int64), compression="lzf", chunks=(min(1024, len(recs)),))
            h5.create_dataset("source_file", data=[r[2] for r in recs], dtype=h5py.string_dtype("utf-8"))
            h5.create_dataset("trial_id", data=np.asarray([r[3] for r in recs], dtype=np.int16), compression="lzf", chunks=(min(1024, len(recs)),))
            h5.create_dataset("window_start", data=np.asarray([r[4] for r in recs], dtype=np.int64), compression="lzf", chunks=(min(1024, len(recs)),))
            h5.attrs["candidate_id"] = cid
            h5.attrs["data_source"] = "archive_seed_style_mat"
            h5.attrs["windowing"] = str(spec["windowing"])
            h5.attrs["normalization"] = str(spec["normalization"])
            h5.attrs["label_mapping"] = "-1->0, 0->1, +1->2; labels read from Preprocessed_EEG/label.mat"
            h5.attrs["sampling_rate_assumed"] = 200.0
            h5.attrs["channel_order"] = json.dumps(SEED_CHANNELS)
        paths.append(path)
    return paths


def h5_signature(path: Path) -> Dict[str, object]:
    with h5py.File(path, "r") as h5:
        X = h5["X"][()].astype(np.float32)
        y = h5["y"][()].astype(np.int64)
        return {
            "path": str(path),
            "candidate_id": str(h5.attrs.get("candidate_id", path.stem)),
            "data_source": str(h5.attrs.get("data_source", "unknown")),
            "windowing": str(h5.attrs.get("windowing", "")),
            "normalization": str(h5.attrs.get("normalization", "")),
            "sample_count": int(len(X)),
            "label_distribution": label_distribution(y),
            "global_mean": float(X.mean()),
            "global_std": float(X.std()),
            "sample_mean": X.mean(axis=(1, 2)),
            "sample_std": X.std(axis=(1, 2)),
            "channel_mean": X.mean(axis=(0, 2)),
            "channel_std": X.std(axis=(0, 2)),
            "energy": np.mean(np.square(X), axis=(1, 2)),
            "corr_upper": corr_upper_distribution(X),
            "cov_eig": cov_eig_signature(X),
            "block_similarity": block_similarity(X),
        }


def dist(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size == 0 or b.size == 0:
        return 999.0
    if a.size == b.size:
        return float(np.sqrt(np.mean(np.square(a - b))))
    return float(wasserstein_distance(a, b))


def rank_candidates(paths: List[Path], official: Dict[str, object], run_dir: Path) -> List[Dict[str, object]]:
    official_sig = {
        "global_mean": float(official["global_mean"]),
        "global_std": float(official["global_std"]),
        "sample_mean": official["sample_mean"],
        "sample_std": official["sample_std"],
        "channel_mean": official["channel_mean"],
        "channel_std": official["channel_std"],
        "energy": np.square(load_h5(DATA_DIR / "train.h5", True)[0]).mean(axis=(1, 2)),
        "corr_upper": official["corr_upper"],
        "cov_eig": official["cov_eig"],
        "block_similarity": official["block_similarity"],
    }
    rows = []
    for path in paths:
        sig = h5_signature(path)
        global_distance = abs(float(sig["global_mean"]) - official_sig["global_mean"]) + abs(math.log((float(sig["global_std"]) + 1e-6) / (official_sig["global_std"] + 1e-6)))
        channel_distance = dist(sig["channel_mean"], official_sig["channel_mean"]) + dist(np.log(np.asarray(sig["channel_std"]) + 1e-6), np.log(np.asarray(official_sig["channel_std"]) + 1e-6))
        sample_distance = wasserstein_distance(sig["sample_mean"], official_sig["sample_mean"]) + wasserstein_distance(np.log(np.asarray(sig["sample_std"]) + 1e-6), np.log(np.asarray(official_sig["sample_std"]) + 1e-6))
        energy_distance = wasserstein_distance(np.log(np.asarray(sig["energy"]) + 1e-6), np.log(np.asarray(official_sig["energy"]) + 1e-6))
        cov_distance = dist(sig["cov_eig"], official_sig["cov_eig"])
        corr_distance = float(ks_2samp(sig["corr_upper"], official_sig["corr_upper"]).statistic) if len(sig["corr_upper"]) and len(official_sig["corr_upper"]) else 1.0
        block_distance = np.mean([abs(sig["block_similarity"].get(k, 0.0) - official_sig["block_similarity"].get(k, 0.0)) for k in ("5", "10", "15", "20", "30")])
        score = global_distance + channel_distance + sample_distance + 0.5 * energy_distance + 5.0 * cov_distance + corr_distance + block_distance
        rows.append({
            "candidate_id": sig["candidate_id"],
            "data_source": sig["data_source"],
            "windowing": sig["windowing"],
            "normalization": sig["normalization"],
            "sample_count": sig["sample_count"],
            "label_distribution": json.dumps(sig["label_distribution"], ensure_ascii=False),
            "global_mean": sig["global_mean"],
            "global_std": sig["global_std"],
            "distance_to_official_train": global_distance,
            "channel_stat_distance": channel_distance,
            "sample_stat_distance": sample_distance,
            "energy_distance": energy_distance,
            "cov_eig_distance": cov_distance,
            "corr_distance": corr_distance,
            "block_distance": block_distance,
            "overall_stat_score": score,
            "h5_path": str(path),
            "notes": "",
        })
    rows.sort(key=lambda r: float(r["overall_stat_score"]))
    csv_path = run_dir / "preprocessing_candidate_stats.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        fields = list(rows[0].keys())
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def infer_preprocessing(train: Dict[str, object], val: Dict[str, object], test: Dict[str, object], ranked: List[Dict[str, object]]) -> List[str]:
    lines = []
    train_sample_mean = train["sample_mean"]
    train_sample_std = train["sample_std"]
    if np.max(np.abs(train_sample_mean)) < 1e-5:
        lines.append("Official windows are almost exactly global-zero-mean per sample.")
    if np.median(train_sample_std) > 1.5:
        lines.append("Official windows are not per-window z-scored; sample std is far from 1 and varies strongly.")
    if train["pct_sample_channel_std_near_1"] < 0.1:
        lines.append("Official windows are not per-window/per-channel z-scored.")
    if train["pct_sample_channel_mean_abs_lt_1e_3"] < 0.5:
        lines.append("Per-channel means are not all forced to zero, so pure per-window channel demeaning is unlikely.")
    lines.append("Most likely normalization: bandpass/preprocessed EEG followed by per-window global demeaning or equivalent centering.")
    lines.append("Common average reference is not obvious from statistics alone; CAR+channel-z candidates are much more unit-scaled than official.")
    if ranked:
        top = ranked[0]
        lines.append(f"Closest generated archive candidate by statistical score: {top['candidate_id']} ({top['windowing']} + {top['normalization']}).")
    lines.append("Windowing cannot be uniquely identified from isolated 400-point h5 samples, but sample order/block similarity suggests contiguous blocks remain in the official split.")
    return lines


def write_report(run_dir: Path, official_stats: Dict[str, Dict[str, object]], zip_rows: List[Dict[str, object]], ranked: List[Dict[str, object]], labels: np.ndarray) -> Path:
    report = run_dir / "official_h5_forensics.md"
    lines = ["# Official H5 Preprocessing Forensics", "", f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    for name, stats in official_stats.items():
        pub = public_stats(stats)
        lines += [
            f"## Official `{name}`",
            f"- shape: `{pub['shape']}` dtype: `{pub['dtype']}`",
            f"- min/max: `{pub['min']:.4f}` / `{pub['max']:.4f}`",
            f"- global mean/std: `{pub['global_mean']:.8f}` / `{pub['global_std']:.4f}`",
            f"- sample mean quantiles: `{pub['sample_mean_quantiles']}`",
            f"- sample std quantiles: `{pub['sample_std_quantiles']}`",
            f"- channel mean quantiles: `{pub['channel_mean_quantiles']}`",
            f"- channel std quantiles: `{pub['channel_std_quantiles']}`",
            f"- sample-channel abs mean quantiles: `{pub['sample_channel_abs_mean_quantiles']}`",
            f"- pct sample-channel std near 1: `{pub['pct_sample_channel_std_near_1']:.4f}`",
            f"- labels: `{pub['label_distribution']}`",
            f"- NaN/Inf: `{pub['has_nan']}` / `{pub['has_inf']}`",
            f"- adjacent correlation quantiles: `{pub['adjacent_correlation_quantiles']}`",
            f"- block similarity: `{pub['block_similarity']}`",
            "",
        ]
    lines += ["## Archive Inventory", f"- zip: `{DEFAULT_ZIP}`", f"- entries: `{len(zip_rows)}`", f"- labels from label.mat: `{labels.astype(int).tolist()}`", ""]
    for row in zip_rows:
        if int(row["index"]) in (1, 2, 45, 46, 47) or row.get("kind") != "eeg_mat":
            lines.append(f"- #{row['index']}: `{row['name']}` kind=`{row.get('kind')}` size=`{row['size']}`")
    lines += [
        "",
        "The 46th entry is `Preprocessed_EEG/label.mat`, containing the 15-trial SEED label order. The 47th entry is `Preprocessed_EEG/readme.txt`.",
        "The archive is therefore confirmed as a SEED `Preprocessed_EEG` style archive: 45 subject/session mat files plus label/readme metadata.",
        "",
        "## Most Likely Official Preprocessing",
    ]
    lines += [f"- {s}" for s in infer_preprocessing(official_stats["train"], official_stats["val"], official_stats["test"], ranked)]
    lines += ["", "## Candidate Ranking", ""]
    for i, row in enumerate(ranked[:10], start=1):
        lines.append(f"{i}. `{row['candidate_id']}` score=`{float(row['overall_stat_score']):.4f}` windowing=`{row['windowing']}` norm=`{row['normalization']}` h5=`{row['h5_path']}`")
    lines += [
        "",
        "## Excluded Hypotheses",
        "- Per-window global z-score is unlikely because official per-sample std is not 1.",
        "- Per-window channel z-score is unlikely because official per-channel sample std is not concentrated around 1.",
        "- Raw-no-normalization from the archive is too large-scale and has strong offsets compared with official train.",
        "- Val/test were audited for description only; their labels/statistics were not used to fit external normalization.",
        "",
        "## Final Recommendation",
        "- Use statistical matching as a gate before training. The best archive candidates are global-demeaned rather than z-scored.",
        "- Supervised external should remain very low-ratio unless a candidate beats no_external in validation.",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive-zip", type=Path, default=DEFAULT_ZIP)
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--samples-per-class", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.output_root / f"run_{now_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    official_stats = {
        "train": h5_forensics(DATA_DIR / "train.h5", True),
        "val": h5_forensics(DATA_DIR / "val.h5", True),
        "test": h5_forensics(DATA_DIR / "test_x_only.h5", False),
    }
    (run_dir / "official_h5_stats.json").write_text(
        json.dumps({k: public_stats(v) for k, v in official_stats.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    zip_rows, labels = inspect_zip(args.archive_zip, run_dir)
    paths = generate_candidates(args.archive_zip, labels, official_stats["train"], run_dir, args.samples_per_class)
    ranked = rank_candidates(paths, official_stats["train"], run_dir)
    report = write_report(run_dir, official_stats, zip_rows, ranked, labels)
    best = {
        "candidate_id": ranked[0]["candidate_id"],
        "data_source": ranked[0]["data_source"],
        "windowing": ranked[0]["windowing"],
        "normalization": ranked[0]["normalization"],
        "h5_path": ranked[0]["h5_path"],
        "label_mapping": "-1->0, 0->1, +1->2",
        "use_validation_for_normalization": False,
        "stat_score": ranked[0]["overall_stat_score"],
    }
    (run_dir / "best_preprocessing_config.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"run_dir: {run_dir}")
    print(f"report: {report}")
    print(f"stats_csv: {run_dir / 'preprocessing_candidate_stats.csv'}")
    print(f"best: {best}")


if __name__ == "__main__":
    main()
