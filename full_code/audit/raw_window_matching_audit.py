#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Raw SEED window provenance matching for the course H5 files.

This is a standalone controller. It audits the official H5 files and the
SEED Preprocessed_EEG archive, tries to place sampled 62x400 official windows
back into raw .mat trials, then falls back to statistical reconstruction when
exact provenance is not supported by the data.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from scipy.io import loadmat, whosmat
from scipy.signal import detrend
from scipy.stats import ks_2samp, wasserstein_distance


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "course_project" / "SEED"
DEFAULT_ZIP = Path(r"C:\Users\Archery\Downloads\archive (1).zip")
OUTPUT_ROOT = ROOT / "outputs_raw_window_matching"
WINDOW = 400
STANDARD_LABELS = np.asarray([1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1], dtype=np.int64)
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
BLOCK_SIZES = (5, 10, 15, 20, 30)
EEG_KEY_SUFFIX = "eeg"
PYTHON_EXE = Path(sys.executable)
OLD_SUPPLEMENT = ROOT / "outputs_external" / "supplement_seed_like.h5"


SAMPLE_FIELDS = [
    "split", "h5_index", "h5_label_or_unknown", "best_mat_file", "best_trial",
    "best_window_start", "best_preprocess", "best_corr", "best_channel_corr",
    "best_mse", "second_best_corr", "confidence", "match_quality",
]


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def mkdirs(run_dir: Path) -> Dict[str, Path]:
    dirs = {
        "audit": run_dir / "audit",
        "sample_matching": run_dir / "sample_matching",
        "full_matching": run_dir / "full_matching",
        "preprocessing_search": run_dir / "preprocessing_search",
        "reconstructed_external": run_dir / "reconstructed_external",
        "training_check": run_dir / "training_check",
        "reports": run_dir / "reports",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def write_json(path: Path, data: Dict[str, object]) -> None:
    path.write_text(json.dumps(jsonable(data), ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, object]], fields: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        keys = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fields = keys
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def quantiles(x: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {}
    qs = [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    return {str(q): float(np.quantile(arr, q)) for q in qs}


def label_distribution(y: Optional[np.ndarray]) -> Dict[str, int]:
    if y is None:
        return {}
    vals, counts = np.unique(y.astype(int), return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(vals, counts)}


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
    vals = []
    for i in range(max(0, n)):
        a = X[i].reshape(-1)
        b = X[i + 1].reshape(-1)
        vals.append(float(np.corrcoef(a, b)[0, 1]))
    return quantiles(np.asarray(vals)) if vals else {}


def block_similarity(X: np.ndarray, block_sizes: Iterable[int] = BLOCK_SIZES) -> Dict[str, float]:
    flat = X.reshape(X.shape[0], -1)
    out = {}
    for b in block_sizes:
        vals = []
        for start in range(0, len(flat) - b + 1, b):
            c = np.corrcoef(flat[start:start + b])
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
        "pct_sample_abs_mean_lt_1e_3": float(np.mean(np.abs(sample_mean) < 1e-3)),
        "pct_sample_channel_mean_abs_lt_1e_3": float(np.mean(np.abs(sample_channel_mean) < 1e-3)),
        "pct_sample_channel_std_near_1": float(np.mean((sample_channel_std > 0.9) & (sample_channel_std < 1.1))),
        "label_distribution": label_distribution(y),
        "has_nan": bool(np.isnan(X).any()),
        "has_inf": bool(np.isinf(X).any()),
        "adjacent_correlation_quantiles": adjacent_correlation(X),
        "block_similarity": block_similarity(X),
        "sample_mean": sample_mean,
        "sample_std": sample_std,
        "channel_mean": channel_mean,
        "channel_std": channel_std,
        "corr_upper": corr_upper_distribution(X),
        "cov_eig": cov_eig_signature(X),
    }


def public_h5_stats(stats: Dict[str, object]) -> Dict[str, object]:
    hidden = {"sample_mean", "sample_std", "channel_mean", "channel_std", "corr_upper", "cov_eig"}
    return {k: v for k, v in stats.items() if k not in hidden}


def trial_number(key: str) -> Optional[int]:
    low = key.lower()
    if "eeg" not in low:
        return None
    tail = low.split("eeg")[-1].strip("_")
    if tail.isdigit():
        return int(tail)
    return None


def ensure_62_by_t(arr: np.ndarray) -> Optional[np.ndarray]:
    x = np.asarray(arr)
    if x.ndim != 2:
        return None
    if x.shape[0] == 62:
        return x.astype(np.float32, copy=False)
    if x.shape[1] == 62:
        return x.T.astype(np.float32, copy=False)
    return None


def read_archive_labels(zf: zipfile.ZipFile) -> np.ndarray:
    labels = STANDARD_LABELS.copy()
    names = [n for n in zf.namelist() if n.lower().endswith("label.mat")]
    if names:
        mat = loadmat(io.BytesIO(zf.read(names[0])))
        for key, val in mat.items():
            if not key.startswith("__"):
                arr = np.asarray(val).reshape(-1)
                if arr.size >= 15:
                    labels = arr[:15].astype(np.int64)
                    break
    return labels


def iter_mat_names(zf: zipfile.ZipFile) -> List[str]:
    return sorted(n for n in zf.namelist() if n.lower().endswith(".mat") and not n.lower().endswith("label.mat"))


def iter_trials(mat: Dict[str, np.ndarray], labels_orig: np.ndarray) -> Iterator[Tuple[int, str, np.ndarray, int]]:
    found = []
    for key, value in mat.items():
        t = trial_number(key)
        if t is None or not (1 <= t <= len(labels_orig)):
            continue
        arr = ensure_62_by_t(value)
        if arr is not None and arr.shape[1] >= WINDOW:
            found.append((t, key, arr, LABEL_MAP[int(labels_orig[t - 1])]))
    for item in sorted(found, key=lambda x: x[0]):
        yield item


def audit_official_h5(dirs: Dict[str, Path]) -> Dict[str, Dict[str, object]]:
    out = {}
    for split, require_y in [("train", True), ("val", True), ("test", False)]:
        path = DATA_DIR / ("test_x_only.h5" if split == "test" else f"{split}.h5")
        out[split] = h5_forensics(path, require_y=require_y)
    lines = ["# Official H5 Audit", "", f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    for split in ["train", "val", "test"]:
        stats = public_h5_stats(out[split])
        lines += [f"## {split}", "```json", json.dumps(jsonable(stats), ensure_ascii=False, indent=2), "```", ""]
    (dirs["audit"] / "official_h5_audit.md").write_text("\n".join(lines), encoding="utf-8")
    return out


def audit_archive(zip_path: Path, dirs: Dict[str, Path]) -> Tuple[List[Dict[str, object]], np.ndarray]:
    rows: List[Dict[str, object]] = []
    trial_rows: List[Dict[str, object]] = []
    readme_text = ""
    labels = STANDARD_LABELS.copy()
    with zipfile.ZipFile(zip_path, "r") as zf:
        labels = read_archive_labels(zf)
        names = zf.namelist()
        for i, name in enumerate(names, start=1):
            info = zf.getinfo(name)
            row: Dict[str, object] = {"index": i, "name": name, "size": int(info.file_size)}
            low = name.lower()
            if low.endswith("readme.txt"):
                try:
                    readme_text = zf.read(name).decode("utf-8", errors="replace")
                except Exception as exc:
                    readme_text = f"Could not read readme: {exc}"
            if low.endswith(".mat") and not low.endswith("label.mat"):
                try:
                    entries = whosmat(io.BytesIO(zf.read(name)))
                    eeg_entries = [(k, shape) for k, shape, dtype in entries if trial_number(k) is not None and 62 in shape]
                    lengths = [max(shape) for k, shape in eeg_entries]
                    row.update({
                        "kind": "eeg_mat",
                        "key_count": len(entries),
                        "eeg_trial_count": len(eeg_entries),
                        "trial_length_min": int(min(lengths)) if lengths else "",
                        "trial_length_max": int(max(lengths)) if lengths else "",
                        "trial_keys": ";".join(k for k, shape in eeg_entries),
                    })
                except Exception as exc:
                    row.update({"kind": "eeg_mat", "error": str(exc)})
            elif low.endswith("label.mat"):
                row["kind"] = "label_mat"
            elif low.endswith("readme.txt"):
                row["kind"] = "readme"
            else:
                row["kind"] = "other"
            rows.append(row)

        for name in iter_mat_names(zf):
            mat = loadmat(io.BytesIO(zf.read(name)))
            for trial_id, key, arr, y in iter_trials(mat, labels):
                trial_rows.append({
                    "mat_file": name,
                    "trial": trial_id,
                    "key": key,
                    "shape": f"{arr.shape[0]}x{arr.shape[1]}",
                    "length": int(arr.shape[1]),
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    "has_nan": bool(np.isnan(arr).any()),
                    "has_inf": bool(np.isinf(arr).any()),
                    "label_original": int(labels[trial_id - 1]),
                    "label_mapped": int(y),
                })

    write_csv(dirs["audit"] / "archive_file_inventory.csv", rows)
    write_csv(dirs["audit"] / "archive_trial_audit.csv", trial_rows)
    eeg_mats = [r["name"] for r in rows if r.get("kind") == "eeg_mat"]
    lengths = [int(r["length"]) for r in trial_rows]
    lines = [
        "# Archive MAT Audit",
        "",
        f"Archive: `{zip_path}`",
        f"File count: {len(rows)}",
        f"EEG mat count: {len(eeg_mats)}",
        f"Label.mat content: {labels.astype(int).tolist()}",
        f"Mapped labels: {[LABEL_MAP[int(v)] for v in labels.tolist()]}",
        "",
        "## Trial Length Distribution",
        "```json",
        json.dumps(quantiles(np.asarray(lengths)), indent=2),
        "```",
        "",
        "## README",
        "```text",
        readme_text[:8000],
        "```",
        "",
        "## Trial Statistics",
        f"Trials audited: {len(trial_rows)}",
        f"Mean quantiles: {json.dumps(quantiles(np.asarray([r['mean'] for r in trial_rows])), indent=2)}",
        f"Std quantiles: {json.dumps(quantiles(np.asarray([r['std'] for r in trial_rows])), indent=2)}",
    ]
    (dirs["audit"] / "archive_mat_audit.md").write_text("\n".join(lines), encoding="utf-8")
    return rows, labels


@dataclass
class PreprocessSpec:
    name: str
    sign_flip: bool = False
    detrend_trial: bool = False
    start_mode: str = "all"


def stage1_preprocess_specs() -> List[PreprocessSpec]:
    # Coarse screening is done by invariance groups. The exact refinement still
    # expands these groups back to P0-P11 and reports the actual best preprocess.
    return [
        PreprocessSpec(name="G_shift"),
        PreprocessSpec(name="G_channel"),
        PreprocessSpec(name="G_car"),
        PreprocessSpec(name="G_car_channel"),
    ]


def round2_preprocess_specs(top_names: Sequence[str]) -> List[PreprocessSpec]:
    base = list(top_names)[:4] or ["P1", "P10", "P3", "P4"]
    specs = []
    for name in base:
        specs.append(PreprocessSpec(name=name, sign_flip=True))
        specs.append(PreprocessSpec(name=f"{name}_detrend", detrend_trial=True))
        specs.append(PreprocessSpec(name=f"{name}_start400", start_mode="multiples400"))
        specs.append(PreprocessSpec(name=f"{name}_start200", start_mode="multiples200"))
        specs.append(PreprocessSpec(name=f"{name}_fixed20", start_mode="fixed20"))
        specs.append(PreprocessSpec(name=f"{name}_middle", start_mode="middle"))
    return specs


def base_preprocess_name(name: str) -> str:
    if name == "G_shift":
        return "P0"
    if name == "G_channel":
        return "P3"
    if name == "G_car":
        return "P5"
    if name == "G_car_channel":
        return "P7"
    if name.startswith("P") and name[1:].isdigit():
        return name
    return name.split("_")[0]


def expand_exact_preprocess_names(name: str) -> List[str]:
    if name == "G_shift":
        return ["P0", "P1", "P2", "P8", "P10", "P11"]
    if name == "G_channel":
        return ["P3", "P4", "P9"]
    if name == "G_car":
        return ["P5", "P6"]
    if name == "G_car_channel":
        return ["P7"]
    if name.endswith("_sign_flip"):
        base = name.replace("_sign_flip", "")
        return [f"{n}_sign_flip" for n in expand_exact_preprocess_names(base)]
    return [name]


def preprocess_window(win: np.ndarray, spec_name: str, trial: np.ndarray, official_train: Dict[str, object]) -> np.ndarray:
    base = base_preprocess_name(spec_name)
    x = win.astype(np.float32, copy=True)
    if base == "P0":
        out = x
    elif base == "P1":
        out = x - x.mean()
    elif base == "P2":
        out = (x - x.mean()) / max(float(x.std()), 1e-6)
    elif base == "P3":
        out = x - x.mean(axis=1, keepdims=True)
    elif base == "P4":
        out = (x - x.mean(axis=1, keepdims=True)) / np.maximum(x.std(axis=1, keepdims=True), 1e-6)
    elif base == "P5":
        out = x - x.mean(axis=0, keepdims=True)
    elif base == "P6":
        car = x - x.mean(axis=0, keepdims=True)
        out = car - car.mean()
    elif base == "P7":
        car = x - x.mean(axis=0, keepdims=True)
        out = car - car.mean(axis=1, keepdims=True)
    elif base == "P8":
        out = trial[:, win.start_marker:win.start_marker + WINDOW] if hasattr(win, "start_marker") else x
        out = out - trial.mean()
    elif base == "P9":
        out = x - trial.mean(axis=1, keepdims=True)
    elif base == "P10":
        target_mean = float(official_train["global_mean"])
        target_std = float(official_train["global_std"])
        out = ((x - x.mean()) / max(float(x.std()), 1e-6)) * target_std + target_mean
    elif base == "P11":
        target_mean = float(official_train["global_mean"])
        out = (x - x.mean()) + target_mean
    else:
        raise ValueError(f"Unknown preprocess {spec_name}")
    if "detrend" in spec_name:
        out = detrend(out, axis=1, type="linear").astype(np.float32)
    return out.astype(np.float32, copy=False)


def preprocess_batch(wins: np.ndarray, spec_name: str, trial: np.ndarray, starts: np.ndarray, official_train: Dict[str, object]) -> np.ndarray:
    base = base_preprocess_name(spec_name)
    X = wins.astype(np.float32, copy=True)
    if base == "P0":
        out = X
    elif base == "P1":
        out = X - X.mean(axis=(1, 2), keepdims=True)
    elif base == "P2":
        out = (X - X.mean(axis=(1, 2), keepdims=True)) / np.maximum(X.std(axis=(1, 2), keepdims=True), 1e-6)
    elif base == "P3":
        out = X - X.mean(axis=2, keepdims=True)
    elif base == "P4":
        out = (X - X.mean(axis=2, keepdims=True)) / np.maximum(X.std(axis=2, keepdims=True), 1e-6)
    elif base == "P5":
        out = X - X.mean(axis=1, keepdims=True)
    elif base == "P6":
        car = X - X.mean(axis=1, keepdims=True)
        out = car - car.mean(axis=(1, 2), keepdims=True)
    elif base == "P7":
        car = X - X.mean(axis=1, keepdims=True)
        out = car - car.mean(axis=2, keepdims=True)
    elif base == "P8":
        out = X - trial.mean()
    elif base == "P9":
        out = X - trial.mean(axis=1, keepdims=True)[None, :, :]
    elif base == "P10":
        target_mean = float(official_train["global_mean"])
        target_std = float(official_train["global_std"])
        out = ((X - X.mean(axis=(1, 2), keepdims=True)) / np.maximum(X.std(axis=(1, 2), keepdims=True), 1e-6)) * target_std + target_mean
    elif base == "P11":
        target_mean = float(official_train["global_mean"])
        out = (X - X.mean(axis=(1, 2), keepdims=True)) + target_mean
    else:
        raise ValueError(f"Unknown preprocess {spec_name}")
    if "detrend" in spec_name:
        out = detrend(out, axis=2, type="linear").astype(np.float32)
    return out.astype(np.float32, copy=False)


def starts_for_mode(length: int, mode: str, stride: int = 50) -> np.ndarray:
    max_start = length - WINDOW
    if max_start < 0:
        return np.asarray([], dtype=np.int64)
    if mode == "multiples400":
        vals = np.arange(0, max_start + 1, 400)
    elif mode == "multiples200":
        vals = np.arange(0, max_start + 1, 200)
    elif mode == "fixed20":
        vals = np.linspace(0, max_start, num=min(20, max_start + 1), dtype=np.int64)
    elif mode == "middle":
        all_starts = np.arange(0, max_start + 1, 400)
        mid = length / 2.0
        order = np.argsort(np.abs((all_starts + WINDOW / 2.0) - mid))
        vals = np.sort(all_starts[order[:30]])
    else:
        vals = np.arange(0, max_start + 1, stride)
    return vals.astype(np.int64)


def stack_windows(trial: np.ndarray, starts: np.ndarray) -> np.ndarray:
    return np.stack([trial[:, int(s):int(s) + WINDOW] for s in starts]).astype(np.float32)


def fingerprint_batch(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32, copy=False)
    z = X - X.mean(axis=(1, 2), keepdims=True)
    z = z / np.maximum(z.std(axis=(1, 2), keepdims=True), 1e-6)
    # 10 representative channels x 10 time bins, plus channel std and time profile.
    ch_idx = np.linspace(0, 61, num=10, dtype=int)
    time_binned = z[:, ch_idx, :].reshape(z.shape[0], 10, 10, 40).mean(axis=3).reshape(z.shape[0], -1)
    ch_std = z.std(axis=2)
    t_profile = z.reshape(z.shape[0], 62, 10, 40).mean(axis=(1, 3))
    f = np.concatenate([time_binned, ch_std, t_profile], axis=1)
    f = f - f.mean(axis=1, keepdims=True)
    f = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-8)
    return f.astype(np.float32, copy=False)


def normalized_flat(X: np.ndarray) -> np.ndarray:
    flat = X.reshape(X.shape[0], -1).astype(np.float32, copy=False)
    flat = flat - flat.mean(axis=1, keepdims=True)
    flat = flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-8)
    return flat


def per_channel_corr(a: np.ndarray, b: np.ndarray) -> float:
    vals = []
    for c in range(a.shape[0]):
        aa = a[c] - a[c].mean()
        bb = b[c] - b[c].mean()
        denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
        vals.append(float(np.dot(aa, bb) / denom) if denom > 1e-8 else 0.0)
    return float(np.mean(vals))


def match_quality(corr: float) -> str:
    if corr >= 0.95:
        return "high_exact"
    if corr >= 0.85:
        return "possible"
    if corr >= 0.75:
        return "weak"
    return "failed"


def select_sample_items(rng: np.random.Generator) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    for split, per_class in [("train", 30), ("val", 10)]:
        X, y = load_h5(DATA_DIR / f"{split}.h5", True)
        for cls in sorted(np.unique(y).astype(int)):
            idx = np.where(y == cls)[0]
            take = rng.choice(idx, size=min(per_class, len(idx)), replace=False)
            for i in sorted(take.tolist()):
                items.append({"split": split, "index": int(i), "label": int(y[i]), "X": X[i]})
    Xtest, _ = load_h5(DATA_DIR / "test_x_only.h5", False)
    take = rng.choice(np.arange(len(Xtest)), size=min(30, len(Xtest)), replace=False)
    for i in sorted(take.tolist()):
        items.append({"split": "test", "index": int(i), "label": None, "X": Xtest[i]})
    return items[:150]


def update_top(top: List[List[Tuple[float, str, int, int, str]]], sample_ids: np.ndarray, scores: np.ndarray,
               mat_name: str, trial_id: int, starts: np.ndarray, spec_name: str, limit: int) -> None:
    # top[sample] stores (fingerprint_score, mat, trial, start, preprocess).
    for local_col, sample_id in enumerate(sample_ids):
        col = scores[:, local_col]
        if col.size == 0:
            continue
        k = min(limit, col.size)
        idx = np.argpartition(col, -k)[-k:]
        bucket = top[int(sample_id)]
        for j in idx:
            bucket.append((float(col[j]), mat_name, int(trial_id), int(starts[j]), spec_name))
        if len(bucket) > limit * 4:
            bucket.sort(key=lambda x: x[0], reverse=True)
            del bucket[limit:]


def exact_refine_one(sample_x: np.ndarray, candidate: Tuple[float, str, int, int, str], zf: zipfile.ZipFile,
                     labels: np.ndarray, official_train: Dict[str, object], radius: int = 200, step: int = 1) -> List[Dict[str, object]]:
    _, mat_name, trial_id, coarse_start, spec_name = candidate
    mat = loadmat(io.BytesIO(zf.read(mat_name)))
    trial = None
    for t, key, arr, _y in iter_trials(mat, labels):
        if t == trial_id:
            trial = arr
            break
    if trial is None:
        return []
    max_start = trial.shape[1] - WINDOW
    lo = max(0, int(coarse_start) - radius)
    hi = min(max_start, int(coarse_start) + radius)
    starts = np.arange(lo, hi + 1, step, dtype=np.int64)
    if starts.size == 0:
        return []
    out = []
    wins = stack_windows(trial, starts)
    sample_norm = normalized_flat(sample_x[None, :, :])[0]
    for exact_name in expand_exact_preprocess_names(spec_name):
        flip = exact_name.endswith("_sign_flip")
        clean_name = exact_name.replace("_sign_flip", "")
        Xp = preprocess_batch(wins, clean_name, trial, starts, official_train)
        if flip:
            Xp = -Xp
        cand_norm = normalized_flat(Xp)
        corrs = cand_norm @ sample_norm
        best_indices = np.argpartition(corrs, -min(2, len(corrs)))[-min(2, len(corrs)):]
        for j in best_indices:
            x = Xp[int(j)]
            corr = float(corrs[int(j)])
            out.append({
                "mat": mat_name,
                "trial": int(trial_id),
                "start": int(starts[int(j)]),
                "preprocess": exact_name,
                "corr": corr,
                "channel_corr": per_channel_corr(sample_x, x),
                "mse": float(np.mean(np.square(sample_x - x))),
            })
    return out


def sample_matching(zip_path: Path, labels: np.ndarray, official_train: Dict[str, object], dirs: Dict[str, Path],
                    round_name: str = "", specs: Optional[List[PreprocessSpec]] = None) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    rng = np.random.default_rng(42 if not round_name else 4242)
    items = select_sample_items(rng)
    specs = specs or stage1_preprocess_specs()
    sample_X = np.stack([it["X"] for it in items]).astype(np.float32)
    sample_fp = fingerprint_batch(sample_X)
    sample_by_label: Dict[object, np.ndarray] = {}
    for label in [0, 1, 2, None]:
        ids = [i for i, it in enumerate(items) if it["label"] == label or (label is None and it["split"] == "test")]
        sample_by_label[label] = np.asarray(ids, dtype=np.int64)
    top: List[List[Tuple[float, str, int, int, str]]] = [[] for _ in items]
    with zipfile.ZipFile(zip_path, "r") as zf:
        mat_names = iter_mat_names(zf)
        for mat_i, mat_name in enumerate(mat_names, start=1):
            print(f"[sample{round_name}] scanning {mat_i}/{len(mat_names)} {mat_name}", flush=True)
            mat = loadmat(io.BytesIO(zf.read(mat_name)))
            for trial_id, key, arr, y in iter_trials(mat, labels):
                label_ids = np.concatenate([sample_by_label.get(y, np.asarray([], dtype=np.int64)), sample_by_label[None]])
                if label_ids.size == 0:
                    continue
                for spec in specs:
                    trial_arr = detrend(arr, axis=1, type="linear").astype(np.float32) if spec.detrend_trial else arr
                    starts = starts_for_mode(trial_arr.shape[1], spec.start_mode, stride=50)
                    if starts.size == 0:
                        continue
                    # Chunk to keep memory bounded.
                    for offset in range(0, len(starts), 512):
                        chunk = starts[offset:offset + 512]
                        wins = stack_windows(trial_arr, chunk)
                        Xp = preprocess_batch(wins, spec.name, trial_arr, chunk, official_train)
                        if spec.sign_flip:
                            Xp = -Xp
                            spec_name = f"{spec.name}_sign_flip"
                        else:
                            spec_name = spec.name
                        fp = fingerprint_batch(Xp)
                        scores = fp @ sample_fp[label_ids].T
                        update_top(top, label_ids, scores, mat_name, trial_id, chunk, spec_name, 4)
    rows: List[Dict[str, object]] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for sample_id, item in enumerate(items):
            candidates = sorted(top[sample_id], key=lambda x: x[0], reverse=True)[:4]
            exacts = []
            seen = set()
            for cand in candidates:
                key = (cand[1], cand[2], cand[3], cand[4])
                if key in seen:
                    continue
                seen.add(key)
                exacts.extend(exact_refine_one(item["X"], cand, zf, labels, official_train))
            exacts.sort(key=lambda r: r["corr"], reverse=True)
            best = exacts[0] if exacts else {"mat": "", "trial": "", "start": "", "preprocess": "", "corr": -1.0, "channel_corr": "", "mse": ""}
            second = exacts[1]["corr"] if len(exacts) > 1 else -1.0
            conf = float(best["corr"] - second) if isinstance(best["corr"], float) else ""
            rows.append({
                "split": item["split"],
                "h5_index": item["index"],
                "h5_label_or_unknown": "unknown" if item["label"] is None else item["label"],
                "best_mat_file": best["mat"],
                "best_trial": best["trial"],
                "best_window_start": best["start"],
                "best_preprocess": best["preprocess"],
                "best_corr": best["corr"],
                "best_channel_corr": best["channel_corr"],
                "best_mse": best["mse"],
                "second_best_corr": second,
                "confidence": conf,
                "match_quality": match_quality(float(best["corr"])),
            })
    suffix = f"_{round_name}" if round_name else ""
    csv_path = dirs["sample_matching"] / f"sample_window_match_results{suffix}.csv"
    write_csv(csv_path, rows, SAMPLE_FIELDS)
    summary = summarize_match_rows(rows)
    report_path = dirs["sample_matching"] / f"matching_feasibility_report{suffix}.md"
    report_path.write_text(make_matching_report(rows, summary, f"Sample matching {round_name or 'round1'}"), encoding="utf-8")
    return rows, summary


def summarize_match_rows(rows: List[Dict[str, object]]) -> Dict[str, object]:
    counts = Counter(str(r["match_quality"]) for r in rows)
    n = max(1, len(rows))
    preprocess_counts = Counter(str(r["best_preprocess"]) for r in rows)
    return {
        "n": len(rows),
        "quality_counts": dict(counts),
        "high_exact_ratio": counts.get("high_exact", 0) / n,
        "possible_or_high_ratio": (counts.get("high_exact", 0) + counts.get("possible", 0)) / n,
        "best_preprocess_counts": dict(preprocess_counts),
        "median_best_corr": float(np.median([float(r["best_corr"]) for r in rows])) if rows else 0.0,
        "max_best_corr": float(np.max([float(r["best_corr"]) for r in rows])) if rows else 0.0,
    }


def make_matching_report(rows: List[Dict[str, object]], summary: Dict[str, object], title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "## Decision Metrics",
        f"- Samples: {summary['n']}",
        f"- high_exact ratio: {summary['high_exact_ratio']:.3f}",
        f"- possible+high ratio: {summary['possible_or_high_ratio']:.3f}",
        f"- median best corr: {summary['median_best_corr']:.4f}",
        f"- max best corr: {summary['max_best_corr']:.4f}",
        "",
        "## Quality Counts",
        "```json",
        json.dumps(summary["quality_counts"], indent=2),
        "```",
        "",
        "## Best Preprocess Counts",
        "```json",
        json.dumps(summary["best_preprocess_counts"], indent=2),
        "```",
        "",
        "## Top 20 Matches",
        "",
    ]
    for row in sorted(rows, key=lambda r: float(r["best_corr"]), reverse=True)[:20]:
        lines.append(
            f"- {row['split']}[{row['h5_index']}] label={row['h5_label_or_unknown']} "
            f"corr={float(row['best_corr']):.4f} {row['match_quality']} "
            f"{row['best_mat_file']} trial={row['best_trial']} start={row['best_window_start']} prep={row['best_preprocess']}"
        )
    return "\n".join(lines)


def choose_stage(summary: Dict[str, object]) -> str:
    if float(summary["high_exact_ratio"]) >= 0.30:
        return "2A"
    if float(summary["possible_or_high_ratio"]) >= 0.50:
        return "2B"
    return "2C"


def run_full_matching_placeholder(sample_rows: List[Dict[str, object]], dirs: Dict[str, Path]) -> Dict[str, object]:
    # The expensive all-sample matcher is only meaningful when stage1 is exact.
    # This conservative implementation records the inferred exact provenance from
    # the high-confidence sample rows and avoids fabricating full provenance.
    high = [r for r in sample_rows if r["match_quality"] == "high_exact"]
    out_path = dirs["full_matching"] / "full_window_match_results.csv"
    write_csv(out_path, high, SAMPLE_FIELDS)
    analysis = {
        "full_matching_executed": False,
        "reason": "Stage1 did not provide enough exact matches for a trustworthy full pass." if not high else "High exact sample matches recorded; use as provenance anchors.",
        "high_exact_rows_written": len(high),
    }
    (dirs["reports"] / "split_source_analysis.md").write_text(
        "# Split Source Analysis\n\n"
        "Full all-sample matching was not expanded unless sample matching crossed the exact gate. "
        "See `full_matching/full_window_match_results.csv` for high-confidence anchors.\n\n"
        f"```json\n{json.dumps(analysis, indent=2)}\n```",
        encoding="utf-8",
    )
    (dirs["reports"] / "inferred_exact_preprocessing_pipeline.md").write_text(
        "# Inferred Exact Preprocessing Pipeline\n\n"
        "No exact h5-to-raw pipeline can be asserted unless the exact-matching gate is passed.",
        encoding="utf-8",
    )
    return analysis


def get_starts(length: int, windowing: str, rng: np.random.Generator, k: int) -> List[int]:
    if length < WINDOW:
        return []
    max_start = length - WINDOW
    if windowing == "first_nonoverlap":
        vals = list(range(0, max_start + 1, WINDOW))[:k]
    elif windowing == "full_nonoverlap_stride400":
        vals = list(range(0, max_start + 1, 400))[:max(k, 1)]
    elif windowing == "full_overlap_stride200":
        vals = list(range(0, max_start + 1, 200))[:max(k, 1)]
    elif windowing == "middle_nonoverlap":
        all_starts = np.asarray(list(range(0, max_start + 1, WINDOW)), dtype=np.int64)
        order = np.argsort(np.abs((all_starts + WINDOW / 2) - length / 2))
        vals = sorted(all_starts[order[:k]].astype(int).tolist())
    elif windowing.startswith("fixed_k_per_trial_k"):
        kk = int(windowing.split("k")[-1])
        vals = np.linspace(0, max_start, num=min(kk, max_start + 1), dtype=int).tolist()
    elif windowing.startswith("random_k_per_trial_k"):
        kk = int(windowing.split("k")[-1])
        vals = rng.choice(max_start + 1, size=min(kk, max_start + 1), replace=False).astype(int).tolist()
    else:
        raise ValueError(windowing)
    return sorted(set(int(v) for v in vals))


def normalize_candidate(win: np.ndarray, norm: str, trial: np.ndarray, official_train: Dict[str, object]) -> np.ndarray:
    x = win.astype(np.float32, copy=True)
    if norm == "raw_no_norm":
        return x
    if norm == "window_center_only":
        return x - x.mean()
    if norm == "per_window_global_zscore":
        return (x - x.mean()) / max(float(x.std()), 1e-6)
    if norm == "per_window_channel_center":
        return x - x.mean(axis=1, keepdims=True)
    if norm == "per_window_channel_zscore":
        return (x - x.mean(axis=1, keepdims=True)) / np.maximum(x.std(axis=1, keepdims=True), 1e-6)
    if norm == "CAR_then_window_center_only":
        car = x - x.mean(axis=0, keepdims=True)
        return car - car.mean()
    if norm == "official_train_global_align":
        target_mean = float(official_train["global_mean"])
        target_std = float(official_train["global_std"])
        return ((x - x.mean()) / max(float(x.std()), 1e-6)) * target_std + target_mean
    if norm == "official_train_center_only":
        target_mean = float(official_train["global_mean"])
        return (x - x.mean()) + target_mean
    if norm == "trial_center_then_window":
        return x - trial.mean()
    raise ValueError(norm)


def candidate_specs() -> List[Dict[str, object]]:
    windowings = [
        "first_nonoverlap",
        "full_nonoverlap_stride400",
        "full_overlap_stride200",
        "middle_nonoverlap",
        "fixed_k_per_trial_k5",
        "fixed_k_per_trial_k10",
        "fixed_k_per_trial_k20",
        "fixed_k_per_trial_k30",
        "random_k_per_trial_k10",
        "random_k_per_trial_k20",
    ]
    norms = [
        "raw_no_norm",
        "window_center_only",
        "per_window_global_zscore",
        "per_window_channel_center",
        "per_window_channel_zscore",
        "CAR_then_window_center_only",
        "official_train_global_align",
        "official_train_center_only",
        "trial_center_then_window",
    ]
    specs = []
    for i, (w, n) in enumerate((w, n) for w in windowings for n in norms):
        # Keep all combinations but cap per-candidate samples to keep disk sane.
        specs.append({"id": f"c{i + 1:02d}_{w}_{n}", "windowing": w, "normalization": n, "k": 30})
    return specs


def generate_candidates(zip_path: Path, labels: np.ndarray, official_train: Dict[str, object], dirs: Dict[str, Path],
                        samples_per_class: int = 30) -> List[Path]:
    specs = candidate_specs()
    rng = np.random.default_rng(42)
    rows = {s["id"]: [] for s in specs}
    counts = {s["id"]: Counter() for s in specs}
    with zipfile.ZipFile(zip_path, "r") as zf:
        mat_names = iter_mat_names(zf)
        for mat_i, mat_name in enumerate(mat_names, start=1):
            if all(all(counts[s["id"]][c] >= samples_per_class for c in (0, 1, 2)) for s in specs):
                break
            print(f"[stage2C] candidate scan {mat_i}/{len(mat_names)} {mat_name}", flush=True)
            mat = loadmat(io.BytesIO(zf.read(mat_name)))
            for trial_id, key, arr, y in iter_trials(mat, labels):
                for spec in specs:
                    cid = str(spec["id"])
                    if counts[cid][y] >= samples_per_class:
                        continue
                    starts = get_starts(arr.shape[1], str(spec["windowing"]), rng, int(spec["k"]))
                    for start in starts:
                        if counts[cid][y] >= samples_per_class:
                            break
                        win = arr[:, start:start + WINDOW]
                        x = normalize_candidate(win, str(spec["normalization"]), arr, official_train)
                        rows[cid].append((x, y, mat_name, trial_id, start))
                        counts[cid][y] += 1
    paths = []
    out_dir = dirs["preprocessing_search"] / "candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        cid = str(spec["id"])
        recs = rows[cid]
        if not recs:
            continue
        path = out_dir / f"{cid}.h5"
        with h5py.File(path, "w") as h5:
            h5.create_dataset("X", data=np.stack([r[0] for r in recs]).astype(np.float32), compression="lzf", chunks=(min(64, len(recs)), 62, 400))
            h5.create_dataset("y", data=np.asarray([r[1] for r in recs], dtype=np.int64), compression="lzf", chunks=(min(1024, len(recs)),))
            h5.create_dataset("source_file", data=[r[2] for r in recs], dtype=h5py.string_dtype("utf-8"))
            h5.create_dataset("trial_id", data=np.asarray([r[3] for r in recs], dtype=np.int16), compression="lzf", chunks=(min(1024, len(recs)),))
            h5.create_dataset("window_start", data=np.asarray([r[4] for r in recs], dtype=np.int64), compression="lzf", chunks=(min(1024, len(recs)),))
            h5.attrs["candidate_id"] = cid
            h5.attrs["data_source"] = "archive_seed_preprocessed_eeg"
            h5.attrs["windowing"] = str(spec["windowing"])
            h5.attrs["normalization"] = str(spec["normalization"])
            h5.attrs["label_mapping"] = "-1->0, 0->1, +1->2 from archive label.mat"
        paths.append(path)
    return paths


def dist(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.size == 0 or bb.size == 0:
        return 999.0
    if aa.size == bb.size:
        return float(np.sqrt(np.mean(np.square(aa - bb))))
    return float(wasserstein_distance(aa, bb))


def h5_signature(path: Path) -> Dict[str, object]:
    with h5py.File(path, "r") as h5:
        X = h5["X"][()].astype(np.float32)
        y = h5["y"][()].astype(np.int64)
        return {
            "path": str(path),
            "candidate_id": str(h5.attrs.get("candidate_id", path.stem)),
            "data_source": str(h5.attrs.get("data_source", "")),
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
            "corr_upper": corr_upper_distribution(X, max_samples=90),
            "cov_eig": cov_eig_signature(X, max_samples=90),
            "block_similarity": block_similarity(X),
        }


def rank_candidates(paths: List[Path], official_train: Dict[str, object], dirs: Dict[str, Path]) -> List[Dict[str, object]]:
    Xtrain, _ = load_h5(DATA_DIR / "train.h5", True)
    official = {
        "global_mean": float(official_train["global_mean"]),
        "global_std": float(official_train["global_std"]),
        "sample_mean": official_train["sample_mean"],
        "sample_std": official_train["sample_std"],
        "channel_mean": official_train["channel_mean"],
        "channel_std": official_train["channel_std"],
        "energy": np.mean(np.square(Xtrain), axis=(1, 2)),
        "corr_upper": official_train["corr_upper"],
        "cov_eig": official_train["cov_eig"],
        "block_similarity": official_train["block_similarity"],
    }
    rows = []
    for path in paths:
        sig = h5_signature(path)
        global_distance = abs(float(sig["global_mean"]) - official["global_mean"]) + abs(math.log((float(sig["global_std"]) + 1e-6) / (official["global_std"] + 1e-6)))
        sample_distance = wasserstein_distance(sig["sample_mean"], official["sample_mean"]) + wasserstein_distance(np.log(np.asarray(sig["sample_std"]) + 1e-6), np.log(np.asarray(official["sample_std"]) + 1e-6))
        channel_distance = dist(sig["channel_mean"], official["channel_mean"]) + dist(np.log(np.asarray(sig["channel_std"]) + 1e-6), np.log(np.asarray(official["channel_std"]) + 1e-6))
        energy_distance = wasserstein_distance(np.log(np.asarray(sig["energy"]) + 1e-6), np.log(np.asarray(official["energy"]) + 1e-6))
        cov_distance = dist(sig["cov_eig"], official["cov_eig"])
        corr_distance = float(ks_2samp(sig["corr_upper"], official["corr_upper"]).statistic) if len(sig["corr_upper"]) and len(official["corr_upper"]) else 1.0
        block_distance = float(np.mean([abs(sig["block_similarity"].get(str(k), 0.0) - official["block_similarity"].get(str(k), 0.0)) for k in BLOCK_SIZES]))
        score = global_distance + sample_distance + channel_distance + 0.5 * energy_distance + 5.0 * cov_distance + corr_distance + block_distance
        rows.append({
            "candidate_id": sig["candidate_id"],
            "data_source": sig["data_source"],
            "windowing": sig["windowing"],
            "normalization": sig["normalization"],
            "sample_count": sig["sample_count"],
            "label_distribution": json.dumps(sig["label_distribution"], ensure_ascii=False),
            "global_mean": sig["global_mean"],
            "global_std": sig["global_std"],
            "global_mean_std_distance": global_distance,
            "per_sample_mean_std_distance": sample_distance,
            "per_channel_mean_std_distance": channel_distance,
            "covariance_eigenvalue_distance": cov_distance,
            "correlation_upper_triangle_distance": corr_distance,
            "energy_distribution_distance": energy_distance,
            "block_similarity_distance": block_distance,
            "overall_stat_score": score,
            "h5_path": str(path),
        })
    rows.sort(key=lambda r: float(r["overall_stat_score"]))
    write_csv(dirs["preprocessing_search"] / "preprocessing_candidate_stats.csv", rows)
    return rows


def copy_top_candidates(ranked: List[Dict[str, object]], dirs: Dict[str, Path], top_k: int = 5) -> List[Dict[str, object]]:
    copied = []
    for i, row in enumerate(ranked[:top_k], start=1):
        src = Path(str(row["h5_path"]))
        dst = dirs["reconstructed_external"] / f"top_candidate_{i}_{row['candidate_id']}.h5"
        shutil.copy2(src, dst)
        new_row = dict(row)
        new_row["h5_path"] = str(dst)
        copied.append(new_row)
    if copied:
        best = {
            "selected_by": "lowest statistical distance to official train",
            "top_candidates": copied,
            "warning": "Fallback candidates are not proven exact provenance matches.",
        }
        write_json(dirs["reconstructed_external"] / "best_preprocessing_config.json", best)
    return copied


TRAIN_FIELDS = [
    "candidate_id", "data_source", "windowing", "normalization", "stat_score_rank",
    "external_ratio", "seed", "best_val_acc", "best_epoch", "final_val_acc", "macro_f1",
    "prediction_distribution", "output_dir", "checkpoint_path", "status", "error_message",
]


def run_cmd(cmd: List[str], log_path: Path) -> Tuple[int, str]:
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    return proc.returncode, proc.stdout


def latest_result(out_root: Path, run_name: str) -> Dict[str, object]:
    dirs = sorted(out_root.glob(f"{run_name}_*"), key=lambda p: p.stat().st_mtime)
    if not dirs:
        raise RuntimeError(f"No output directory for {run_name}")
    return json.loads((dirs[-1] / "run_results.json").read_text(encoding="utf-8"))


def train_one(dirs: Dict[str, Path], item: Dict[str, object], rank: int, ratio: Optional[float], seed: int) -> Dict[str, object]:
    out_root = dirs["training_check"] / "training_runs"
    safe_id = str(item["candidate_id"]).replace(":", "_").replace("\\", "_").replace("/", "_")[:80]
    ratio_tag = "none" if ratio is None else str(ratio).replace(".", "p")
    run_name = f"rawmatch_{safe_id}_r{ratio_tag}_s{seed}"
    cmd = [
        str(PYTHON_EXE), "seed_multiscale_crnn_experiment.py",
        "--model", "multiscale_crnn",
        "--epochs", "8",
        "--patience", "4",
        "--batch-size", "64",
        "--seed", str(seed),
        "--output-root", str(out_root),
        "--run-name", run_name,
    ]
    if ratio is not None:
        cmd += ["--use-supplement", "--supplement-h5", str(item["h5_path"]), "--external-ratio", str(ratio)]
    code, output = run_cmd(cmd, dirs["training_check"] / "logs" / f"{run_name}.log")
    if code != 0:
        return {
            "candidate_id": item["candidate_id"],
            "data_source": item.get("data_source", ""),
            "windowing": item.get("windowing", ""),
            "normalization": item.get("normalization", ""),
            "stat_score_rank": rank,
            "external_ratio": "" if ratio is None else ratio,
            "seed": seed,
            "status": "failed",
            "error_message": f"training command exited {code}",
        }
    result = latest_result(out_root, run_name)
    val = result.get("validation_metrics_at_best") or {}
    return {
        "candidate_id": item["candidate_id"],
        "data_source": item.get("data_source", ""),
        "windowing": item.get("windowing", ""),
        "normalization": item.get("normalization", ""),
        "stat_score_rank": rank,
        "external_ratio": "" if ratio is None else ratio,
        "seed": seed,
        "best_val_acc": result.get("best_val_acc", ""),
        "best_epoch": result.get("best_epoch", ""),
        "final_val_acc": result.get("final_val_acc", ""),
        "macro_f1": result.get("best_val_macro_f1", ""),
        "prediction_distribution": json.dumps(val.get("prediction_distribution", {}), ensure_ascii=False),
        "output_dir": result.get("output_dir", ""),
        "checkpoint_path": result.get("best_acc_checkpoint", ""),
        "status": "ok",
        "error_message": "",
    }


def training_check(candidates: List[Dict[str, object]], dirs: Dict[str, Path]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    control = {"candidate_id": "no_external", "data_source": "official_only", "windowing": "", "normalization": "", "h5_path": ""}
    print("[stage4] training no_external control", flush=True)
    rows.append(train_one(dirs, control, 0, None, 42))
    old_items = []
    if OLD_SUPPLEMENT.exists():
        old_items.append({
            "candidate_id": "old_supplement_seed_like",
            "data_source": "existing_supplement",
            "windowing": "prior",
            "normalization": "prior",
            "h5_path": str(OLD_SUPPLEMENT),
        })
    train_items = old_items + candidates[:5]
    ratios = [0.005, 0.01, 0.02]
    for rank, item in enumerate(train_items, start=1):
        for ratio in ratios:
            print(f"[stage4] training {item['candidate_id']} ratio={ratio}", flush=True)
            rows.append(train_one(dirs, item, rank, ratio, 42))
    ok_control = [r for r in rows if r["candidate_id"] == "no_external" and r.get("status") == "ok"]
    control_acc = float(ok_control[0].get("best_val_acc", 0.0)) if ok_control else 0.0
    winners = [r for r in rows if r.get("status") == "ok" and r["candidate_id"] != "no_external" and float(r.get("best_val_acc", 0.0)) > control_acc]
    # Requirement: if a candidate exceeds no_external, rerun one extra seed.
    for winner in winners[:2]:
        item = next((c for c in train_items if c["candidate_id"] == winner["candidate_id"]), None)
        if item:
            for seed in [3407]:
                print(f"[stage4] rerun winner {item['candidate_id']} seed={seed}", flush=True)
                rows.append(train_one(dirs, item, 99, float(winner["external_ratio"]), seed))
    write_csv(dirs["training_check"] / "reconstructed_external_training_results.csv", rows, TRAIN_FIELDS)
    return rows


def infer_external_decision(training_rows: List[Dict[str, object]]) -> Dict[str, object]:
    ok = [r for r in training_rows if r.get("status") == "ok"]
    control = [r for r in ok if r["candidate_id"] == "no_external"]
    old = [r for r in ok if r["candidate_id"] == "old_supplement_seed_like"]
    rec = [r for r in ok if r["candidate_id"] not in {"no_external", "old_supplement_seed_like"}]
    control_acc = max([float(r.get("best_val_acc", 0.0)) for r in control] or [0.0])
    old_acc = max([float(r.get("best_val_acc", 0.0)) for r in old] or [0.0])
    rec_acc = max([float(r.get("best_val_acc", 0.0)) for r in rec] or [0.0])
    return {
        "no_external_best_val_acc": control_acc,
        "old_supplement_best_val_acc": old_acc,
        "reconstructed_best_val_acc": rec_acc,
        "reconstructed_exceeds_no_external": rec_acc > control_acc,
        "reconstructed_exceeds_old_supplement": rec_acc > old_acc if old else None,
        "recommendation": "continue only with the winning low-ratio reconstructed candidate" if rec_acc > control_acc else "stop supervised external for now; exact provenance was not established and candidates did not beat control",
    }


def make_summary_reports(run_dir: Path, dirs: Dict[str, Path], stage: str, sample_summary: Dict[str, object],
                         round2_summary: Optional[Dict[str, object]], ranked: List[Dict[str, object]],
                         train_decision: Dict[str, object], full_info: Dict[str, object], errors: List[str]) -> Dict[str, object]:
    best_candidate = ranked[0] if ranked else {}
    high = sample_summary.get("high_exact_ratio", 0.0)
    poss = sample_summary.get("possible_or_high_ratio", 0.0)
    failed_ratio = sample_summary.get("quality_counts", {}).get("failed", 0) / max(1, sample_summary.get("n", 1))
    exact_possible = high >= 0.30
    likely_reason = (
        "course H5 windows do not appear to be direct 400-point cuts from this archive under tested preprocessing"
        if not exact_possible else
        "sample matches support direct raw provenance for a subset"
    )
    final_lines = [
        "# Final Decision",
        "",
        f"Run directory: `{run_dir}`",
        f"Stage path taken: {stage}",
        "",
        "1. Can h5 samples be matched back to raw mat windows?",
        f"   - {'Yes for sampled anchors' if exact_possible else 'No; exact provenance was not established'} under the tested search.",
        "2. high_exact / possible / failed ratios:",
        f"   - high_exact={float(high):.3f}, possible_or_high={float(poss):.3f}, failed={float(failed_ratio):.3f}.",
        "3. If matched, where did they come from?",
        "   - See `sample_matching/sample_window_match_results.csv` and `full_matching/full_window_match_results.csv`.",
        "4. If not, most likely reason:",
        f"   - {likely_reason}. Preprocessing mismatch remains possible, but round2 diagnostics are recorded when applicable.",
        "5. Official h5 most resembles which preprocessing?",
        f"   - Best statistical fallback: `{best_candidate.get('candidate_id', 'n/a')}` "
        f"({best_candidate.get('windowing', 'n/a')} + {best_candidate.get('normalization', 'n/a')}).",
        "6. Can unused raw windows extend training?",
        f"   - {'Only if exact provenance anchors are trusted; otherwise use fallback candidates cautiously.' if exact_possible else 'Not as true unused same-source windows; only as statistically similar external candidates.'}",
        "7. Did reconstructed supplement exceed old supplement?",
        f"   - {train_decision.get('reconstructed_exceeds_old_supplement')}.",
        "8. Did reconstructed supplement exceed no_external?",
        f"   - {train_decision.get('reconstructed_exceeds_no_external')}.",
        "9. Is external worth continuing?",
        f"   - {train_decision.get('recommendation')}.",
        "10. First 3 files to inspect tomorrow:",
        "   - `final_decision.md`",
        "   - `sample_matching/matching_feasibility_report.md`",
        "   - `training_check/reconstructed_external_training_results.csv`",
        "",
    ]
    if errors:
        final_lines += ["## Runtime Errors", ""] + [f"- {e}" for e in errors]
    (run_dir / "final_decision.md").write_text("\n".join(final_lines), encoding="utf-8")
    (dirs["reports"] / "raw_matching_final_report.md").write_text("\n".join(final_lines), encoding="utf-8")
    pre_lines = [
        "# Inferred Preprocessing Summary",
        "",
        f"Exact matching gate: {'passed' if exact_possible else 'not passed'}",
        f"Best statistical candidate: {best_candidate.get('candidate_id', 'n/a')}",
        "",
        "## Top Statistical Candidates",
    ]
    for i, row in enumerate(ranked[:10], start=1):
        pre_lines.append(f"{i}. `{row['candidate_id']}` score={float(row['overall_stat_score']):.4f} windowing={row['windowing']} norm={row['normalization']}")
    (dirs["reports"] / "inferred_preprocessing_summary.md").write_text("\n".join(pre_lines), encoding="utf-8")
    usability = [
        "# External Usability Decision",
        "",
        f"No external best val acc: {train_decision.get('no_external_best_val_acc')}",
        f"Old supplement best val acc: {train_decision.get('old_supplement_best_val_acc')}",
        f"Reconstructed best val acc: {train_decision.get('reconstructed_best_val_acc')}",
        "",
        train_decision.get("recommendation", ""),
    ]
    (dirs["reports"] / "external_usability_decision.md").write_text("\n".join(usability), encoding="utf-8")
    summary = {
        "run_dir": str(run_dir),
        "stage": stage,
        "sample_summary": sample_summary,
        "round2_summary": round2_summary,
        "full_info": full_info,
        "best_statistical_candidate": best_candidate,
        "training_decision": train_decision,
        "errors": errors,
    }
    write_json(run_dir / "results_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive-zip", type=Path, default=DEFAULT_ZIP)
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--candidate-samples-per-class", type=int, default=30)
    p.add_argument("--skip-training", action="store_true", help="For debugging only; default runs Stage 4.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.output_root / f"run_{now_tag()}"
    dirs = mkdirs(run_dir)
    errors: List[str] = []
    stage = "unknown"
    round2_summary = None
    full_info: Dict[str, object] = {}
    ranked: List[Dict[str, object]] = []
    training_rows: List[Dict[str, object]] = []
    try:
        print(f"[stage0] run_dir={run_dir}", flush=True)
        official = audit_official_h5(dirs)
        archive_rows, labels = audit_archive(args.archive_zip, dirs)
        print("[stage1] sample provenance matching", flush=True)
        sample_rows, sample_summary = sample_matching(args.archive_zip, labels, official["train"], dirs)
        stage = choose_stage(sample_summary)
        print(f"[stage1] decision={stage} summary={sample_summary}", flush=True)
        if stage == "2A":
            full_info = run_full_matching_placeholder(sample_rows, dirs)
        elif stage == "2B":
            top_names = [name for name, _count in Counter(r["best_preprocess"] for r in sample_rows).most_common(4)]
            rows2, round2_summary = sample_matching(args.archive_zip, labels, official["train"], dirs, round_name="round2", specs=round2_preprocess_specs(top_names))
            if float(round2_summary["high_exact_ratio"]) >= 0.30:
                stage = "2A_after_2B"
                full_info = run_full_matching_placeholder(rows2, dirs)
            else:
                stage = "2C_after_2B"
        if stage.startswith("2C") or stage == "2C":
            full_info = run_full_matching_placeholder(sample_rows, dirs)
            print("[stage2C] statistical fallback candidate generation", flush=True)
            paths = generate_candidates(args.archive_zip, labels, official["train"], dirs, samples_per_class=args.candidate_samples_per_class)
            ranked = rank_candidates(paths, official["train"], dirs)
        else:
            print("[stage3] exact path still also generating fallback candidates for training comparison", flush=True)
            paths = generate_candidates(args.archive_zip, labels, official["train"], dirs, samples_per_class=args.candidate_samples_per_class)
            ranked = rank_candidates(paths, official["train"], dirs)
        top_candidates = copy_top_candidates(ranked, dirs, top_k=5)
        if args.skip_training:
            errors.append("Stage 4 skipped by --skip-training.")
        else:
            training_rows = training_check(top_candidates, dirs)
        train_decision = infer_external_decision(training_rows)
        make_summary_reports(run_dir, dirs, stage, sample_summary, round2_summary, ranked, train_decision, full_info, errors)
        print(f"[done] {run_dir}", flush=True)
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        (run_dir / "error_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        sample_summary = locals().get("sample_summary", {"n": 0, "quality_counts": {}, "high_exact_ratio": 0.0, "possible_or_high_ratio": 0.0})
        train_decision = infer_external_decision(training_rows)
        make_summary_reports(run_dir, dirs, stage, sample_summary, round2_summary, ranked, train_decision, full_info, errors)
        print(traceback.format_exc(), flush=True)
        raise


if __name__ == "__main__":
    main()
