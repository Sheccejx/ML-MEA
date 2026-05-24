#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inspect external EEG files and convert safe SEED-like data to H5.

The script is intentionally conservative. It only writes supervised
`supplement_seed_like.h5` when data can be represented as `(N, 62, 400)` and
labels are already `0/1/2` or can be mapped from official SEED `-1/0/1`.
Feature-only archives such as `(N, 5, 62)` are reported, not forced into raw EEG.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parent
DEFAULT_SCAN_DIRS = [
    ROOT / "course_project" / "external_data",
    ROOT / "archive",
    ROOT / "course_project" / "SEED" / "SEED",
]
DEFAULT_OUTPUT = ROOT / "outputs_external" / "supplement_seed_like.h5"
DEFAULT_REPORT = ROOT / "outputs_external" / "conversion_report.txt"

SEED_CHANNELS = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "FT7",
    "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6",
    "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2",
]
TRIAL_LABELS_ORIGINAL = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
WINDOW_LENGTH = 400
DEFAULT_STRIDE = 400
EEG_KEY_RE = re.compile(r"(?:^|_)eeg_?(\d+)$", re.IGNORECASE)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def short_counter(arr: np.ndarray) -> Dict[str, int]:
    vals, counts = np.unique(arr, return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(vals, counts)}


def scan_files(scan_dirs: Iterable[Path]) -> List[Path]:
    exts = {".h5", ".mat", ".npz", ".npy", ".csv", ".pkl", ".edf", ".zip"}
    files: List[Path] = []
    for root in scan_dirs:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in exts:
                files.append(path)
    return sorted(set(files))


def inspect_h5(path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"path": str(path), "file_type": "h5", "can_supervised_seed_like": False}
    with h5py.File(path, "r") as h5:
        info["attrs"] = jsonable(dict(h5.attrs))
        info["keys"] = list(h5.keys())
        datasets = {}
        for key in h5.keys():
            obj = h5[key]
            if isinstance(obj, h5py.Dataset):
                datasets[key] = {"shape": list(obj.shape), "dtype": str(obj.dtype), "attrs": jsonable(dict(obj.attrs))}
        info["datasets"] = datasets
        if "X" in h5 and "y" in h5:
            x_shape = tuple(int(v) for v in h5["X"].shape)
            y_shape = tuple(int(v) for v in h5["y"].shape)
            info["eeg_shape"] = list(x_shape)
            info["y_shape"] = list(y_shape)
            if len(y_shape) == 1 and y_shape[0] > 0:
                y = h5["y"][()]
                info["labels"] = short_counter(y)
                info["label_values"] = sorted(int(v) for v in np.unique(y))
            info["can_supervised_seed_like"] = (
                len(x_shape) == 3
                and x_shape[1] == 62
                and x_shape[2] == WINDOW_LENGTH
                and len(y_shape) == 1
                and y_shape[0] == x_shape[0]
                and set(info.get("label_values", [])) <= {0, 1, 2}
            )
            if info["can_supervised_seed_like"]:
                info["decision"] = "usable_supervised_seed_like_h5"
        elif any(k.startswith("trial") for k in h5.keys()):
            info["decision"] = "trial_level_h5_not_used_as_external_by_default"
    return info


def inspect_npz(path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"path": str(path), "file_type": "npz", "can_supervised_seed_like": False}
    z = np.load(path)
    arrays = {}
    for key in z.files:
        arr = z[key]
        item = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
        if arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer):
            item["counts"] = short_counter(arr)
            item["values"] = sorted(int(v) for v in np.unique(arr))[:30]
        arrays[key] = item
    info["arrays"] = arrays
    if any(item["shape"][-2:] == [5, 62] for item in arrays.values() if len(item["shape"]) >= 2):
        info["decision"] = "feature_only_band_by_channel_not_raw_62x400"
    return info


def trial_number_from_key(key: str) -> Optional[int]:
    match = EEG_KEY_RE.search(key)
    return int(match.group(1)) if match else None


def detect_eeg_trial_keys(mat: Dict[str, np.ndarray]) -> List[Tuple[int, str]]:
    found = []
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        trial_no = trial_number_from_key(key)
        if trial_no is None:
            continue
        arr = np.asarray(value)
        if arr.ndim == 2 and 62 in arr.shape:
            found.append((trial_no, key))
    return sorted(found)


def inspect_mat_dict(name: str, mat: Dict[str, np.ndarray]) -> Dict[str, Any]:
    shapes = {}
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        if hasattr(value, "shape"):
            shapes[key] = list(value.shape)
    trials = detect_eeg_trial_keys(mat)
    return {
        "path": name,
        "file_type": "mat",
        "keys": list(shapes.keys()),
        "key_shapes": shapes,
        "eeg_trial_keys": [key for _, key in trials],
        "can_convert_raw_seed_trials": len(trials) > 0,
        "label_mapping": "-1/0/1 trial labels -> 0/1/2" if len(trials) > 0 else None,
    }


def inspect_mat(path: Path) -> Dict[str, Any]:
    return inspect_mat_dict(str(path), loadmat(path))


def inspect_zip(path: Path, max_members: int = 3) -> Dict[str, Any]:
    info: Dict[str, Any] = {"path": str(path), "file_type": "zip", "can_supervised_seed_like": False}
    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        mats = [name for name in names if name.lower().endswith(".mat")]
        npzs = [name for name in names if name.lower().endswith(".npz")]
        info["n_files"] = len(names)
        info["mat_files"] = len(mats)
        info["npz_files"] = len(npzs)
        samples = []
        for name in mats[:max_members]:
            try:
                samples.append(inspect_mat_dict(name, loadmat(io.BytesIO(zf.read(name)))))
            except Exception as exc:
                samples.append({"path": name, "error": str(exc)})
        info["mat_samples"] = samples
        info["can_supervised_seed_like"] = any(s.get("can_convert_raw_seed_trials") for s in samples)
        if info["can_supervised_seed_like"]:
            info["decision"] = "zip_contains_seed_style_mat_trials"
    return info


def inspect_file(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".h5":
            return inspect_h5(path)
        if suffix == ".npz":
            return inspect_npz(path)
        if suffix == ".mat":
            return inspect_mat(path)
        if suffix == ".zip":
            return inspect_zip(path)
        return {"path": str(path), "file_type": suffix.lstrip("."), "decision": "not_inspected_for_conversion"}
    except Exception as exc:
        return {"path": str(path), "file_type": suffix.lstrip("."), "error": str(exc)}


def ensure_62_by_t(arr: np.ndarray) -> Optional[np.ndarray]:
    x = np.asarray(arr)
    if x.ndim != 2:
        return None
    if x.shape[0] == 62:
        return x.astype(np.float32, copy=False)
    if x.shape[1] == 62:
        return x.T.astype(np.float32, copy=False)
    return None


def parse_subject_session(name: str) -> Tuple[int, int]:
    base = Path(name).stem
    m = re.match(r"(\d+)_", base)
    subject = int(m.group(1)) if m else -1
    return subject, -1


def channel_stats_from_h5(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        X = h5["X"]
        sums = np.zeros((X.shape[1],), dtype=np.float64)
        sums2 = np.zeros((X.shape[1],), dtype=np.float64)
        count = 0
        for start in range(0, X.shape[0], 512):
            arr = X[start:start + 512].astype(np.float64)
            sums += arr.sum(axis=(0, 2))
            sums2 += np.square(arr).sum(axis=(0, 2))
            count += arr.shape[0] * arr.shape[2]
    mean = (sums / max(1, count)).reshape(-1, 1).astype(np.float32)
    var = np.maximum(sums2 / max(1, count) - np.square(mean.reshape(-1)), 1e-12)
    std = np.sqrt(var).reshape(-1, 1).astype(np.float32)
    return mean, std


def balanced_indices_by_class(
    y: np.ndarray,
    samples_per_class: Optional[int],
    max_windows: Optional[int],
    windowing: str,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: List[np.ndarray] = []
    labels = sorted(int(v) for v in np.unique(y))
    per_class = samples_per_class
    if per_class is None and max_windows is not None:
        per_class = max(1, max_windows // max(1, len(labels)))
    for label in labels:
        idx = np.where(y == label)[0]
        if per_class is None or per_class >= len(idx):
            chosen = idx
        elif windowing == "first_windows":
            chosen = idx[:per_class]
        elif windowing == "middle_windows":
            start = max(0, (len(idx) - per_class) // 2)
            chosen = idx[start:start + per_class]
        else:
            chosen = rng.choice(idx, size=per_class, replace=False)
        selected.append(np.asarray(chosen, dtype=np.int64))
    out = np.concatenate(selected) if selected else np.asarray([], dtype=np.int64)
    if windowing in {"random_windows", "non_overlap_balanced", "stride_200_overlap_balanced"}:
        rng.shuffle(out)
    else:
        out = np.sort(out)
    if max_windows is not None and len(out) > max_windows:
        out = np.sort(rng.choice(out, size=max_windows, replace=False))
    return out.astype(np.int64)


def stat_distance_filter_indices(
    source_h5: Path,
    candidate_idx: np.ndarray,
    official_train_h5: Path,
    ratio: float,
) -> np.ndarray:
    if ratio <= 0 or ratio >= 1 or len(candidate_idx) == 0:
        return candidate_idx
    official_mean, official_std = channel_stats_from_h5(official_train_h5)
    keep_n = max(1, int(round(len(candidate_idx) * ratio)))
    distances: List[Tuple[float, int]] = []
    with h5py.File(source_h5, "r") as h5:
        for idx in candidate_idx.tolist():
            x = h5["X"][idx].astype(np.float32)
            m = x.mean(axis=1, keepdims=True)
            s = x.std(axis=1, keepdims=True)
            d = float(np.mean(np.square((m - official_mean) / (official_std + 1e-6))) + np.mean(np.square(np.log((s + 1e-6) / (official_std + 1e-6)))))
            distances.append((d, int(idx)))
    distances.sort(key=lambda item: item[0])
    return np.asarray([idx for _, idx in distances[:keep_n]], dtype=np.int64)


def normalize_windows(
    X: np.ndarray,
    normalization: str,
    external_mean: Optional[np.ndarray],
    external_std: Optional[np.ndarray],
    official_mean: Optional[np.ndarray],
    official_std: Optional[np.ndarray],
) -> np.ndarray:
    X = X.astype(np.float32, copy=False)
    if normalization == "no_zscore":
        return X
    if normalization in {"per_window_zscore", "per_trial_zscore"}:
        mean = X.mean(axis=2, keepdims=True)
        std = X.std(axis=2, keepdims=True)
        return ((X - mean) / np.maximum(std, 1e-6)).astype(np.float32)
    if normalization == "external_train_stat_zscore":
        if external_mean is None or external_std is None:
            external_mean = X.mean(axis=(0, 2), keepdims=False).reshape(X.shape[1], 1)
            external_std = X.std(axis=(0, 2), keepdims=False).reshape(X.shape[1], 1)
        return ((X - external_mean.reshape(1, X.shape[1], 1)) / np.maximum(external_std.reshape(1, X.shape[1], 1), 1e-6)).astype(np.float32)
    if normalization == "align_to_official_train_stat":
        if official_mean is None or official_std is None:
            raise ValueError("align_to_official_train_stat requires --official-train-h5")
        if external_mean is None or external_std is None:
            external_mean = X.mean(axis=(0, 2), keepdims=False).reshape(X.shape[1], 1)
            external_std = X.std(axis=(0, 2), keepdims=False).reshape(X.shape[1], 1)
        z = (X - external_mean.reshape(1, X.shape[1], 1)) / np.maximum(external_std.reshape(1, X.shape[1], 1), 1e-6)
        return (z * official_std.reshape(1, X.shape[1], 1) + official_mean.reshape(1, X.shape[1], 1)).astype(np.float32)
    raise ValueError(f"Unknown normalization: {normalization}")


def create_seed_like_h5_from_existing(
    source_h5: Path,
    output_h5: Path,
    max_windows: Optional[int],
    seed: int,
    normalization: str = "no_zscore",
    windowing: str = "random_windows",
    samples_per_class: Optional[int] = None,
    official_train_h5: Optional[Path] = None,
    select_by_stat_distance: bool = False,
    stat_distance_ratio: float = 1.0,
) -> Dict[str, Any]:
    with h5py.File(source_h5, "r") as src:
        x_shape = tuple(int(v) for v in src["X"].shape)
        y = src["y"][()]
        if not (len(x_shape) == 3 and x_shape[1:] == (62, 400) and set(np.unique(y).tolist()) <= {0, 1, 2}):
            raise ValueError(f"Not a supervised SEED-like H5: {source_h5}")
        n_total = x_shape[0]
        idx = balanced_indices_by_class(y, samples_per_class, max_windows, windowing, seed)
        if select_by_stat_distance:
            if official_train_h5 is None:
                raise ValueError("--select-by-stat-distance requires --official-train-h5")
            idx = stat_distance_filter_indices(source_h5, idx, official_train_h5, stat_distance_ratio)
        idx = np.asarray(idx, dtype=np.int64)
        external_mean = external_std = official_mean = official_std = None
        if normalization in {"external_train_stat_zscore", "align_to_official_train_stat"}:
            selected_for_stats = src["X"][np.sort(idx)].astype(np.float32)
            external_mean = selected_for_stats.mean(axis=(0, 2), keepdims=False).reshape(62, 1).astype(np.float32)
            external_std = selected_for_stats.std(axis=(0, 2), keepdims=False).reshape(62, 1).astype(np.float32)
        if normalization == "align_to_official_train_stat":
            if official_train_h5 is None:
                raise ValueError("--normalization align_to_official_train_stat requires --official-train-h5")
            official_mean, official_std = channel_stats_from_h5(official_train_h5)

        output_h5.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_h5, "w") as out:
            x_chunk = (max(1, min(64, len(idx))), 62, 400)
            meta_chunk = (max(1, min(1024, len(idx))),)
            X = out.create_dataset("X", shape=(len(idx), 62, 400), dtype="float32", compression="lzf", chunks=x_chunk)
            yy = out.create_dataset("y", shape=(len(idx),), dtype="int64", compression="lzf", chunks=meta_chunk)
            source = out.create_dataset("source", shape=(len(idx),), dtype="int16", compression="lzf", chunks=meta_chunk)
            subject = out.create_dataset("subject", shape=(len(idx),), dtype="int16", compression="lzf", chunks=meta_chunk)
            trial = out.create_dataset("trial", shape=(len(idx),), dtype="int16", compression="lzf", chunks=meta_chunk)
            window_start = out.create_dataset("window_start", shape=(len(idx),), dtype="int64", compression="lzf", chunks=meta_chunk)
            source_file = out.create_dataset("source_file", shape=(len(idx),), dtype=h5py.string_dtype(encoding="utf-8"))
            chunk = 2048
            source_name = str(source_h5)
            for start in range(0, len(idx), chunk):
                stop = min(start + chunk, len(idx))
                sel = idx[start:stop]
                order = np.argsort(sel)
                sorted_sel = sel[order]
                inv = np.argsort(order)
                arr = src["X"][sorted_sel].astype(np.float32)[inv]
                X[start:stop] = normalize_windows(arr, normalization, external_mean, external_std, official_mean, official_std)
                yy[start:stop] = src["y"][sorted_sel][inv]
                source[start:stop] = 1
                subject[start:stop] = -1
                trial[start:stop] = -1
                window_start[start:stop] = sel * int(src.attrs.get("stride", 400))
                source_file[start:stop] = [source_name] * (stop - start)
            out.attrs["created_at"] = now()
            out.attrs["source_h5"] = str(source_h5)
            out.attrs["source_kind"] = "external_seed_like_h5"
            out.attrs["label_mapping"] = src.attrs.get("label_mapping", "already 0/1/2")
            out.attrs["channel_order"] = json.dumps(SEED_CHANNELS, ensure_ascii=False)
            out.attrs["sampling_rate"] = 200.0
            out.attrs["window_length"] = 400
            out.attrs["normalization"] = normalization
            out.attrs["windowing"] = windowing
            out.attrs["samples_per_class"] = -1 if samples_per_class is None else int(samples_per_class)
            out.attrs["seed"] = int(seed)
            out.attrs["select_by_stat_distance"] = bool(select_by_stat_distance)
            out.attrs["stat_distance_ratio"] = float(stat_distance_ratio)
            out.attrs["selection"] = f"{len(idx)} of {n_total} windows"
            out.attrs["label_distribution"] = json.dumps(short_counter(y[idx]), ensure_ascii=False)
            out.attrs["channel_mean"] = json.dumps(X[:].mean(axis=(0, 2)).astype(float).tolist())
            out.attrs["channel_std"] = json.dumps(X[:].std(axis=(0, 2)).astype(float).tolist())
    return inspect_h5(output_h5)


def convert_seed_mat_zip(
    source_zip: Path,
    output_h5: Path,
    stride: int,
    max_windows: Optional[int],
    seed: int,
    window_zscore: bool,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    with zipfile.ZipFile(source_zip, "r") as zf:
        names = sorted(name for name in zf.namelist() if name.lower().endswith(".mat") and "label" not in name.lower())
        rows: List[Tuple[np.ndarray, int, int, int, str]] = []
        for name in names:
            mat = loadmat(io.BytesIO(zf.read(name)))
            subject_id, session_id = parse_subject_session(name)
            for trial_no, key in detect_eeg_trial_keys(mat):
                arr = ensure_62_by_t(mat[key])
                if arr is None or not 1 <= trial_no <= len(TRIAL_LABELS_ORIGINAL):
                    continue
                y = LABEL_MAP[TRIAL_LABELS_ORIGINAL[trial_no - 1]]
                for start in range(0, arr.shape[1] - WINDOW_LENGTH + 1, stride):
                    win = arr[:, start:start + WINDOW_LENGTH]
                    if window_zscore:
                        win = (win - win.mean(axis=1, keepdims=True)) / np.maximum(win.std(axis=1, keepdims=True), 1e-6)
                    rows.append((win.astype(np.float32), y, subject_id, trial_no, name))
        if max_windows is not None and max_windows < len(rows):
            keep = np.sort(rng.choice(len(rows), size=max_windows, replace=False))
            rows = [rows[int(i)] for i in keep]

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5, "w") as out:
        x_chunk = (max(1, min(64, len(rows))), 62, 400)
        meta_chunk = (max(1, min(1024, len(rows))),)
        X = out.create_dataset("X", data=np.stack([r[0] for r in rows]), dtype="float32", compression="lzf", chunks=x_chunk)
        y = out.create_dataset("y", data=np.asarray([r[1] for r in rows], dtype=np.int64), compression="lzf", chunks=meta_chunk)
        out.create_dataset("source", data=np.ones(len(rows), dtype=np.int16), compression="lzf", chunks=meta_chunk)
        out.create_dataset("subject", data=np.asarray([r[2] for r in rows], dtype=np.int16), compression="lzf", chunks=meta_chunk)
        out.create_dataset("trial", data=np.asarray([r[3] for r in rows], dtype=np.int16), compression="lzf", chunks=meta_chunk)
        out.attrs["created_at"] = now()
        out.attrs["source_zip"] = str(source_zip)
        out.attrs["source_kind"] = "external_seed_mat_zip"
        out.attrs["label_mapping"] = "-1->0, 0->1, 1->2"
        out.attrs["channel_order"] = json.dumps(SEED_CHANNELS, ensure_ascii=False)
        out.attrs["sampling_rate"] = 200.0
        out.attrs["window_length"] = 400
        out.attrs["stride"] = int(stride)
        out.attrs["per_window_channel_zscore"] = bool(window_zscore)
    return inspect_h5(output_h5)


def write_report(path: Path, inspections: List[Dict[str, Any]], conversion: Optional[Dict[str, Any]]) -> None:
    lines = [f"# External Raw EEG Inspection / Conversion Report", "", f"Generated: {now()}", ""]
    for item in inspections:
        lines.append(f"## {item.get('path')}")
        lines.append(f"- file_type: {item.get('file_type')}")
        if item.get("eeg_shape"):
            lines.append(f"- EEG shape: {item['eeg_shape']}")
        if item.get("y_shape"):
            lines.append(f"- y shape: {item['y_shape']}")
        if item.get("labels"):
            lines.append(f"- labels: {item['labels']}")
        if item.get("label_values"):
            lines.append(f"- label values: {item['label_values']}")
        if item.get("datasets"):
            lines.append(f"- datasets: {json.dumps(item['datasets'], ensure_ascii=False)}")
        if item.get("arrays"):
            lines.append(f"- arrays: {json.dumps(item['arrays'], ensure_ascii=False)}")
        if item.get("mat_files") is not None:
            lines.append(f"- mat_files: {item.get('mat_files')}")
        if item.get("mat_samples"):
            first = item["mat_samples"][0]
            lines.append(f"- sample mat keys/shapes: {json.dumps(first.get('key_shapes', {}), ensure_ascii=False)}")
        lines.append(f"- decision: {item.get('decision', 'not_seed_like_supervised')}")
        lines.append("")
    if conversion:
        lines.append("## Conversion Output")
        lines.append(f"- output: {conversion.get('path')}")
        lines.append(f"- X shape: {conversion.get('eeg_shape')}")
        lines.append(f"- y shape: {conversion.get('y_shape')}")
        lines.append(f"- labels: {conversion.get('labels')}")
        lines.append(f"- decision: {conversion.get('decision')}")
        lines.append("")
    lines.append("## Safety Notes")
    lines.append("- SLEEP data is not converted or mixed into SEED supervised training.")
    lines.append("- Feature-only `(N, 5, 62)` archives are not forced into raw `(N, 62, 400)` windows.")
    lines.append("- Validation/test labels are not read for supplement conversion.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", type=Path, action="append", default=None)
    parser.add_argument("--source", type=Path, default=None, help="Source to convert. If omitted, choose first safe SEED-like external H5/zip.")
    parser.add_argument("--output-h5", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-windows", type=int, default=3000, help="Cap output for a lightweight supplement. Use 0 for all.")
    parser.add_argument("--normalization", choices=["no_zscore", "per_window_zscore", "per_trial_zscore", "external_train_stat_zscore", "align_to_official_train_stat"], default="no_zscore")
    parser.add_argument("--windowing", choices=["first_windows", "middle_windows", "random_windows", "non_overlap_balanced", "stride_200_overlap_balanced"], default="random_windows")
    parser.add_argument("--window-size", type=int, default=WINDOW_LENGTH)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--samples-per-class", type=int, default=None)
    parser.add_argument("--official-train-h5", type=Path, default=ROOT / "course_project" / "SEED" / "train.h5")
    parser.add_argument("--select-by-stat-distance", action="store_true")
    parser.add_argument("--stat-distance-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--no-window-zscore", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.window_size != WINDOW_LENGTH:
        raise ValueError(f"This project expects 400-point SEED windows; got --window-size {args.window_size}")
    scan_dirs = args.scan_dir or DEFAULT_SCAN_DIRS
    files = scan_files(scan_dirs)
    inspections = [inspect_file(path) for path in files]
    print(f"scanned files: {len(inspections)}")
    for item in inspections:
        print(f"- {item.get('path')} | {item.get('file_type')} | {item.get('decision', 'not_seed_like_supervised')}")

    conversion = None
    if not args.inspect_only:
        max_windows = None if args.max_windows == 0 else args.max_windows
        source = args.source
        if source is None:
            safe = [Path(item["path"]) for item in inspections if item.get("can_supervised_seed_like") and "external_data" in item["path"]]
            if not safe:
                safe = [Path(item["path"]) for item in inspections if item.get("can_supervised_seed_like")]
            if not safe:
                raise RuntimeError("No safe supervised SEED-like external source found. Use inspect-only or self-supervised pretrain.")
            source = safe[0]
        if source.suffix.lower() == ".h5":
            conversion = create_seed_like_h5_from_existing(
                source,
                args.output_h5,
                max_windows,
                args.seed,
                normalization=args.normalization,
                windowing=args.windowing,
                samples_per_class=args.samples_per_class,
                official_train_h5=args.official_train_h5,
                select_by_stat_distance=args.select_by_stat_distance,
                stat_distance_ratio=args.stat_distance_ratio,
            )
        elif source.suffix.lower() == ".zip":
            conversion = convert_seed_mat_zip(source, args.output_h5, args.stride, max_windows, args.seed, not args.no_window_zscore)
        else:
            raise ValueError(f"Conversion source must be a safe H5 or SEED-style MAT zip, got {source}")
        print(f"converted: {args.output_h5}")
        print(f"X shape: {conversion.get('eeg_shape')} y shape: {conversion.get('y_shape')} labels: {conversion.get('labels')}")

    write_report(args.report, inspections, conversion)
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
