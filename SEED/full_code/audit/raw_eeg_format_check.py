#!/usr/bin/env python3
"""Check whether the course H5 files expose raw-like EEG tensors."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Dict

import h5py
import numpy as np

import two_day_rescue_utils as u


def inspect_h5(path: Path, require_y: bool) -> Dict[str, object]:
    with h5py.File(path, "r") as h5:
        info: Dict[str, object] = {"path": str(path), "keys": list(h5.keys())}
        X = h5["X"]
        info["X_shape"] = list(X.shape)
        info["X_dtype"] = str(X.dtype)
        arr = np.asarray(X[()], dtype=np.float32)
        info["value_min"] = float(np.nanmin(arr))
        info["value_max"] = float(np.nanmax(arr))
        info["value_mean"] = float(np.nanmean(arr))
        info["value_std"] = float(np.nanstd(arr))
        info["has_nan"] = bool(np.isnan(arr).any())
        info["has_inf"] = bool(np.isinf(arr).any())
        if require_y and "y" in h5:
            y = np.asarray(h5["y"][()], dtype=int)
            info["y_shape"] = list(y.shape)
            info["y_distribution"] = {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
        return info


def main() -> None:
    out_dir = u.OUTPUT_ROOT / f"raw_like_eeg_feasibility_{u.stamp()}"
    out_dir.mkdir(parents=True, exist_ok=False)
    result: Dict[str, object] = {"status": "running", "output_dir": str(out_dir), "errors": []}
    try:
        train = inspect_h5(u.DATA / "train.h5", True)
        val = inspect_h5(u.DATA / "val.h5", True)
        test = inspect_h5(u.DATA / "test_x_only.h5", False)
        shape = train["X_shape"]
        is_raw_like = bool(len(shape) == 3 and shape[1] == 62 and shape[2] >= 64 and val["X_shape"][1:] == shape[1:] and test["X_shape"][1:] == shape[1:])
        is_feature_vector = bool(len(shape) == 2)
        verdict = "raw_like_eeg" if is_raw_like else "clean_feature_vector_or_unknown"
        readme = [
            "# Raw-Like EEG Feasibility Check",
            "",
            f"- Verdict: `{verdict}`",
            f"- Train X shape: `{train['X_shape']}`",
            f"- Val X shape: `{val['X_shape']}`",
            f"- Test X shape: `{test['X_shape']}`",
            f"- Train dtype: `{train['X_dtype']}`, Val dtype: `{val['X_dtype']}`, Test dtype: `{test['X_dtype']}`",
            f"- Train value range: `{train['value_min']:.6g}` to `{train['value_max']:.6g}`, mean/std `{train['value_mean']:.6g}` / `{train['value_std']:.6g}`",
            f"- Val value range: `{val['value_min']:.6g}` to `{val['value_max']:.6g}`, mean/std `{val['value_mean']:.6g}` / `{val['value_std']:.6g}`",
            f"- Test value range: `{test['value_min']:.6g}` to `{test['value_max']:.6g}`, mean/std `{test['value_mean']:.6g}` / `{test['value_std']:.6g}`",
            f"- NaN/Inf flags: train `{train['has_nan']}/{train['has_inf']}`, val `{val['has_nan']}/{val['has_inf']}`, test `{test['has_nan']}/{test['has_inf']}`",
            f"- Train label shape/distribution: `{train.get('y_shape')}` / `{train.get('y_distribution')}`",
            f"- Val label shape/distribution: `{val.get('y_shape')}` / `{val.get('y_distribution')}`",
            f"- Channels close to 62: `{len(shape) == 3 and shape[1] == 62}`",
            f"- Fixed time axis: `{len(shape) == 3 and val['X_shape'][2] == shape[2] and test['X_shape'][2] == shape[2]}`",
            f"- Looks like flat clean feature vector: `{is_feature_vector}`",
            "",
        ]
        if is_raw_like:
            readme.extend(
                [
                    "The course H5 files expose `N x channels x time` tensors, so a simple Conv1d-over-time baseline is feasible. The companion `simple_1dcnn_baseline.py` keeps this as a modest neural baseline rather than a complex CRNN.",
                    "",
                    "It is suitable for MLP after feature extraction and suitable for a neural probability stacker because both consume fixed-dimensional representations. Direct SLEEP-style CNN+BiLSTM/CRNN is possible structurally, but it remains risky because the sample count is small and block/subject distribution shift is visible in previous audits.",
                ]
            )
        else:
            readme.extend(
                [
                    "The current data is not suitable for directly applying a SLEEP-style CRNN because it lacks a clear fixed `channels x time` structure. The project should retain MLP and neural probability stacker experiments as neural baselines.",
                    "",
                    "It is suitable for MLP if the representation is fixed-dimensional, and suitable for a neural probability stacker when saved candidate probabilities are available. It should not be hard-reshaped into fake channels/time for the main result.",
                ]
            )
        (out_dir / "README_raw_like_eeg_feasibility.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
        result.update({"status": "completed", "verdict": verdict, "is_raw_like": is_raw_like, "train": train, "val": val, "test": test})
    except Exception as exc:
        result["status"] = "failed_partial"
        result["errors"].append({"error": repr(exc), "trace": traceback.format_exc()})
        (out_dir / "README_raw_like_eeg_feasibility.md").write_text(
            "# Raw-Like EEG Feasibility Check\n\nRun failed; see `run_results.json` for traceback.\n", encoding="utf-8"
        )
    finally:
        u.write_json(out_dir / "run_results.json", result)
        print(f"OUTPUT_DIR={out_dir}")


if __name__ == "__main__":
    main()
