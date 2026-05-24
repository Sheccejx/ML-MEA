#!/usr/bin/env python3
"""Shared utilities for the final two-day SEED rescue experiments."""

from __future__ import annotations

import csv
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "course_project" / "SEED"
OUTPUT_ROOT = ROOT / "outputs_experiments"
V2_OUT = OUTPUT_ROOT / "clean_feature_block_fusion_v2_20260522_085528"
V25_OUT = OUTPUT_ROOT / "clean_feature_block_fusion_v25_20260522_150944"
EPS = 1e-10


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
            out = {}
            for key in keys:
                val = row.get(key, "")
                out[key] = json.dumps(jsonable(val), ensure_ascii=False) if isinstance(val, (dict, list, tuple, np.ndarray)) else jsonable(val)
            writer.writerow(out)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:180]


def normalize_prob(prob: np.ndarray) -> np.ndarray:
    p = np.asarray(prob, dtype=np.float64)
    p = np.nan_to_num(p, nan=1 / 3, posinf=1.0, neginf=EPS)
    p = np.clip(p, EPS, None)
    return p / p.sum(axis=1, keepdims=True)


def evaluate_prob(y: np.ndarray, prob: np.ndarray, idx: Optional[np.ndarray] = None) -> Dict[str, Any]:
    pred = normalize_prob(prob).argmax(axis=1)
    yy = y if idx is None else y[idx]
    pp = pred if idx is None else pred[idx]
    rec = recall_score(yy, pp, labels=[0, 1, 2], average=None, zero_division=0)
    cm = confusion_matrix(yy, pp, labels=[0, 1, 2]).astype(int)
    return {
        "val_acc": float(accuracy_score(yy, pp)),
        "macro_f1": float(f1_score(yy, pp, average="macro", zero_division=0)),
        "min_recall": float(rec.min()),
        "per_class_recall": {str(i): float(v) for i, v in enumerate(rec)},
        "confusion_matrix": cm.tolist(),
        "prediction_distribution": {str(k): int(v) for k, v in Counter(pp.tolist()).items()},
    }


def score_metrics(m: Dict[str, Any]) -> float:
    return float(m["val_acc"] + 0.2 * m["macro_f1"] + 0.1 * m["min_recall"])


def write_seed_from_prob(path: Path, prob: np.ndarray) -> Dict[str, Any]:
    p = normalize_prob(prob)
    pred = p.argmax(axis=1)
    with path.open("w", encoding="utf-8", newline="\n") as fp:
        for v in pred:
            fp.write(f"{int(v)}\n")
    return validate_seed_against_prob(path, p)


def validate_seed_against_prob(path: Path, prob: np.ndarray) -> Dict[str, Any]:
    vals = []
    ok = True
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            v = int(line.strip())
            vals.append(v)
            ok = ok and v in {0, 1, 2}
        except Exception:
            ok = False
    argmax = normalize_prob(prob).argmax(axis=1).astype(int)
    vals_arr = np.asarray(vals, dtype=int) if vals else np.asarray([], dtype=int)
    return {
        "path": str(path),
        "line_count": len(vals),
        "exactly_450_lines": len(vals) == 450,
        "labels_only_0_1_2": bool(ok),
        "matches_probability_argmax": bool(len(vals_arr) == len(argmax) and np.array_equal(vals_arr, argmax)),
        "distribution": {str(k): int(v) for k, v in Counter(vals).items()},
    }


def draw_confusion_matrix_png(path: Path, cm: Sequence[Sequence[int]], title: str = "Confusion Matrix") -> None:
    cm_arr = np.asarray(cm, dtype=int)
    cell = 90
    left = 110
    top = 80
    w = left + cell * 3 + 40
    h = top + cell * 3 + 70
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        small = ImageFont.truetype("arial.ttf", 14)
        big = ImageFont.truetype("arial.ttf", 24)
    except Exception:
        font = small = big = ImageFont.load_default()
    draw.text((20, 20), title, fill=(20, 20, 20), font=big)
    vmax = max(int(cm_arr.max()), 1)
    for i in range(3):
        draw.text((left + i * cell + 35, top - 28), f"Pred {i}", fill=(20, 20, 20), font=small)
        draw.text((20, top + i * cell + 35), f"True {i}", fill=(20, 20, 20), font=small)
        for j in range(3):
            val = int(cm_arr[i, j])
            shade = int(245 - 155 * val / vmax)
            color = (shade, shade + 5 if shade < 250 else 250, 255)
            x0 = left + j * cell
            y0 = top + i * cell
            draw.rectangle([x0, y0, x0 + cell, y0 + cell], fill=color, outline=(80, 110, 140), width=2)
            draw.text((x0 + 36, y0 + 34), str(val), fill=(0, 0, 0), font=font)
    img.save(path)


def make_splits(n: int) -> Dict[str, List[Tuple[str, np.ndarray, np.ndarray]]]:
    rng = np.random.default_rng(20260522)
    shuffled = np.arange(n)
    rng.shuffle(shuffled)
    a = np.sort(shuffled[: n // 2])
    b = np.sort(shuffled[n // 2 :])
    block_ids = np.arange(n) // 10
    even = np.where(block_ids % 2 == 0)[0]
    odd = np.where(block_ids % 2 == 1)[0]
    return {
        "random_half": [("fold1", a, b), ("fold2", b, a)],
        "block10_even_odd": [("fold1", even, odd), ("fold2", odd, even)],
    }


def split_metrics(y: np.ndarray, prob: np.ndarray, splits: Dict[str, List[Tuple[str, np.ndarray, np.ndarray]]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    full = evaluate_prob(y, prob)["val_acc"]
    for split_name, folds in splits.items():
        accs, f1s, mins = [], [], []
        for _, _, eval_idx in folds:
            m = evaluate_prob(y, prob, eval_idx)
            accs.append(m["val_acc"])
            f1s.append(m["macro_f1"])
            mins.append(m["min_recall"])
        out[f"{split_name}_eval_acc_mean"] = float(np.mean(accs))
        out[f"{split_name}_eval_acc_min"] = float(np.min(accs))
        out[f"{split_name}_eval_macro_f1_mean"] = float(np.mean(f1s))
        out[f"{split_name}_eval_min_recall_mean"] = float(np.mean(mins))
        out[f"{split_name}_drop_from_full"] = float(full - np.mean(accs))
    out["combined_split_eval_acc_mean"] = float(np.mean([out["random_half_eval_acc_mean"], out["block10_even_odd_eval_acc_mean"]]))
    return out


def load_feature_cache(name: str = "common_average_reference__stats_band_covpca20") -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    cache = V2_OUT / "feature_cache"
    npz = cache / f"{name}.npz"
    if npz.exists():
        z = np.load(npz)
        return z["train"].astype(np.float32), z["val"].astype(np.float32), z["test"].astype(np.float32), {"feature_cache": str(npz), "feature_name": name}
    prefix = name.replace("__", "_")
    paths = [cache / f"{prefix}_{split}.npy" for split in ("train", "val", "test")]
    if all(p.exists() for p in paths):
        return tuple(np.load(p).astype(np.float32) for p in paths) + ({"feature_cache": str(cache), "feature_name": name},)
    raise FileNotFoundError(f"Feature cache not found for {name}")


def candidate_prob_files() -> List[Dict[str, Any]]:
    sources = [
        ("v25", V25_OUT, V25_OUT / "v2_v25_combined_pool_summary.csv"),
        ("v2", V2_OUT, V2_OUT / "all_clean_candidates_summary.csv"),
    ]
    out: List[Dict[str, Any]] = []
    seen = set()
    for tag, root, summary in sources:
        if not summary.exists():
            continue
        for row in read_csv(summary):
            name = row.get("name") or row.get("candidate_name")
            if not name:
                continue
            val_path = root / "all_val_probs" / f"{safe_name(name)}.npy"
            test_path = root / "all_test_probs" / f"{safe_name(name)}.npy"
            if not val_path.exists() or not test_path.exists():
                continue
            key = (str(val_path), str(test_path))
            if key in seen:
                continue
            seen.add(key)
            out.append({"source": tag, "name": name, "val_path": val_path, "test_path": test_path, "row": row})
    return out


def load_candidate_pool(y: np.ndarray, max_candidates: int = 120) -> List[Tuple[str, np.ndarray, np.ndarray, Dict[str, Any]]]:
    pool = []
    seen_probs = set()
    for item in candidate_prob_files():
        pv = normalize_prob(np.load(item["val_path"]))
        pt = normalize_prob(np.load(item["test_path"]))
        if pv.shape != (len(y), 3) or pt.shape != (450, 3):
            continue
        key = (np.round(pv, 8).tobytes(), np.round(pt, 8).tobytes())
        if key in seen_probs:
            continue
        seen_probs.add(key)
        m = evaluate_prob(y, pv)
        meta = {"source": item["source"], "name": item["name"], **m, "score": score_metrics(m)}
        pool.append((f"{item['source']}__{item['name']}", pv, pt, meta))
    pool.sort(key=lambda x: (x[3]["val_acc"], x[3]["macro_f1"], x[3]["score"]), reverse=True)
    return pool[:max_candidates]


def md_table(rows: Sequence[Dict[str, Any]], cols: Sequence[str]) -> str:
    lines = ["|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        vals = []
        for col in cols:
            val = row.get(col, "")
            vals.append(f"{val:.6f}" if isinstance(val, float) else str(val).replace("\n", " "))
        lines.append("|" + "|".join(vals) + "|")
    return "\n".join(lines)
