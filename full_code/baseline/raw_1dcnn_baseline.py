#!/usr/bin/env python3
"""Simple raw-like EEG Conv1d baseline."""

from __future__ import annotations

import traceback
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import car_block_feature_refinement as base
import two_day_rescue_utils as u


class SimpleEEGConv1D(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(64, 96, kernel_size=5, padding=2),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(96, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        z = z.mean(dim=-1)
        return self.head(z)


def preprocess_raw(Xtr: np.ndarray, Xv: np.ndarray, Xte: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xtr = base.car(Xtr)
    Xv = base.car(Xv)
    Xte = base.car(Xte)
    mu = Xtr.mean(axis=(0, 2), keepdims=True)
    sd = np.maximum(Xtr.std(axis=(0, 2), keepdims=True), 1e-6)
    return ((Xtr - mu) / sd).astype(np.float32), ((Xv - mu) / sd).astype(np.float32), ((Xte - mu) / sd).astype(np.float32)


def predict_prob(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    outs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), 256):
            xb = torch.from_numpy(X[start : start + 256]).to(device)
            outs.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return u.normalize_prob(np.concatenate(outs, axis=0))


def train_one(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xv: np.ndarray,
    yv: np.ndarray,
    dropout: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> Tuple[SimpleEEGConv1D, Dict[str, float], List[Dict[str, float]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SimpleEEGConv1D(Xtr.shape[1], dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.int64)))
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(ds, batch_size=64, shuffle=True, generator=gen)
    best_state = None
    best_score = -1.0
    best_epoch = 0
    wait = 0
    history: List[Dict[str, float]] = []
    for epoch in range(1, 151):
        model.train()
        total_loss = 0.0
        total = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(xb)
            total += len(xb)
        pv = predict_prob(model, Xv, device)
        m = u.evaluate_prob(yv, pv)
        score = u.score_metrics(m)
        history.append({"epoch": epoch, "train_loss": total_loss / max(total, 1), "val_acc": m["val_acc"], "macro_f1": m["macro_f1"], "min_recall": m["min_recall"], "score": score})
        if score > best_score + 1e-10:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= 25:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_score": float(best_score), "best_epoch": float(best_epoch)}, history


def main() -> None:
    out_dir = u.OUTPUT_ROOT / f"simple_1dcnn_baseline_{u.stamp()}"
    out_dir.mkdir(parents=True, exist_ok=False)
    result: Dict[str, object] = {"status": "running", "output_dir": str(out_dir), "errors": []}
    try:
        Xtr_raw, ytr = base.load_xy(u.DATA / "train.h5", True)
        Xv_raw, yv = base.load_xy(u.DATA / "val.h5", True)
        Xte_raw, _ = base.load_xy(u.DATA / "test_x_only.h5", False)
        if Xtr_raw.ndim != 3 or Xtr_raw.shape[1] != 62:
            raise RuntimeError(f"Not raw-like EEG; got shape {Xtr_raw.shape}")
        Xtr, Xv, Xte = preprocess_raw(Xtr_raw, Xv_raw, Xte_raw)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rows: List[Dict[str, object]] = []
        full_history: List[Dict[str, object]] = []
        best = None
        for seed in [2024, 2025, 2026]:
            for dropout in [0.3, 0.5]:
                for wd in [1e-4, 1e-3]:
                    model, meta, hist = train_one(Xtr, ytr, Xv, yv, dropout, wd, seed, device)
                    pv = predict_prob(model, Xv, device)
                    pt = predict_prob(model, Xte, device)
                    m = u.evaluate_prob(yv, pv)
                    rec = {
                        "seed": seed,
                        "dropout": dropout,
                        "weight_decay": wd,
                        **meta,
                        **m,
                        "test_prediction_distribution": {str(k): int(v) for k, v in Counter(pt.argmax(axis=1).tolist()).items()},
                    }
                    rows.append(rec)
                    for h in hist:
                        full_history.append({"seed": seed, "dropout": dropout, "weight_decay": wd, **h})
                    if best is None or (m["val_acc"], m["macro_f1"], u.score_metrics(m)) > (
                        best["metrics"]["val_acc"],
                        best["metrics"]["macro_f1"],
                        best["score"],
                    ):
                        best = {"val_probs": pv, "test_probs": pt, "metrics": m, "score": u.score_metrics(m), "config": rec}
        if best is None:
            raise RuntimeError("No CNN configs completed.")
        np.save(out_dir / "cnn_val_probs.npy", best["val_probs"])
        np.save(out_dir / "cnn_test_probs.npy", best["test_probs"])
        seed_check = u.write_seed_from_prob(out_dir / "cnn_SEED.txt", best["test_probs"])
        u.write_csv(out_dir / "training_log.csv", full_history)
        u.write_csv(out_dir / "grid_summary.csv", sorted(rows, key=lambda r: (r["val_acc"], r["macro_f1"]), reverse=True))
        u.draw_confusion_matrix_png(out_dir / "confusion_matrix.png", best["metrics"]["confusion_matrix"], "1D-CNN Validation Confusion Matrix")
        readme = [
            "# Simple 1D-CNN Baseline",
            "",
            "This is a compact Conv1d-over-time baseline for the raw-like `N x 62 x time` course H5 tensors. It avoids SLEEP-style CRNN complexity and does not use hidden test labels.",
            "",
            f"- Output directory: `{out_dir}`",
            f"- Input shape: `{list(Xtr.shape)}`",
            f"- Device: `{device}`",
            f"- Best config: `{best['config']}`",
            f"- Validation acc: `{best['metrics']['val_acc']:.6f}`",
            f"- Macro-F1: `{best['metrics']['macro_f1']:.6f}`",
            f"- Min recall: `{best['metrics']['min_recall']:.6f}`",
            f"- Confusion matrix: `{best['metrics']['confusion_matrix']}`",
            f"- Test label distribution: `{seed_check['distribution']}`",
            f"- SEED validation: `{seed_check}`",
            "",
            "This baseline is mainly for the poster's neural/deep-learning chain. If it underperforms clean feature fusion, the likely explanation is small target sample size and block-wise distribution shift.",
        ]
        (out_dir / "README_simple_1dcnn_baseline.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
        result.update({"status": "completed", "best_config": best["config"], "metrics": best["metrics"], "seed_validation": seed_check})
    except Exception as exc:
        result["status"] = "failed_partial"
        result["errors"].append({"error": repr(exc), "trace": traceback.format_exc()})
        (out_dir / "README_simple_1dcnn_baseline.md").write_text(
            "# Simple 1D-CNN Baseline\n\nRun failed or was not feasible; see `run_results.json` for traceback.\n", encoding="utf-8"
        )
    finally:
        u.write_json(out_dir / "run_results.json", result)
        print(f"OUTPUT_DIR={out_dir}")


if __name__ == "__main__":
    main()
