#!/usr/bin/env python3
"""PyTorch MLP baseline on the existing clean feature cache."""

from __future__ import annotations

import traceback
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import car_block_feature_refinement as base
import two_day_rescue_utils as u


class CleanFeatureMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(16, hidden_dim // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(16, hidden_dim // 2), 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def standardize(Ftr: np.ndarray, Fv: np.ndarray, Fte: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    mu = Ftr.mean(axis=0, keepdims=True)
    sd = Ftr.std(axis=0, keepdims=True)
    sd = np.maximum(sd, 1e-6)
    return ((Ftr - mu) / sd).astype(np.float32), ((Fv - mu) / sd).astype(np.float32), ((Fte - mu) / sd).astype(np.float32), {
        "train_feature_mean_abs": float(np.mean(np.abs(mu))),
        "train_feature_std_mean": float(np.mean(sd)),
    }


def probs(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    outs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), 512):
            xb = torch.from_numpy(X[start : start + 512]).to(device)
            outs.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return u.normalize_prob(np.concatenate(outs, axis=0))


def train_one(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xv: np.ndarray,
    yv: np.ndarray,
    hidden_dim: int,
    dropout: float,
    weight_decay: float,
    lr: float,
    seed: int,
    device: torch.device,
) -> Tuple[CleanFeatureMLP, Dict[str, float], List[Dict[str, float]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CleanFeatureMLP(Xtr.shape[1], hidden_dim, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.int64)))
    gen = torch.Generator()
    gen.manual_seed(seed)
    batch_size = 64 if len(Xtr) >= 512 else 32 if len(Xtr) >= 128 else 16
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen)
    best_state = None
    best_metric = -1.0
    best_epoch = 0
    wait = 0
    history: List[Dict[str, float]] = []
    for epoch in range(1, 201):
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
        pv = probs(model, Xv, device)
        m = u.evaluate_prob(yv, pv)
        metric = u.score_metrics(m)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total, 1),
            "val_acc": m["val_acc"],
            "macro_f1": m["macro_f1"],
            "min_recall": m["min_recall"],
            "score": metric,
        }
        history.append(row)
        if metric > best_metric + 1e-10:
            best_metric = metric
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= 25:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_score": float(best_metric), "best_epoch": float(best_epoch), "batch_size": float(batch_size), "lr": float(lr)}, history


def main() -> None:
    out_dir = u.OUTPUT_ROOT / f"neural_mlp_baseline_{u.stamp()}"
    out_dir.mkdir(parents=True, exist_ok=False)
    result: Dict[str, object] = {"status": "running", "output_dir": str(out_dir), "errors": []}
    rows: List[Dict[str, object]] = []
    try:
        _, ytr = base.load_xy(u.DATA / "train.h5", True)
        _, yv = base.load_xy(u.DATA / "val.h5", True)
        Ftr, Fv, Fte, feat_meta = u.load_feature_cache("common_average_reference__stats_band_covpca20")
        Xtr, Xv, Xte, scale_meta = standardize(Ftr, Fv, Fte)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        best = None
        full_history: List[Dict[str, object]] = []
        for seed in [2024, 2025, 2026]:
            for wd in [1e-4, 1e-3, 1e-2]:
                for dropout in [0.2, 0.4, 0.6]:
                    for lr in [1e-3, 5e-4]:
                        for hidden in [64, 128, 256]:
                            model, train_meta, hist = train_one(Xtr, ytr, Xv, yv, hidden, dropout, wd, lr, seed, device)
                            pv = probs(model, Xv, device)
                            pt = probs(model, Xte, device)
                            m = u.evaluate_prob(yv, pv)
                            rec = {
                                "seed": seed,
                                "weight_decay": wd,
                                "dropout": dropout,
                                "lr": lr,
                                "hidden_dim": hidden,
                                **train_meta,
                                **m,
                                "test_prediction_distribution": {str(k): int(v) for k, v in Counter(pt.argmax(axis=1).tolist()).items()},
                            }
                            rows.append(rec)
                            for h in hist:
                                full_history.append({"seed": seed, "weight_decay": wd, "dropout": dropout, "lr": lr, "hidden_dim": hidden, **h})
                            if best is None or (m["val_acc"], m["macro_f1"], u.score_metrics(m)) > (
                                best["metrics"]["val_acc"],
                                best["metrics"]["macro_f1"],
                                best["score"],
                            ):
                                best = {"model": model, "val_probs": pv, "test_probs": pt, "metrics": m, "score": u.score_metrics(m), "config": rec}
        if best is None:
            raise RuntimeError("No MLP configurations completed.")
        np.save(out_dir / "mlp_val_probs.npy", best["val_probs"])
        np.save(out_dir / "mlp_test_probs.npy", best["test_probs"])
        seed_check = u.write_seed_from_prob(out_dir / "mlp_SEED.txt", best["test_probs"])
        u.write_csv(out_dir / "training_log.csv", full_history)
        u.write_csv(out_dir / "grid_summary.csv", sorted(rows, key=lambda r: (r["val_acc"], r["macro_f1"]), reverse=True))
        u.draw_confusion_matrix_png(out_dir / "confusion_matrix.png", best["metrics"]["confusion_matrix"], "MLP Validation Confusion Matrix")
        readme = [
            "# Neural MLP Baseline",
            "",
            "This is a real PyTorch neural baseline trained on the existing clean feature cache, not on hidden test labels.",
            "",
            f"- Output directory: `{out_dir}`",
            f"- Feature cache: `{feat_meta['feature_cache']}`",
            f"- Feature dimension: `{Xtr.shape[1]}`",
            f"- Device: `{device}`",
            f"- Best config: `{best['config']}`",
            "- Hyperparameter search: hidden_dim in `[64, 128, 256]`, hidden_dim2=`hidden_dim//2` with a floor of 16, dropout in `[0.2, 0.4, 0.6]`, weight_decay in `[1e-4, 1e-3, 1e-2]`, lr in `[1e-3, 5e-4]`, seeds `[2024, 2025, 2026]`.",
            f"- Validation acc: `{best['metrics']['val_acc']:.6f}`",
            f"- Macro-F1: `{best['metrics']['macro_f1']:.6f}`",
            f"- Min recall: `{best['metrics']['min_recall']:.6f}`",
            f"- Confusion matrix: `{best['metrics']['confusion_matrix']}`",
            f"- Test label distribution: `{seed_check['distribution']}`",
            f"- SEED validation: `{seed_check}`",
            "",
            "## Failure Analysis / Poster Note",
            "",
            "The MLP is retained as a neural baseline and ablation even if it does not beat the best probability fusion. The likely causes are small sample size, block-wise distribution shift, and the fact that clean feature tree/probability-fusion models can be more stable than neural models on this validation split.",
        ]
        (out_dir / "README_neural_mlp_baseline.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
        result.update({"status": "completed", "best": best["config"], "metrics": best["metrics"], "seed_validation": seed_check, "feature_meta": feat_meta, "scale_meta": scale_meta})
    except Exception as exc:
        result["status"] = "failed_partial"
        result["errors"].append({"error": repr(exc), "trace": traceback.format_exc()})
        (out_dir / "README_neural_mlp_baseline.md").write_text(
            "# Neural MLP Baseline\n\nRun failed; see `run_results.json` for traceback.\n", encoding="utf-8"
        )
    finally:
        u.write_json(out_dir / "run_results.json", result)
        print(f"OUTPUT_DIR={out_dir}")


if __name__ == "__main__":
    main()
