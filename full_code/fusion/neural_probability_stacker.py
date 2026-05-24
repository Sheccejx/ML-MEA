#!/usr/bin/env python3
"""Regularized neural stacker over saved candidate probabilities."""

from __future__ import annotations

import json
import traceback
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

import car_block_feature_refinement as base
import two_day_rescue_utils as u


class ProbabilityStacker(nn.Module):
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


def softmax_logits(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for start in range(0, len(X), 512):
            xb = torch.from_numpy(X[start : start + 512].astype(np.float32)).to(device)
            outs.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return u.normalize_prob(np.concatenate(outs, axis=0))


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    hidden_dim: int,
    dropout: float,
    weight_decay: float,
    lr: float,
    label_smoothing: float,
    seed: int,
    device: torch.device,
) -> Tuple[ProbabilityStacker, Dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ProbabilityStacker(X.shape[1], hidden_dim, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    ds = TensorDataset(torch.from_numpy(X[train_idx].astype(np.float32)), torch.from_numpy(y[train_idx].astype(np.int64)))
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(ds, batch_size=96, shuffle=True, generator=gen)
    best_state = None
    best_score = -1.0
    best_epoch = 0
    wait = 0
    for epoch in range(1, 121):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        pv = softmax_logits(model, X[eval_idx], device)
        m = u.evaluate_prob(y[eval_idx], pv)
        score = u.score_metrics(m)
        if score > best_score + 1e-10:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= 12:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_score": float(best_score), "best_epoch": float(best_epoch)}


def fit_oof_and_test(
    Xv: np.ndarray,
    Xt: np.ndarray,
    yv: np.ndarray,
    hidden_dim: int,
    dropout: float,
    weight_decay: float,
    lr: float,
    label_smoothing: float,
    seed: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros((len(yv), 3), dtype=np.float64)
    test_parts = []
    fold_rows: List[Dict[str, object]] = []
    for fold, (tr, ev) in enumerate(skf.split(Xv, yv), 1):
        model, meta = train_model(Xv, yv, tr, ev, hidden_dim, dropout, weight_decay, lr, label_smoothing, seed + fold, device)
        oof[ev] = softmax_logits(model, Xv[ev], device)
        test_parts.append(softmax_logits(model, Xt, device))
        m = u.evaluate_prob(yv, oof, ev)
        fold_rows.append({"fold": fold, "eval_size": int(len(ev)), **meta, **m})
    return u.normalize_prob(oof), u.normalize_prob(np.mean(test_parts, axis=0)), fold_rows


def weighted_reference(name: str, weights: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    val = None
    test = None
    for cand_name, w in weights.items():
        vp = u.V2_OUT / "all_val_probs" / f"{u.safe_name(cand_name)}.npy"
        tp = u.V2_OUT / "all_test_probs" / f"{u.safe_name(cand_name)}.npy"
        if not vp.exists() or not tp.exists():
            raise FileNotFoundError(f"Missing probability for {name}: {cand_name}")
        pv = u.normalize_prob(np.load(vp))
        pt = u.normalize_prob(np.load(tp))
        val = pv * w if val is None else val + pv * w
        test = pt * w if test is None else test + pt * w
    return u.normalize_prob(val), u.normalize_prob(test)


def evaluate_named(name: str, pv: np.ndarray, pt: np.ndarray, yv: np.ndarray, out_dir: Path) -> Dict[str, object]:
    m = u.evaluate_prob(yv, pv)
    seed_check = u.write_seed_from_prob(out_dir / f"{name}_SEED.txt", pt)
    return {
        "candidate_name": name,
        **m,
        **u.split_metrics(yv, pv, u.make_splits(len(yv))),
        "test_prediction_distribution": seed_check["distribution"],
        "SEED_txt_path": str(out_dir / f"{name}_SEED.txt"),
        "seed_validation": seed_check,
    }


def main() -> None:
    out_dir = u.OUTPUT_ROOT / f"neural_probability_stacker_{u.stamp()}"
    out_dir.mkdir(parents=True, exist_ok=False)
    result: Dict[str, object] = {"status": "running", "output_dir": str(out_dir), "errors": []}
    try:
        _, yv = base.load_xy(u.DATA / "val.h5", True)
        pool = u.load_candidate_pool(yv, max_candidates=60)
        if len(pool) < 2:
            raise RuntimeError("Not enough candidate probability files for stacking.")
        registry = []
        for i, (name, pv, pt, meta) in enumerate(pool, 1):
            registry.append(
                {
                    "rank": i,
                    "candidate_name": name,
                    "val_shape": list(pv.shape),
                    "test_shape": list(pt.shape),
                    "val_acc": meta.get("val_acc"),
                    "macro_f1": meta.get("macro_f1"),
                    "min_recall": meta.get("min_recall"),
                    "test_distribution": {str(k): int(v) for k, v in Counter(pt.argmax(axis=1).tolist()).items()},
                    "source": meta.get("source"),
                }
            )
        u.write_csv(out_dir / "candidate_registry.csv", registry)
        Xv = np.concatenate([p[1] for p in pool], axis=1).astype(np.float32)
        Xt = np.concatenate([p[2] for p in pool], axis=1).astype(np.float32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rows: List[Dict[str, object]] = []
        fold_rows: List[Dict[str, object]] = []
        best = None
        for seed in [2024, 2025, 2026]:
            for hidden in [8, 16, 32]:
                for dropout in [0.3, 0.5, 0.7]:
                    for wd in [1e-3, 1e-2]:
                        for lr in [1e-3, 5e-4]:
                            for label_smoothing in [0.0, 0.05, 0.1]:
                                pv, pt, fr = fit_oof_and_test(Xv, Xt, yv, hidden, dropout, wd, lr, label_smoothing, seed, device)
                                m = u.evaluate_prob(yv, pv)
                                rec = {
                                    "seed": seed,
                                    "hidden_dim": hidden,
                                    "dropout": dropout,
                                    "weight_decay": wd,
                                    "lr": lr,
                                    "label_smoothing": label_smoothing,
                                    **m,
                                    **u.split_metrics(yv, pv, u.make_splits(len(yv))),
                                    "test_prediction_distribution": {str(k): int(v) for k, v in Counter(pt.argmax(axis=1).tolist()).items()},
                                }
                                rows.append(rec)
                                for r in fr:
                                    fold_rows.append({"seed": seed, "hidden_dim": hidden, "dropout": dropout, "weight_decay": wd, "lr": lr, "label_smoothing": label_smoothing, **r})
                                objective = rec["combined_split_eval_acc_mean"] + 0.25 * rec["macro_f1"] + 0.10 * rec["min_recall"]
                                if best is None or objective > best["objective"]:
                                    best = {"objective": objective, "val_probs": pv, "test_probs": pt, "metrics": m, "config": rec}
        if best is None:
            raise RuntimeError("No stacker configs completed.")
        np.save(out_dir / "stacker_val_probs.npy", best["val_probs"])
        np.save(out_dir / "stacker_test_probs.npy", best["test_probs"])
        seed_check = u.write_seed_from_prob(out_dir / "stacker_SEED.txt", best["test_probs"])
        sorted_rows = sorted(rows, key=lambda r: (r["combined_split_eval_acc_mean"], r["val_acc"]), reverse=True)
        u.write_csv(out_dir / "stacker_grid_summary.csv", sorted_rows)
        u.write_csv(out_dir / "stacker_training_log.csv", sorted_rows)
        u.write_csv(out_dir / "stacker_fold_log.csv", fold_rows)
        u.write_json(out_dir / "stacker_best_config.json", best["config"])
        v25_json = json.loads((u.V25_OUT / "run_results.json").read_text(encoding="utf-8"))
        old_bal_v, old_bal_t = weighted_reference("old_balanced", v25_json["picks"]["old_balanced"]["weights"])
        reg_v, reg_t = weighted_reference("regularized_fusion", v25_json["picks"]["balanced_plus"]["weights"])
        safe_v, safe_t = weighted_reference("safe_v25", v25_json["picks"]["safest"]["weights"])
        ablations = []
        ablations.append(evaluate_named("old_balanced_alone", old_bal_v, old_bal_t, yv, out_dir))
        ablations.append(evaluate_named("neural_stacker_alone", best["val_probs"], best["test_probs"], yv, out_dir))
        ablations.append(evaluate_named("old_balanced_plus_neural_stacker_avg", u.normalize_prob(0.5 * old_bal_v + 0.5 * best["val_probs"]), u.normalize_prob(0.5 * old_bal_t + 0.5 * best["test_probs"]), yv, out_dir))
        ablations.append(evaluate_named("safe_v25_plus_neural_stacker_avg", u.normalize_prob(0.5 * safe_v + 0.5 * best["val_probs"]), u.normalize_prob(0.5 * safe_t + 0.5 * best["test_probs"]), yv, out_dir))
        ablations.append(evaluate_named("regularized_fusion_plus_neural_stacker_avg", u.normalize_prob(0.5 * reg_v + 0.5 * best["val_probs"]), u.normalize_prob(0.5 * reg_t + 0.5 * best["test_probs"]), yv, out_dir))
        u.write_csv(out_dir / "stacker_ablation_summary.csv", ablations)
        u.draw_confusion_matrix_png(out_dir / "confusion_matrix.png", best["metrics"]["confusion_matrix"], "Neural Stacker OOF Confusion Matrix")
        readme = [
            "# Neural Probability Stacker",
            "",
            "A small PyTorch neural network was trained on concatenated candidate validation probabilities, with 5-fold OOF validation probabilities and averaged test probabilities to reduce direct validation memorization.",
            "",
            f"- Output directory: `{out_dir}`",
            f"- Number of base candidates: `{len(pool)}`",
            f"- Input dimension: `{Xv.shape[1]}`",
            f"- Best config: `{best['config']}`",
            "- Hyperparameter search: hidden_dim in `[8, 16, 32]`, dropout in `[0.3, 0.5, 0.7]`, weight_decay in `[1e-3, 1e-2]`, lr in `[1e-3, 5e-4]`, label_smoothing in `[0.0, 0.05, 0.1]`, seeds `[2024, 2025, 2026]`.",
            f"- Candidate registry: `{out_dir / 'candidate_registry.csv'}`",
            f"- OOF val acc: `{best['metrics']['val_acc']:.6f}`",
            f"- OOF macro-F1: `{best['metrics']['macro_f1']:.6f}`",
            f"- OOF min recall: `{best['metrics']['min_recall']:.6f}`",
            f"- Test distribution: `{seed_check['distribution']}`",
            f"- SEED validation: `{seed_check}`",
            "",
            "## Ablation",
            "",
            u.md_table(ablations, ["candidate_name", "val_acc", "macro_f1", "min_recall", "test_prediction_distribution", "block10_even_odd_eval_acc_mean", "SEED_txt_path"]),
            "",
            "If the neural stacker does not improve over fixed probability fusion, it is still retained as the neural-network ensemble ablation. The OOF design is intentionally conservative because the validation set is also the only labeled target for stacking.",
        ]
        (out_dir / "README_neural_probability_stacker.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
        result.update({"status": "completed", "pool_size": len(pool), "input_dim": Xv.shape[1], "best_config": best["config"], "metrics": best["metrics"], "seed_validation": seed_check, "ablations": ablations})
    except Exception as exc:
        result["status"] = "failed_partial"
        result["errors"].append({"error": repr(exc), "trace": traceback.format_exc()})
        (out_dir / "README_neural_probability_stacker.md").write_text(
            "# Neural Probability Stacker\n\nRun failed; see `run_results.json` for traceback.\n", encoding="utf-8"
        )
    finally:
        u.write_json(out_dir / "run_results.json", result)
        print(f"OUTPUT_DIR={out_dir}")


if __name__ == "__main__":
    main()
