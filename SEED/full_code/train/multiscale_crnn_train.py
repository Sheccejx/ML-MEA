#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SEED MultiScale CRNN experiment.

This file keeps the original notebook/baseline untouched and adds one
standalone experiment adapted from the teammate SLEEP CRNN idea.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "course_project" / "SEED"
OUTPUT_ROOT = ROOT / "outputs_experiments"
NUM_WORKERS = 0


@dataclass
class TrainConfig:
    model: str = "multiscale_crnn"
    epochs: int = 60
    batch_size: int = 32
    lr: float = 5e-4
    weight_decay: float = 3e-4
    patience: int = 15
    dropout: float = 0.5
    hidden_dim: int = 64
    mixup_alpha: float = 0.0
    clip_grad_norm: float = 1.0
    seed: int = 17
    normalize: str = "none"
    use_supplement: bool = False
    supplement_h5: str = ""
    external_ratio: float = 0.3
    source_aware_loss: bool = False
    external_loss_weight: float = 1.0
    pretrain_checkpoint: str = ""
    pretrain_type: str = "none"
    pretrain_epochs: int = 0
    freeze_backbone_epochs: int = 0
    backbone_lr: float = 0.0
    classifier_lr: float = 0.0


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_h5_xy(path: Path, require_y: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    with h5py.File(path, "r") as f:
        X = f["X"][()].astype(np.float32)
        if require_y:
            y = f["y"][()].astype(np.int64)
            return X, y
        return X, None


def infer_seed_shape(train_path: Path) -> Dict[str, object]:
    with h5py.File(train_path, "r") as f:
        x_shape = tuple(int(v) for v in f["X"].shape)
        y = f["y"][()].astype(np.int64)
    labels = np.unique(y)
    expected = np.arange(len(labels))
    if not np.array_equal(labels, expected):
        raise ValueError(f"CrossEntropyLoss expects labels 0..C-1, got labels {labels.tolist()}")
    return {
        "x_shape": x_shape,
        "channels": int(x_shape[1]),
        "time_point": int(x_shape[2]),
        "num_classes": int(len(labels)),
        "labels": labels.astype(int).tolist(),
        "label_distribution": {str(int(k)): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
    }


class H5EEGDataset(Dataset):
    def __init__(
        self,
        h5_path: Path,
        has_y: bool = True,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ):
        X, y = load_h5_xy(h5_path, require_y=has_y)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None
        self.mean = torch.tensor(mean, dtype=torch.float32) if mean is not None else None
        self.std = torch.tensor(std, dtype=torch.float32) if std is not None else None
        if self.y is not None:
            assert len(self.X) == len(self.y), "X and y length mismatch"

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = self.X[idx]
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / (self.std + 1e-6)
        if self.y is None:
            return x
        return x, self.y[idx]


class ExternalSeedLikeDataset(H5EEGDataset):
    def __init__(
        self,
        h5_path: Path,
        expected_channels: int,
        expected_time: int,
        num_classes: int,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ):
        super().__init__(h5_path, has_y=True, mean=mean, std=std)
        if self.X.ndim != 3:
            raise ValueError(f"Supplement X must be 3D, got {tuple(self.X.shape)}")
        if tuple(self.X.shape[1:]) != (expected_channels, expected_time):
            raise ValueError(
                f"Supplement X shape {tuple(self.X.shape)} does not match "
                f"SEED ({expected_channels}, {expected_time})"
            )
        labels = sorted(int(v) for v in torch.unique(self.y).tolist())
        if not set(labels) <= set(range(num_classes)):
            raise ValueError(f"Supplement labels must be within 0..{num_classes - 1}, got {labels}")


class SourceTaggedDataset(Dataset):
    def __init__(self, dataset: Dataset, source_id: int):
        self.dataset = dataset
        self.source_id = int(source_id)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        x, y = self.dataset[idx]
        return x, y, torch.tensor(self.source_id, dtype=torch.long)


class EEGNormalize(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        return (x - mean) / (std + 1e-6)


class EEGNetClassifier(nn.Module):
    def __init__(
        self,
        chans: int,
        time_point: int = 400,
        num_classes: int = 3,
        f1: int = 8,
        d: int = 2,
        pk1: int = 4,
        pk2: int = 8,
        dp: float = 0.5,
        max_norm1: float = 1.0,
        norm: nn.Module = nn.Identity(),
    ):
        super().__init__()
        f2 = f1 * d
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(f1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(f1, d * f1, (chans, 1), groups=f1, bias=False),
            nn.BatchNorm2d(d * f1),
            nn.ELU(),
            nn.AvgPool2d((1, pk1), stride=pk1),
            nn.Dropout(dp),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(d * f1, f2, (1, 16), groups=f2, bias=False, padding=(0, 8)),
            nn.Conv2d(f2, f2, kernel_size=1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pk2), stride=pk2),
            nn.Dropout(dp),
        )
        self._apply_max_norm(self.block2[0], max_norm1)
        self.embed_dim = f2 * ((time_point // pk1) // pk2)
        self.norm = norm
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def _apply_max_norm(self, layer: nn.Module, max_norm: float) -> None:
        for name, param in layer.named_parameters():
            if "weight" in name:
                param.data = torch.renorm(param.data, p=2, dim=0, maxnorm=max_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = x.unsqueeze(dim=1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)


class SEBlock1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.fc(x).view(x.size(0), x.size(1), 1)
        return x * weight


class MultiScaleTemporalCNN(nn.Module):
    def __init__(
        self,
        in_chans: int,
        branch_channels: int = 24,
        out_channels: int = 72,
        kernels: Tuple[int, int, int] = (3, 7, 15),
        dropout: float = 0.5,
    ):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernels:
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(in_chans, branch_channels, kernel_size=k, padding=k // 2, bias=False),
                    nn.BatchNorm1d(branch_channels),
                    nn.GELU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                    nn.Dropout(dropout * 0.5),
                )
            )
        merged_channels = branch_channels * len(kernels)
        self.fuse = nn.Sequential(
            nn.Conv1d(merged_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout * 0.5),
        )
        self.se = SEBlock1D(out_channels, reduction=8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [branch(x) for branch in self.branches]
        x = torch.cat(features, dim=1)
        x = self.fuse(x)
        return self.se(x)


class SEED_MultiScaleCRNN(nn.Module):
    def __init__(
        self,
        chans: int,
        time_point: int = 400,
        num_classes: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.5,
        norm: nn.Module = nn.Identity(),
    ):
        super().__init__()
        self.norm = norm
        self.cnn = MultiScaleTemporalCNN(
            in_chans=chans,
            branch_channels=24,
            out_channels=72,
            kernels=(3, 7, 15),
            dropout=dropout,
        )
        self.lstm = nn.LSTM(
            input_size=72,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        feat = out.mean(dim=1)
        return self.classifier(feat)


def compute_train_mean_std(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=(0, 2), keepdims=False).reshape(X.shape[1], 1).astype(np.float32)
    std = X.std(axis=(0, 2), keepdims=False).reshape(X.shape[1], 1).astype(np.float32)
    return mean, std


def compute_class_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(y.astype(int), minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return weights.astype(np.float32)


def label_distribution(y: np.ndarray) -> Dict[str, int]:
    vals, counts = np.unique(y.astype(int), return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(vals, counts)}


def build_supplement_subset(ds: Dataset, seed_train_n: int, external_ratio: float, seed: int) -> Dataset:
    if external_ratio <= 0:
        raise ValueError("--external-ratio must be > 0 when supplement is enabled")
    if external_ratio >= 1:
        raise ValueError("--external-ratio must be < 1; validation remains official SEED only")
    target_n = int(round(seed_train_n * external_ratio / (1.0 - external_ratio)))
    target_n = max(1, min(len(ds), target_n))
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(ds), size=target_n, replace=False))
    return Subset(ds, idx.tolist())


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float, num_classes: int):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[idx]
    y_onehot = torch.zeros(x.size(0), num_classes, device=x.device)
    y_onehot.scatter_(1, y.view(-1, 1), 1.0)
    mixed_y = lam * y_onehot + (1.0 - lam) * y_onehot[idx]
    return mixed_x, mixed_y


def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor, class_weights: torch.Tensor) -> torch.Tensor:
    log_prob = torch.log_softmax(logits, dim=1)
    return -(soft_targets * log_prob * class_weights.view(1, -1)).sum(dim=1).mean()


def metrics_from_logits(logits: np.ndarray, y_true: np.ndarray, labels: List[int]) -> Dict[str, object]:
    pred = logits.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=labels).astype(int).tolist(),
        "prediction_distribution": {str(int(k)): int(v) for k, v in zip(*np.unique(pred, return_counts=True))},
    }


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    model.eval()
    logits_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    for batch in loader:
        if isinstance(batch, (tuple, list)):
            xb, yb = batch[0], batch[1]
            labels_all.append(yb.cpu().numpy())
        else:
            xb = batch
        xb = xb.to(device)
        logits_all.append(model(xb).detach().cpu().numpy())
    logits = np.concatenate(logits_all, axis=0)
    labels = np.concatenate(labels_all, axis=0) if labels_all else None
    return logits, labels


def write_seed_txt(path: Path, pred: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fp:
        for value in pred.astype(int).tolist():
            fp.write(f"{value}\n")


def write_submission_csv(path: Path, pred: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["label"])
        for value in pred.astype(int).tolist():
            writer.writerow([value])


def build_model(cfg: TrainConfig, shape_info: Dict[str, object]) -> nn.Module:
    norm = EEGNormalize() if cfg.normalize == "sample" else nn.Identity()
    kwargs = {
        "chans": int(shape_info["channels"]),
        "time_point": int(shape_info["time_point"]),
        "num_classes": int(shape_info["num_classes"]),
        "dropout": cfg.dropout,
        "norm": norm,
    }
    if cfg.model == "eegnet":
        return EEGNetClassifier(
            chans=kwargs["chans"],
            time_point=kwargs["time_point"],
            num_classes=kwargs["num_classes"],
            dp=kwargs["dropout"],
            norm=kwargs["norm"],
        )
    if cfg.model == "multiscale_crnn":
        return SEED_MultiScaleCRNN(
            chans=kwargs["chans"],
            time_point=kwargs["time_point"],
            num_classes=kwargs["num_classes"],
            hidden_dim=cfg.hidden_dim,
            dropout=kwargs["dropout"],
            norm=kwargs["norm"],
        )
    raise ValueError(f"Unknown model: {cfg.model}")


def backbone_parameter_names(model: nn.Module) -> List[str]:
    names = []
    for name, _ in model.named_parameters():
        if name.startswith("cnn.") or name.startswith("lstm.") or name.startswith("block"):
            names.append(name)
    return names


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    prefixes = ("cnn.", "lstm.", "block1.", "block2.", "block3.")
    for name, param in model.named_parameters():
        if name.startswith(prefixes):
            param.requires_grad = trainable


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    if cfg.backbone_lr > 0 or cfg.classifier_lr > 0:
        backbone_names = set(backbone_parameter_names(model))
        backbone_params = [p for n, p in model.named_parameters() if n in backbone_names and p.requires_grad]
        head_params = [p for n, p in model.named_parameters() if n not in backbone_names and p.requires_grad]
        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": cfg.backbone_lr if cfg.backbone_lr > 0 else cfg.lr})
        if head_params:
            groups.append({"params": head_params, "lr": cfg.classifier_lr if cfg.classifier_lr > 0 else cfg.lr})
        return torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    return torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.lr, weight_decay=cfg.weight_decay)


def load_pretrain_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> Dict[str, object]:
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw = state.get("model_state", state)
    current = model.state_dict()
    compatible = {
        k: v for k, v in raw.items()
        if k in current and tuple(current[k].shape) == tuple(v.shape)
    }
    model.load_state_dict(compatible, strict=False)
    return {"path": str(checkpoint_path), "loaded_tensors": len(compatible), "available_tensors": len(raw)}


def masked_input(x: torch.Tensor, mask_ratio: float = 0.25, span: int = 40) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = torch.zeros_like(x, dtype=torch.bool)
    total = x.size(2)
    spans = max(1, int(round((total * mask_ratio) / max(1, span))))
    for i in range(x.size(0)):
        for _ in range(spans):
            start = int(torch.randint(0, max(1, total - span + 1), (1,), device=x.device).item())
            mask[i, :, start:start + span] = True
    corrupted = x.clone()
    corrupted[mask] = 0.0
    return corrupted, mask


class MaskedReconstructionModel(nn.Module):
    def __init__(self, classifier_model: nn.Module, channels: int):
        super().__init__()
        self.norm = classifier_model.norm if hasattr(classifier_model, "norm") else nn.Identity()
        self.cnn = classifier_model.cnn
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(72, 72, kernel_size=4, stride=4),
            nn.GELU(),
            nn.Conv1d(72, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        z = self.cnn(x)
        rec = self.decoder(z)
        if rec.size(2) != x.size(2):
            rec = nn.functional.interpolate(rec, size=x.size(2), mode="linear", align_corners=False)
        return rec


def run_external_pretrain(
    model: nn.Module,
    cfg: TrainConfig,
    shape_info: Dict[str, object],
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
    device: torch.device,
    out_dir: Path,
) -> Optional[Dict[str, object]]:
    if cfg.pretrain_type == "none" or cfg.pretrain_epochs <= 0:
        return None
    if not cfg.supplement_h5:
        raise ValueError("--pretrain-type requires --supplement-h5")
    ext_ds = ExternalSeedLikeDataset(
        Path(cfg.supplement_h5),
        expected_channels=int(shape_info["channels"]),
        expected_time=int(shape_info["time_point"]),
        num_classes=int(shape_info["num_classes"]),
        mean=mean,
        std=std,
    )
    loader = DataLoader(ext_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=NUM_WORKERS)
    pretrain_dir = out_dir / "pretrain"
    pretrain_dir.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, float]] = []
    if cfg.pretrain_type in {"masked_reconstruction", "denoising_reconstruction"}:
        recon = MaskedReconstructionModel(model, int(shape_info["channels"])).to(device)
        opt = torch.optim.AdamW(recon.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        for epoch in range(1, cfg.pretrain_epochs + 1):
            recon.train()
            loss_sum = 0.0
            n = 0
            for xb, _ in loader:
                xb = xb.to(device)
                if cfg.pretrain_type == "masked_reconstruction":
                    inp, mask = masked_input(xb)
                    target_mask = mask
                else:
                    inp = xb + 0.15 * torch.randn_like(xb)
                    target_mask = torch.ones_like(xb, dtype=torch.bool)
                opt.zero_grad(set_to_none=True)
                rec = recon(inp)
                loss = nn.functional.mse_loss(rec[target_mask], xb[target_mask])
                loss.backward()
                nn.utils.clip_grad_norm_(recon.parameters(), cfg.clip_grad_norm)
                opt.step()
                loss_sum += float(loss.item()) * int(xb.size(0))
                n += int(xb.size(0))
            history.append({"epoch": epoch, "loss": loss_sum / max(1, n)})
            print(f"pretrain {cfg.pretrain_type} epoch {epoch}/{cfg.pretrain_epochs} loss={history[-1]['loss']:.5f}")
        ckpt = pretrain_dir / f"{cfg.pretrain_type}_checkpoint.pth"
        torch.save({"model_state": model.state_dict(), "config": asdict(cfg), "history": history}, ckpt)
        return {"type": cfg.pretrain_type, "epochs": cfg.pretrain_epochs, "checkpoint": str(ckpt), "history": history}
    if cfg.pretrain_type == "supervised":
        num_classes = int(shape_info["num_classes"])
        weights = torch.tensor(compute_class_weights(ext_ds.y.numpy(), num_classes), dtype=torch.float32, device=device)
        opt = build_optimizer(model, cfg)
        criterion = nn.CrossEntropyLoss(weight=weights)
        for epoch in range(1, cfg.pretrain_epochs + 1):
            model.train()
            loss_sum = 0.0
            n = 0
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad(set_to_none=True)
                loss = criterion(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
                opt.step()
                loss_sum += float(loss.item()) * int(yb.size(0))
                n += int(yb.size(0))
            history.append({"epoch": epoch, "loss": loss_sum / max(1, n)})
            print(f"pretrain supervised epoch {epoch}/{cfg.pretrain_epochs} loss={history[-1]['loss']:.5f}")
        ckpt = pretrain_dir / "supervised_pretrain_checkpoint.pth"
        torch.save({"model_state": model.state_dict(), "config": asdict(cfg), "history": history}, ckpt)
        return {"type": cfg.pretrain_type, "epochs": cfg.pretrain_epochs, "checkpoint": str(ckpt), "history": history}
    raise ValueError(f"Unknown --pretrain-type: {cfg.pretrain_type}")


def generate_prediction_txt(
    model: nn.Module,
    model_path: Path,
    test_loader: DataLoader,
    save_path: Path,
    device: torch.device,
) -> np.ndarray:
    state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    logits, _ = predict_logits(model, test_loader, device)
    pred = logits.argmax(axis=1).astype(np.int64)
    write_seed_txt(save_path, pred)
    return pred


def train_one(cfg: TrainConfig, data_dir: Path, out_dir: Path) -> Dict[str, object]:
    set_seed(cfg.seed)
    train_path = data_dir / "train.h5"
    val_path = data_dir / "val.h5"
    test_path = data_dir / "test_x_only.h5"
    shape_info = infer_seed_shape(train_path)
    X_train, y_train = load_h5_xy(train_path, require_y=True)
    assert y_train is not None

    mean = std = None
    if cfg.normalize == "train_channel":
        mean, std = compute_train_mean_std(X_train)

    train_ds = H5EEGDataset(train_path, has_y=True, mean=mean, std=std)
    seed_train_ds = train_ds
    supplement_info: Optional[Dict[str, object]] = None
    combined_y = y_train
    if cfg.use_supplement:
        if not cfg.supplement_h5:
            raise ValueError("--use-supplement requires --supplement-h5")
        supplement_path = Path(cfg.supplement_h5)
        ext_ds = ExternalSeedLikeDataset(
            supplement_path,
            expected_channels=int(shape_info["channels"]),
            expected_time=int(shape_info["time_point"]),
            num_classes=int(shape_info["num_classes"]),
            mean=mean,
            std=std,
        )
        ext_subset = build_supplement_subset(ext_ds, len(seed_train_ds), cfg.external_ratio, cfg.seed)
        ext_indices = np.asarray(ext_subset.indices, dtype=np.int64) if isinstance(ext_subset, Subset) else np.arange(len(ext_subset))
        ext_y = ext_ds.y.numpy()[ext_indices]
        if cfg.source_aware_loss:
            train_ds = ConcatDataset([SourceTaggedDataset(seed_train_ds, 0), SourceTaggedDataset(ext_subset, 1)])
        else:
            train_ds = ConcatDataset([seed_train_ds, ext_subset])
        combined_y = np.concatenate([y_train, ext_y], axis=0)
        supplement_info = {
            "path": str(supplement_path),
            "total_external_samples": int(len(ext_ds)),
            "selected_external_samples": int(len(ext_subset)),
            "external_ratio_requested": float(cfg.external_ratio),
            "external_ratio_actual": float(len(ext_subset) / max(1, len(train_ds))),
            "source_aware_loss": bool(cfg.source_aware_loss),
            "external_loss_weight": float(cfg.external_loss_weight),
            "external_label_distribution": label_distribution(ext_y),
        }
    val_ds = H5EEGDataset(val_path, has_y=True, mean=mean, std=std)
    test_ds = H5EEGDataset(test_path, has_y=False, mean=mean, std=std)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=NUM_WORKERS)
    eval_train_loader = DataLoader(train_ds, batch_size=cfg.batch_size * 4, shuffle=False, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size * 4, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size * 4, shuffle=False, num_workers=NUM_WORKERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, shape_info).to(device)
    pretrain_info: Optional[Dict[str, object]] = None
    if cfg.pretrain_checkpoint:
        pretrain_info = load_pretrain_checkpoint(model, Path(cfg.pretrain_checkpoint), device)
        print(f"loaded pretrain checkpoint: {pretrain_info}")
    elif cfg.pretrain_type != "none" and cfg.pretrain_epochs > 0:
        pretrain_info = run_external_pretrain(model, cfg, shape_info, mean, std, device, out_dir)
    num_classes = int(shape_info["num_classes"])
    labels = list(range(num_classes))
    class_weights_np = compute_class_weights(combined_y, num_classes)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion_none = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
    if cfg.freeze_backbone_epochs > 0:
        set_backbone_trainable(model, False)
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
    )

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_acc_path = ckpt_dir / "best_acc_model.pth"
    best_loss_path = ckpt_dir / "best_loss_model.pth"
    last_path = ckpt_dir / "last_model.pth"

    history: List[Dict[str, object]] = []
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_acc_epoch = 0
    best_loss_epoch = 0
    best_acc_state = None
    bad_epochs = 0

    print(f"device = {device}")
    print(f"model = {cfg.model}")
    print(f"shape = channels {shape_info['channels']} | time_point {shape_info['time_point']} | classes {shape_info['num_classes']}")
    print(f"labels = {shape_info['labels']}")
    print(f"SEED train samples = {len(seed_train_ds)} | distribution = {label_distribution(y_train)}")
    if supplement_info is not None:
        print(f"external train samples = {supplement_info['selected_external_samples']} of {supplement_info['total_external_samples']}")
        print(f"external label distribution = {supplement_info['external_label_distribution']}")
    print(f"combined train samples = {len(train_ds)} | distribution = {label_distribution(combined_y)}")
    print(f"class_weights = {class_weights_np.tolist()}")

    for epoch in range(1, cfg.epochs + 1):
        if cfg.freeze_backbone_epochs > 0 and epoch == cfg.freeze_backbone_epochs + 1:
            set_backbone_trainable(model, True)
            optimizer = build_optimizer(model, cfg)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=4,
            )
        model.train()
        train_loss_sum = 0.0
        train_num = 0
        for batch in train_loader:
            if isinstance(batch, (tuple, list)) and len(batch) >= 3:
                xb, yb, source_id = batch[0], batch[1], batch[2]
                source_id = source_id.to(device)
            else:
                xb, yb = batch[0], batch[1]
                source_id = None
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            if cfg.mixup_alpha > 0:
                xb, soft_y = mixup_batch(xb, yb, cfg.mixup_alpha, num_classes)
                logits = model(xb)
                loss = soft_cross_entropy(logits, soft_y, class_weights)
            else:
                logits = model(xb)
                if cfg.source_aware_loss and source_id is not None:
                    per_sample = criterion_none(logits, yb)
                    weights = torch.where(
                        source_id == 1,
                        torch.full_like(per_sample, float(cfg.external_loss_weight)),
                        torch.ones_like(per_sample),
                    )
                    loss = (per_sample * weights).sum() / weights.sum().clamp_min(1e-6)
                else:
                    loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            optimizer.step()
            if hasattr(model, "_apply_max_norm"):
                model._apply_max_norm(model.block2[0], 1)
            batch_size = int(yb.size(0))
            train_loss_sum += float(loss.item()) * batch_size
            train_num += batch_size

        train_logits, train_y = predict_logits(model, eval_train_loader, device)
        val_logits, val_y = predict_logits(model, val_loader, device)
        assert train_y is not None and val_y is not None
        train_metrics = metrics_from_logits(train_logits, train_y, labels)
        val_metrics = metrics_from_logits(val_logits, val_y, labels)

        val_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                val_loss_sum += float(criterion(model(xb), yb).item()) * int(yb.size(0))
        epoch_train_loss = train_loss_sum / max(1, train_num)
        epoch_val_loss = val_loss_sum / max(1, len(val_ds))
        scheduler.step(epoch_val_loss)

        improved_acc = val_metrics["accuracy"] > best_val_acc
        improved_loss = epoch_val_loss < best_val_loss
        if improved_acc:
            best_val_acc = float(val_metrics["accuracy"])
            best_acc_epoch = epoch
            best_acc_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
            torch.save(
                {
                    "model_state": best_acc_state,
                    "config": asdict(cfg),
                    "shape_info": shape_info,
                    "class_weights": class_weights_np.tolist(),
                    "mean": mean,
                    "std": std,
                    "best_epoch": best_acc_epoch,
                    "best_val_acc": best_val_acc,
                },
                best_acc_path,
            )
        else:
            bad_epochs += 1
        if improved_loss:
            best_val_loss = float(epoch_val_loss)
            best_loss_epoch = epoch
            torch.save(
                {
                    "model_state": copy.deepcopy(model.state_dict()),
                    "config": asdict(cfg),
                    "shape_info": shape_info,
                    "class_weights": class_weights_np.tolist(),
                    "mean": mean,
                    "std": std,
                    "best_epoch": best_loss_epoch,
                    "best_val_loss": best_val_loss,
                },
                best_loss_path,
            )

        row = {
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": epoch_val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_prediction_distribution": val_metrics["prediction_distribution"],
            "lr": float(optimizer.param_groups[0]["lr"]),
            "lr_groups": [float(g["lr"]) for g in optimizer.param_groups],
            "best_val_acc": best_val_acc,
            "best_acc_epoch": best_acc_epoch,
            "improved_acc": bool(improved_acc),
            "improved_loss": bool(improved_loss),
        }
        history.append(row)
        print(
            f"{'*' if improved_acc else ' '} Epoch [{epoch:02d}/{cfg.epochs}] | "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Macro-F1: {val_metrics['macro_f1']:.4f} | best={best_val_acc:.4f}@{best_acc_epoch}"
        )
        if bad_epochs >= cfg.patience:
            print(f"early stopping at epoch {epoch}, best val acc {best_val_acc:.4f} at epoch {best_acc_epoch}")
            break

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(cfg),
            "shape_info": shape_info,
            "class_weights": class_weights_np.tolist(),
            "mean": mean,
            "std": std,
            "last_epoch": history[-1]["epoch"] if history else 0,
        },
        last_path,
    )
    (out_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    if best_acc_state is None:
        best_acc_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_acc_state)
    val_logits, val_y = predict_logits(model, val_loader, device)
    train_logits, train_y = predict_logits(model, eval_train_loader, device)
    assert val_y is not None and train_y is not None
    best_train_metrics = metrics_from_logits(train_logits, train_y, labels)
    best_val_metrics = metrics_from_logits(val_logits, val_y, labels)

    test_pred = generate_prediction_txt(model, best_acc_path, test_loader, out_dir / "SEED.txt", device)
    write_submission_csv(out_dir / "submission.csv", test_pred)
    test_logits, _ = predict_logits(model, test_loader, device)
    np.save(out_dir / "test_logits.npy", test_logits.astype(np.float32))

    final_val_acc = float(history[-1]["val_accuracy"]) if history else 0.0
    final_val_f1 = float(history[-1]["val_macro_f1"]) if history else 0.0
    results = {
        "output_dir": str(out_dir),
        "model": cfg.model,
        "shape_info": shape_info,
        "supplement_info": supplement_info,
        "pretrain_info": pretrain_info,
        "class_weights": class_weights_np.tolist(),
        "best_acc_checkpoint": str(best_acc_path),
        "best_loss_checkpoint": str(best_loss_path),
        "last_checkpoint": str(last_path),
        "best_epoch": int(best_acc_epoch),
        "best_val_acc": float(best_val_acc),
        "best_val_macro_f1": float(best_val_metrics["macro_f1"]),
        "final_val_acc": final_val_acc,
        "final_val_macro_f1": final_val_f1,
        "train_metrics_at_best": best_train_metrics,
        "validation_metrics_at_best": best_val_metrics,
        "test_prediction_distribution": {str(int(k)): int(v) for k, v in zip(*np.unique(test_pred, return_counts=True))},
        "seed_txt": str(out_dir / "SEED.txt"),
        "submission_csv": str(out_dir / "submission.csv"),
    }
    (out_dir / "run_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def write_readme_note(out_dir: Path, cfg: TrainConfig, results: Dict[str, object]) -> None:
    text = f"""# SEED MultiScale CRNN Experiment

尝试引入队友 SLEEP 分支中的多尺度 CRNN 思路。SLEEP 任务本身是睡眠分期，标签体系和 SEED 情绪分类不一致，所以没有直接混用 SLEEP 数据，只借鉴了模型结构和训练策略。具体改动是：在 SEED 输入上加入多尺度 temporal CNN，用不同 kernel size 提取短/中/长时间尺度特征，然后加入 SE attention 对特征通道重新加权，最后接 BiLSTM 做时序建模。

结果：
best val acc = {results['best_val_acc']:.4f}
best epoch = {results['best_epoch']}
final val acc = {results['final_val_acc']:.4f}
macro-F1 = {results['best_val_macro_f1']:.4f}

观察：
这个版本保留了原始 SEED 的 `0/1/2` 标签和官方 train/val/test 划分，只在模型和训练策略上做增量实验。class weights 只从训练集标签统计得到；当前官方训练集本身是均衡的，所以权重基本为 1。

下一步：
如果完整 CRNN 的验证集表现不稳定，可以先把 `--model eegnet` 作为同训练策略 baseline 跑一遍，再比较是否是 BiLSTM 过拟合；Mixup 也建议最后再打开，例如 `--mixup-alpha 0.1`。
"""
    (out_dir / "README_experiment.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["multiscale_crnn", "eegnet"], default="multiscale_crnn")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--normalize", choices=["none", "sample", "train_channel"], default="none")
    parser.add_argument("--use-supplement", action="store_true", help="Append supervised SEED-like external windows to train only.")
    parser.add_argument("--supplement-h5", type=Path, default=None, help="Path to supplement_seed_like.h5 with X/y and labels 0/1/2.")
    parser.add_argument("--external-ratio", type=float, default=0.3, help="Approximate fraction of external samples in the training set.")
    parser.add_argument("--source-aware-loss", action="store_true", help="Down/up-weight external samples in mixed supervised batches.")
    parser.add_argument("--external-loss-weight", type=float, default=1.0)
    parser.add_argument("--pretrain-checkpoint", type=Path, default=None)
    parser.add_argument("--pretrain-type", choices=["none", "masked_reconstruction", "denoising_reconstruction", "supervised"], default="none")
    parser.add_argument("--pretrain-epochs", type=int, default=0)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0)
    parser.add_argument("--backbone-lr", type=float, default=0.0)
    parser.add_argument("--classifier-lr", type=float, default=0.0)
    parser.add_argument("--pretrain-external", action="store_true", help="Reserved for self-supervised external pretraining; not used for supervised supplement.")
    parser.add_argument("--finetune-seed", action="store_true", help="Reserved for pretrain + SEED fine-tune workflow.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        dropout=args.dropout,
        hidden_dim=args.hidden_dim,
        mixup_alpha=args.mixup_alpha,
        seed=args.seed,
        normalize=args.normalize,
        use_supplement=args.use_supplement,
        supplement_h5=str(args.supplement_h5) if args.supplement_h5 else "",
        external_ratio=args.external_ratio,
        source_aware_loss=args.source_aware_loss,
        external_loss_weight=args.external_loss_weight,
        pretrain_checkpoint=str(args.pretrain_checkpoint) if args.pretrain_checkpoint else "",
        pretrain_type=args.pretrain_type,
        pretrain_epochs=args.pretrain_epochs,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        backbone_lr=args.backbone_lr,
        classifier_lr=args.classifier_lr,
    )
    if args.pretrain_external or args.finetune_seed:
        print("--pretrain-external/--finetune-seed are placeholders in this supervised script; use supplement only when labels are safe.")
    run_prefix = args.run_name.strip() if args.run_name else f"seed_{cfg.model}"
    out_dir = args.output_root / f"{run_prefix}_{now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
    results = train_one(cfg, args.data_dir, out_dir)
    write_readme_note(out_dir, cfg, results)

    print("\n" + "-" * 40)
    print(f"Best Val Accuracy: {results['best_val_acc']:.4f} at epoch {results['best_epoch']}")
    print(f"Final Val Accuracy: {results['final_val_acc']:.4f}")
    print(f"Best Macro-F1: {results['best_val_macro_f1']:.4f}")
    print(f"Best model: {results['best_acc_checkpoint']}")
    print(f"Saved predictions: {results['seed_txt']}")


if __name__ == "__main__":
    main()
