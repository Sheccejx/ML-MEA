from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    confusion: list[list[int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a fusion EEGNet model from server_fusion_training/data.")
    parser.add_argument("--pretrain-h5", type=Path, default=DATA_DIR / "pretrain.h5")
    parser.add_argument("--train-h5", type=Path, default=DATA_DIR / "train.h5")
    parser.add_argument("--valid-h5", type=Path, default=DATA_DIR / "valid.h5")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "runs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--min-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.04)
    parser.add_argument("--dropout", type=float, default=0.28)
    parser.add_argument("--mixup-alpha", type=float, default=0.15)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument(
        "--model-size",
        choices=("base", "large", "xlarge"),
        default="xlarge",
        help="xlarge is the default pretraining target for the 8 GB RTX 4060.",
    )
    parser.add_argument(
        "--valid-tta-shifts",
        type=str,
        default="0,4,-4",
        help="Comma-separated circular time shifts averaged during validation. Use 0 to disable TTA.",
    )
    parser.add_argument("--pretrain-epochs", type=int, default=160)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--pretrain-weight-decay", type=float, default=1e-4)
    parser.add_argument("--pretrain-mask-ratio", type=float, default=0.35)
    parser.add_argument("--pretrain-channel-mask-prob", type=float, default=0.1)
    parser.add_argument("--pretrain-time-blocks", type=int, default=6)
    parser.add_argument("--pretrain-time-block-frac", type=float, default=0.04)
    parser.add_argument("--pretrain-input-noise", type=float, default=0.015)
    parser.add_argument("--pretrained-temporal", type=Path, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--compile", action="store_true", help="Try torch.compile on CUDA.")
    parser.add_argument("--smoke-test", action="store_true", help="Use small subsets and two epochs.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


class H5EEGDataset(Dataset):
    def __init__(self, path: Path, normalize_window: bool = True) -> None:
        self.path = Path(path)
        self.normalize_window = normalize_window
        self._handle: h5py.File | None = None
        with h5py.File(self.path, "r") as handle:
            self.length = int(len(handle["y"]))
            self.sample_shape = tuple(int(value) for value in handle["X"].shape[1:])
            if self.length != len(handle["X"]):
                raise ValueError(f"X/y length mismatch in {self.path}.")

    def __len__(self) -> int:
        return self.length

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def _file(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        handle = self._file()
        x = torch.from_numpy(np.asarray(handle["X"][index], dtype=np.float32))
        y = torch.tensor(int(handle["y"][index]), dtype=torch.long)
        if self.normalize_window:
            x = torch.nan_to_num(x)
            x = x - x.mean(dim=-1, keepdim=True)
            x = x / x.std(dim=-1, keepdim=True).clamp_min(1e-4)
            x = x.clamp(-8.0, 8.0)
        return x, y


def read_h5_label_counts(path: Path, num_classes: int) -> torch.Tensor:
    with h5py.File(path, "r") as handle:
        labels = np.asarray(handle["y"], dtype=np.int64)
    return torch.tensor(np.bincount(labels, minlength=num_classes), dtype=torch.float32)


class ConvNormAct1d(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )


class SepResidual1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float, pool: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AvgPool1d(pool) if pool > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.pool(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.pool(x)
        return F.gelu(x + residual)


class AttentiveStatsPool1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(32, channels // 2)
        self.attention = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.attention(x).softmax(dim=-1)
        mean = (x * weights).sum(dim=-1)
        variance = ((x - mean.unsqueeze(-1)).pow(2) * weights).sum(dim=-1).clamp_min(1e-5)
        return torch.cat([mean, variance.sqrt()], dim=1)


class SequenceContext1d(nn.Module):
    def __init__(self, channels: int, heads: int, dropout: float, expansion: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(channels)
        self.feed_forward = nn.Sequential(
            nn.Linear(channels, channels * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * expansion, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.transpose(1, 2)
        normed = self.norm1(tokens)
        attended, _ = self.attention(normed, normed, normed, need_weights=False)
        tokens = tokens + self.attention_dropout(attended)
        tokens = tokens + self.feed_forward(self.norm2(tokens))
        return tokens.transpose(1, 2)


class EEGNetTemporalBranch(nn.Module):
    def __init__(
        self,
        chans: int,
        dropout: float,
        embedding_dim: int,
        temporal_width: int = 24,
        model_channels: int = 192,
        context_layers: int = 0,
        context_heads: int = 4,
    ) -> None:
        super().__init__()
        self.temporal_kernels = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, temporal_width, kernel_size=(1, kernel), padding="same", bias=False),
                    nn.BatchNorm2d(temporal_width),
                )
                for kernel in (15, 31, 63)
            ]
        )
        temporal_channels = temporal_width * len(self.temporal_kernels)
        spatial_channels = temporal_channels * 2
        self.spatial = nn.Sequential(
            nn.Conv2d(
                temporal_channels,
                spatial_channels,
                kernel_size=(chans, 1),
                groups=temporal_channels,
                bias=False,
            ),
            nn.BatchNorm2d(spatial_channels),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.mix = ConvNormAct1d(spatial_channels, model_channels, kernel_size=1, dropout=dropout)
        self.blocks = nn.Sequential(
            SepResidual1d(model_channels, kernel_size=15, dropout=dropout, pool=2),
            SepResidual1d(model_channels, kernel_size=11, dropout=dropout, pool=2),
            SepResidual1d(model_channels, kernel_size=7, dropout=dropout),
        )
        self.context = nn.Sequential(
            *[SequenceContext1d(model_channels, context_heads, dropout) for _ in range(context_layers)]
        )
        self.feature_channels = model_channels
        self.pool = AttentiveStatsPool1d(model_channels)
        self.project = nn.Sequential(
            nn.LayerNorm(model_channels * 2),
            nn.Linear(model_channels * 2, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = torch.cat([branch(x) for branch in self.temporal_kernels], dim=1)
        x = self.spatial(x).squeeze(2)
        x = self.mix(x)
        x = self.blocks(x)
        return self.context(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.pool(self.encode_features(x)))


class SpectralBranch(nn.Module):
    def __init__(
        self,
        chans: int,
        dropout: float,
        embedding_dim: int,
        model_channels: int = 160,
        context_layers: int = 0,
        context_heads: int = 4,
    ) -> None:
        super().__init__()
        stem_channels = max(128, model_channels // 2)
        self.stem = nn.Sequential(
            ConvNormAct1d(chans, stem_channels, kernel_size=5, dropout=dropout),
            ConvNormAct1d(stem_channels, model_channels, kernel_size=3, dropout=dropout),
        )
        self.blocks = nn.Sequential(
            SepResidual1d(model_channels, kernel_size=7, dropout=dropout, pool=2),
            SepResidual1d(model_channels, kernel_size=5, dropout=dropout),
        )
        self.context = nn.Sequential(
            *[SequenceContext1d(model_channels, context_heads, dropout) for _ in range(context_layers)]
        )
        self.pool = AttentiveStatsPool1d(model_channels)
        self.project = nn.Sequential(
            nn.LayerNorm(model_channels * 2),
            nn.Linear(model_channels * 2, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(x, dim=-1)
        power = torch.log1p(spectrum.abs().pow(2))[..., 1:]
        power = power - power.mean(dim=-1, keepdim=True)
        power = power / power.std(dim=-1, keepdim=True).clamp_min(1e-4)
        power = self.stem(power)
        power = self.blocks(power)
        power = self.context(power)
        return self.project(self.pool(power))


class ConnectivityBranch(nn.Module):
    def __init__(
        self,
        chans: int,
        dropout: float,
        embedding_dim: int,
        hidden_dim: int = 320,
        deep: bool = False,
    ) -> None:
        super().__init__()
        upper = torch.triu_indices(chans, chans)
        self.register_buffer("upper_row", upper[0], persistent=False)
        self.register_buffer("upper_col", upper[1], persistent=False)
        feature_dim = int(upper.shape[1])
        layers: list[nn.Module] = [
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        if deep:
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        layers.extend(
            [
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            ]
        )
        self.project = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=-1, keepdim=True)
        x = x / x.std(dim=-1, keepdim=True).clamp_min(1e-4)
        covariance = torch.matmul(x, x.transpose(1, 2)) / max(1, x.shape[-1] - 1)
        upper = covariance[:, self.upper_row, self.upper_col]
        return self.project(upper)


class GatedFusion(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_branches: int,
        dropout: float,
        num_classes: int,
        head_dim: int = 384,
        branch_layers: int = 0,
        branch_heads: int = 4,
        deep_head: bool = False,
    ) -> None:
        super().__init__()
        self.branch_context = nn.Sequential(
            *[
                nn.TransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=branch_heads,
                    dim_feedforward=embedding_dim * 3,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(branch_layers)
            ]
        )
        self.branch_gate = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, 1),
        )
        head_layers: list[nn.Module] = [
            nn.LayerNorm(embedding_dim * 2),
            nn.Linear(embedding_dim * 2, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        if deep_head:
            head_layers.extend(
                [
                    nn.Linear(head_dim, head_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            head_dim = head_dim // 2
        head_layers.append(nn.Linear(head_dim, num_classes))
        self.head = nn.Sequential(*head_layers)
        self.num_branches = num_branches

    def forward(self, branches: Iterable[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.stack(list(branches), dim=1)
        if stacked.shape[1] != self.num_branches:
            raise ValueError(f"Expected {self.num_branches} fusion branches, received {stacked.shape[1]}.")
        stacked = self.branch_context(stacked)
        weights = self.branch_gate(stacked).softmax(dim=1)
        weighted = (stacked * weights).sum(dim=1)
        branch_max = stacked.max(dim=1).values
        return self.head(torch.cat([weighted, branch_max], dim=1)), weights.squeeze(-1)


class FusionEEGNet(nn.Module):
    def __init__(self, chans: int, num_classes: int, dropout: float = 0.28, model_size: str = "large") -> None:
        super().__init__()
        if model_size == "base":
            embedding_dim = 256
            self.temporal = EEGNetTemporalBranch(chans, dropout, embedding_dim)
            self.spectral = SpectralBranch(chans, dropout, embedding_dim)
            self.connectivity = ConnectivityBranch(chans, dropout, embedding_dim)
            self.fusion = GatedFusion(
                embedding_dim,
                num_branches=3,
                dropout=dropout,
                num_classes=num_classes,
            )
        elif model_size == "large":
            embedding_dim = 384
            self.temporal = EEGNetTemporalBranch(
                chans,
                dropout,
                embedding_dim,
                temporal_width=40,
                model_channels=320,
                context_layers=3,
                context_heads=8,
            )
            self.spectral = SpectralBranch(
                chans,
                dropout,
                embedding_dim,
                model_channels=256,
                context_layers=2,
                context_heads=8,
            )
            self.connectivity = ConnectivityBranch(
                chans,
                dropout,
                embedding_dim,
                hidden_dim=768,
                deep=True,
            )
            self.fusion = GatedFusion(
                embedding_dim,
                num_branches=3,
                dropout=dropout,
                num_classes=num_classes,
                head_dim=768,
                branch_layers=2,
                branch_heads=8,
                deep_head=True,
            )
        elif model_size == "xlarge":
            embedding_dim = 512
            self.temporal = EEGNetTemporalBranch(
                chans,
                dropout,
                embedding_dim,
                temporal_width=56,
                model_channels=448,
                context_layers=4,
                context_heads=8,
            )
            self.spectral = SpectralBranch(
                chans,
                dropout,
                embedding_dim,
                model_channels=384,
                context_layers=3,
                context_heads=8,
            )
            self.connectivity = ConnectivityBranch(
                chans,
                dropout,
                embedding_dim,
                hidden_dim=1024,
                deep=True,
            )
            self.fusion = GatedFusion(
                embedding_dim,
                num_branches=3,
                dropout=dropout,
                num_classes=num_classes,
                head_dim=1024,
                branch_layers=3,
                branch_heads=8,
                deep_head=True,
            )
        else:
            raise ValueError(f"Unsupported model size: {model_size}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.fusion([self.temporal(x), self.spectral(x), self.connectivity(x)])
        return logits


class MaskedTemporalPretrainer(nn.Module):
    def __init__(self, temporal: EEGNetTemporalBranch, chans: int, time_points: int, dropout: float) -> None:
        super().__init__()
        self.temporal = temporal
        self.time_points = time_points
        hidden = temporal.feature_channels
        self.reconstruction_head = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden // 2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Conv1d(hidden // 2, chans, kernel_size=15, padding=7),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.temporal.encode_features(x)
        features = F.interpolate(features, size=self.time_points, mode="linear", align_corners=False)
        return self.reconstruction_head(features)


def make_pretrain_mask(
    x: torch.Tensor,
    mask_ratio: float,
    channel_mask_prob: float,
    time_blocks: int,
    time_block_frac: float,
) -> torch.Tensor:
    batch, chans, time_points = x.shape
    mask = torch.rand((batch, chans, time_points), device=x.device) < mask_ratio
    if channel_mask_prob > 0:
        mask = mask | (torch.rand((batch, chans, 1), device=x.device) < channel_mask_prob)
    block_len = max(1, min(time_points, int(round(time_points * time_block_frac))))
    if block_len < time_points:
        for _ in range(time_blocks):
            starts = torch.randint(0, time_points - block_len + 1, (batch,), device=x.device)
            for row, start in enumerate(starts.tolist()):
                mask[row, :, start : start + block_len] = True
    return mask


def masked_mse(reconstruction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.float()
    squared = (reconstruction.float() - target.float()).pow(2)
    return (squared * weights).sum() / weights.sum().clamp_min(1.0)


def augment_eeg(x: torch.Tensor) -> torch.Tensor:
    if torch.rand(()) < 0.9:
        noise_scale = 0.02 + 0.03 * torch.rand((x.shape[0], 1, 1), device=x.device)
        x = x + torch.randn_like(x) * noise_scale
    if torch.rand(()) < 0.6:
        keep = (torch.rand((x.shape[0], x.shape[1], 1), device=x.device) > 0.05).float()
        x = x * keep
    if torch.rand(()) < 0.7:
        width = max(4, x.shape[-1] // 12)
        starts = torch.randint(0, x.shape[-1] - width + 1, (x.shape[0],), device=x.device)
        time_index = torch.arange(x.shape[-1], device=x.device).view(1, 1, -1)
        masked = (time_index >= starts.view(-1, 1, 1)) & (time_index < (starts + width).view(-1, 1, 1))
        x = x.masked_fill(masked, 0.0)
    return x


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)
    order = torch.randperm(x.shape[0], device=x.device)
    return lam * x + (1.0 - lam) * x[order], y, y[order], lam


def parse_tta_shifts(spec: str) -> list[int]:
    shifts = [int(value.strip()) for value in spec.split(",") if value.strip()]
    if not shifts:
        raise ValueError("--valid-tta-shifts needs at least one integer shift.")
    if 0 not in shifts:
        shifts.insert(0, 0)
    return list(dict.fromkeys(shifts))


def averaged_logits(model: nn.Module, x: torch.Tensor, tta_shifts: list[int]) -> torch.Tensor:
    logits = []
    for shift in tta_shifts:
        shifted = x if shift == 0 else torch.roll(x, shifts=shift, dims=-1)
        logits.append(model(shifted))
    return torch.stack(logits, dim=0).mean(dim=0)


def update_confusion(confusion: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor) -> None:
    predictions = logits.argmax(dim=1)
    for target, prediction in zip(labels.detach().cpu(), predictions.detach().cpu()):
        confusion[int(target), int(prediction)] += 1


def metrics_from_confusion(loss: float, confusion: torch.Tensor) -> EpochMetrics:
    confusion = confusion.float()
    true_positive = confusion.diag()
    support = confusion.sum(dim=1).clamp_min(1.0)
    predicted = confusion.sum(dim=0).clamp_min(1.0)
    recall = true_positive / support
    precision = true_positive / predicted
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    total = confusion.sum().clamp_min(1.0)
    return EpochMetrics(
        loss=float(loss),
        accuracy=float(true_positive.sum() / total),
        balanced_accuracy=float(recall.mean()),
        macro_f1=float(f1.mean()),
        confusion=confusion.int().tolist(),
    )


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader, tuple[int, int]]:
    pretrain_ds: Dataset = H5EEGDataset(args.pretrain_h5)
    train_ds: Dataset = H5EEGDataset(args.train_h5)
    valid_ds: Dataset = H5EEGDataset(args.valid_h5)
    chans, time_points = train_ds.sample_shape  # type: ignore[attr-defined]
    expected_shape = train_ds.sample_shape  # type: ignore[attr-defined]
    for split_name, dataset in [("pretrain", pretrain_ds), ("valid", valid_ds)]:
        if dataset.sample_shape != expected_shape:  # type: ignore[attr-defined]
            raise ValueError(
                f"Train/{split_name} sample shapes differ: {expected_shape} vs {dataset.sample_shape}."
            )
    if args.smoke_test:
        pretrain_ds = Subset(pretrain_ds, range(min(1024, len(pretrain_ds))))
        train_ds = Subset(train_ds, range(min(1024, len(train_ds))))
        valid_ds = Subset(valid_ds, range(min(512, len(valid_ds))))
        args.epochs = min(args.epochs, 2)
        args.patience = min(args.patience, 2)
        args.pretrain_epochs = min(args.pretrain_epochs, 2)
        args.workers = 0

    loader_kwargs = {
        "num_workers": args.workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.workers > 0,
    }
    pretrain_loader = DataLoader(
        pretrain_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, **loader_kwargs)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, **loader_kwargs)
    return pretrain_loader, train_loader, valid_loader, (int(chans), int(time_points))


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    scaler: torch.amp.GradScaler | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    mixup_alpha: float = 0.0,
    grad_clip: float = 1.0,
    use_amp: bool = False,
    tta_shifts: list[int] | None = None,
) -> EpochMetrics:
    is_train = optimizer is not None and scaler is not None
    model.train(is_train)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    loss_total = 0.0
    sample_total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        metric_y = y
        if is_train:
            x = augment_eeg(x)
            x, target_a, target_b, lam = mixup_batch(x, y, mixup_alpha)
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = averaged_logits(model, x, tta_shifts or [0]) if not is_train else model(x)
            if is_train:
                loss = lam * criterion(logits, target_a) + (1.0 - lam) * criterion(logits, target_b)
            else:
                loss = criterion(logits, y)
        if is_train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
        batch_size = int(y.shape[0])
        loss_total += float(loss.detach()) * batch_size
        sample_total += batch_size
        update_confusion(confusion, logits, metric_y)

    return metrics_from_confusion(loss_total / max(1, sample_total), confusion)


def append_history(path: Path, epoch: int, lr: float, train: EpochMetrics, valid: EpochMetrics) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(
                [
                    "epoch",
                    "lr",
                    "train_loss",
                    "train_accuracy",
                    "train_balanced_accuracy",
                    "train_macro_f1",
                    "valid_loss",
                    "valid_accuracy",
                    "valid_balanced_accuracy",
                    "valid_macro_f1",
                ]
            )
        writer.writerow(
            [
                epoch,
                lr,
                train.loss,
                train.accuracy,
                train.balanced_accuracy,
                train.macro_f1,
                valid.loss,
                valid.accuracy,
                valid.balanced_accuracy,
                valid.macro_f1,
            ]
        )


def append_pretrain_history(path: Path, epoch: int, lr: float, masked_loss: float) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(["epoch", "lr", "masked_mse"])
        writer.writerow([epoch, lr, masked_loss])


def load_pretrained_temporal(model: FusionEEGNet, checkpoint_path: Path) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload["temporal_state"] if "temporal_state" in payload else payload
    model.temporal.load_state_dict(state)
    print(f"Loaded temporal pretrain weights from {checkpoint_path}")


def pretrain_temporal(
    model: FusionEEGNet,
    pretrain_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    run_dir: Path,
    time_points: int,
    use_amp: bool,
) -> Path | None:
    if args.pretrained_temporal is not None:
        load_pretrained_temporal(model, args.pretrained_temporal)
    if args.pretrain_epochs <= 0:
        return args.pretrained_temporal

    pretrainer = MaskedTemporalPretrainer(
        temporal=model.temporal,
        chans=model.connectivity.upper_row.max().item() + 1,
        time_points=time_points,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        pretrainer.parameters(),
        lr=args.pretrain_lr,
        weight_decay=args.pretrain_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.pretrain_lr,
        epochs=args.pretrain_epochs,
        steps_per_epoch=max(1, len(pretrain_loader)),
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=20.0,
        final_div_factor=40.0,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history_path = run_dir / "pretrain_history.csv"
    best_path = run_dir / "pretrain_best.pt"
    best_loss = math.inf

    for epoch in range(1, args.pretrain_epochs + 1):
        pretrainer.train()
        total_loss = 0.0
        total_samples = 0
        for x, _ in pretrain_loader:
            x = x.to(device, non_blocking=True)
            mask = make_pretrain_mask(
                x,
                mask_ratio=args.pretrain_mask_ratio,
                channel_mask_prob=args.pretrain_channel_mask_prob,
                time_blocks=args.pretrain_time_blocks,
                time_block_frac=args.pretrain_time_block_frac,
            )
            masked_x = x.masked_fill(mask, 0.0)
            if args.pretrain_input_noise > 0:
                masked_x = masked_x + torch.randn_like(masked_x) * args.pretrain_input_noise * (~mask).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                reconstruction = pretrainer(masked_x)
                loss = masked_mse(reconstruction, x, mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(pretrainer.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            batch_size = int(x.shape[0])
            total_loss += float(loss.detach()) * batch_size
            total_samples += batch_size

        average_loss = total_loss / max(1, total_samples)
        lr = float(optimizer.param_groups[0]["lr"])
        append_pretrain_history(history_path, epoch, lr, average_loss)
        payload = {
            "epoch": epoch,
            "pretrainer_state": pretrainer.state_dict(),
            "temporal_state": model.temporal.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "masked_mse": average_loss,
            "config": vars(args).copy(),
        }
        torch.save(payload, run_dir / "pretrain_last.pt")
        if average_loss < best_loss:
            best_loss = average_loss
            torch.save(payload, best_path)
        print(
            f"pretrain epoch {epoch:03d}/{args.pretrain_epochs} "
            f"masked_mse={average_loss:.5f} best={best_loss:.5f}"
        )

    load_pretrained_temporal(model, best_path)
    return best_path


def main() -> None:
    args = parse_args()
    required_h5 = [args.pretrain_h5, args.train_h5, args.valid_h5]
    if any(not path.exists() for path in required_h5):
        missing = [str(path) for path in required_h5 if not path.exists()]
        raise FileNotFoundError(f"Run prepare_server_data.py first. Missing H5 files: {missing}")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    run_name = args.run_name or datetime.now().strftime("fusion_eegnet_%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    pretrain_loader, train_loader, valid_loader, (chans, time_points) = make_loaders(args)
    valid_tta_shifts = parse_tta_shifts(args.valid_tta_shifts)
    counts = read_h5_label_counts(args.train_h5, args.num_classes)
    class_weights = counts.sum() / (args.num_classes * counts.clamp_min(1.0))
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=args.label_smoothing,
    )
    model = FusionEEGNet(
        chans=chans,
        num_classes=args.num_classes,
        dropout=args.dropout,
        model_size=args.model_size,
    ).to(device)
    pretrain_checkpoint = pretrain_temporal(
        model,
        pretrain_loader,
        args,
        device,
        run_dir,
        time_points,
        use_amp,
    )
    if args.compile and device.type == "cuda" and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=max(1, len(train_loader)),
        pct_start=0.12,
        anneal_strategy="cos",
        div_factor=max(1.0, args.lr / max(args.min_lr, 1e-8)),
        final_div_factor=20.0,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    config = vars(args).copy()
    config.update(
        {
            "device": str(device),
            "amp": use_amp,
            "input_shape": [chans, time_points],
            "train_label_counts": counts.int().tolist(),
            "class_weights": class_weights.tolist(),
            "parameters": parameter_count,
            "valid_tta_shifts": valid_tta_shifts,
            "pretrain_checkpoint": str(pretrain_checkpoint) if pretrain_checkpoint is not None else None,
            "pretrain_batches": len(pretrain_loader),
            "train_batches": len(train_loader),
            "valid_batches": len(valid_loader),
        }
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
    print(
        f"Device: {device}; AMP: {use_amp}; model_size: {args.model_size}; "
        f"valid_tta_shifts: {valid_tta_shifts}; parameters: {parameter_count:,}"
    )
    print(
        f"Pretrain batches: {len(pretrain_loader)}; train batches: {len(train_loader)}; "
        f"valid batches: {len(valid_loader)}; run_dir: {run_dir}"
    )

    best_score = -math.inf
    best_epoch = 0
    history_path = run_dir / "history.csv"
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            args.num_classes,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            mixup_alpha=args.mixup_alpha,
            grad_clip=args.grad_clip,
            use_amp=use_amp,
        )
        with torch.no_grad():
            valid_metrics = run_epoch(
                model,
                valid_loader,
                criterion,
                device,
                args.num_classes,
                use_amp=use_amp,
                tta_shifts=valid_tta_shifts,
            )
        lr = float(optimizer.param_groups[0]["lr"])
        append_history(history_path, epoch, lr, train_metrics, valid_metrics)
        score = 0.5 * (valid_metrics.macro_f1 + valid_metrics.balanced_accuracy)
        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "train_metrics": asdict(train_metrics),
            "valid_metrics": asdict(valid_metrics),
            "config": config,
        }
        torch.save(payload, run_dir / "last.pt")
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(payload, run_dir / "best.pt")
        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train loss={train_metrics.loss:.4f} f1={train_metrics.macro_f1:.4f} "
            f"valid loss={valid_metrics.loss:.4f} acc={valid_metrics.accuracy:.4f} "
            f"bacc={valid_metrics.balanced_accuracy:.4f} f1={valid_metrics.macro_f1:.4f} "
            f"best_epoch={best_epoch:03d}"
        )
        if epoch - best_epoch >= args.patience:
            print(f"Early stopping after {args.patience} epochs without a validation score improvement.")
            break

    summary = {
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "pretrain_checkpoint": str(pretrain_checkpoint) if pretrain_checkpoint is not None else None,
        "history_csv": str(history_path),
        "best_checkpoint": str(run_dir / "best.pt"),
        "last_checkpoint": str(run_dir / "last.pt"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
