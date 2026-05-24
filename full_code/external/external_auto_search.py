#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Result-driven auto search for raw external SEED-style EEG usage.

The search deliberately keeps validation/test out of preprocessing decisions.
Official validation is used only by the training script for early stopping and
model selection.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "course_project" / "SEED"
TRAIN_H5 = DATA_DIR / "train.h5"
EXTERNAL_SOURCE = ROOT / "course_project" / "external_data" / "SEED_official_processed" / "external_seed_raw_400pt_no_window_zscore.h5"
OUTPUT_ROOT = ROOT / "outputs_external_auto_search"
CONTROL_TARGET = 0.4356


RESULT_FIELDS = [
    "run_id", "stage", "parent_run_id", "hypothesis", "method", "normalization", "windowing",
    "samples_per_class", "stat_distance_filter", "external_ratio", "external_loss_weight",
    "pretrain_type", "freeze_backbone_epochs", "backbone_lr", "classifier_lr", "seed", "epochs",
    "best_val_acc", "best_epoch", "final_val_acc", "macro_f1", "prediction_distribution",
    "output_dir", "checkpoint_path", "status", "error_message", "notes",
]


@dataclass
class Candidate:
    run_id: str
    stage: str
    hypothesis: str
    method: str = "supervised_mix"
    normalization: str = "per_window_zscore"
    windowing: str = "random_windows"
    samples_per_class: int = 200
    external_ratio: float = 0.02
    external_loss_weight: float = 1.0
    pretrain_type: str = "none"
    pretrain_epochs: int = 0
    freeze_backbone_epochs: int = 0
    backbone_lr: float = 0.0
    classifier_lr: float = 0.0
    seed: int = 42
    epochs: int = 7
    patience: int = 3
    stat_distance_filter: bool = False
    stat_distance_ratio: float = 1.0
    parent_run_id: str = ""
    notes: str = ""
    h5_path: Optional[Path] = None
    extra: Dict[str, object] = field(default_factory=dict)


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def label_distribution(y: np.ndarray) -> Dict[str, int]:
    vals, counts = np.unique(y.astype(int), return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(vals, counts)}


def h5_stats(path: Path, include_sample_ranges: bool = True) -> Dict[str, object]:
    with h5py.File(path, "r") as h5:
        X = h5["X"]
        y = h5["y"][()].astype(np.int64) if "y" in h5 else None
        sums = np.zeros((X.shape[1],), dtype=np.float64)
        sums2 = np.zeros((X.shape[1],), dtype=np.float64)
        sample_means: List[np.ndarray] = []
        sample_stds: List[np.ndarray] = []
        finite = True
        min_value = math.inf
        max_value = -math.inf
        count = 0
        for start in range(0, X.shape[0], 512):
            arr = X[start:start + 512].astype(np.float64)
            finite = finite and bool(np.isfinite(arr).all())
            min_value = min(min_value, float(np.nanmin(arr)))
            max_value = max(max_value, float(np.nanmax(arr)))
            sums += arr.sum(axis=(0, 2))
            sums2 += np.square(arr).sum(axis=(0, 2))
            count += arr.shape[0] * arr.shape[2]
            if include_sample_ranges:
                sample_means.append(arr.mean(axis=(1, 2)))
                sample_stds.append(arr.std(axis=(1, 2)))
        ch_mean = sums / max(1, count)
        ch_var = np.maximum(sums2 / max(1, count) - np.square(ch_mean), 1e-12)
        ch_std = np.sqrt(ch_var)
        out = {
            "path": str(path),
            "x_shape": list(X.shape),
            "y_distribution": label_distribution(y) if y is not None else None,
            "channel_mean_range": [float(ch_mean.min()), float(ch_mean.max())],
            "channel_std_range": [float(ch_std.min()), float(ch_std.max())],
            "finite": finite,
            "value_range": [min_value, max_value],
            "attrs": {k: str(v) for k, v in h5.attrs.items()},
        }
        if include_sample_ranges and sample_means:
            sm = np.concatenate(sample_means)
            ss = np.concatenate(sample_stds)
            out["sample_mean_range"] = [float(sm.min()), float(sm.max())]
            out["sample_std_range"] = [float(ss.min()), float(ss.max())]
        return out


def write_audit(run_dir: Path) -> Path:
    audit_path = run_dir / "data_distribution_audit.md"
    official = h5_stats(TRAIN_H5)
    external = h5_stats(EXTERNAL_SOURCE)
    supplement_path = ROOT / "outputs_external" / "supplement_seed_like.h5"
    supplement = h5_stats(supplement_path) if supplement_path.exists() else None
    report_json = EXTERNAL_SOURCE.with_suffix(".conversion_report.json")
    raw_notes = {}
    if report_json.exists():
        try:
            raw_notes = json.loads(report_json.read_text(encoding="utf-8"))
        except Exception:
            raw_notes = {"note": "conversion report exists but could not be parsed"}

    def block(title: str, stats: Dict[str, object]) -> List[str]:
        return [
            f"## {title}",
            f"- X shape: `{stats['x_shape']}`",
            f"- y distribution: `{stats.get('y_distribution')}`",
            f"- channel mean range: `{stats['channel_mean_range']}`",
            f"- channel std range: `{stats['channel_std_range']}`",
            f"- sample mean range: `{stats.get('sample_mean_range')}`",
            f"- sample std range: `{stats.get('sample_std_range')}`",
            f"- finite values: `{stats['finite']}`",
            f"- value range: `{stats['value_range']}`",
            "",
        ]

    lines = [
        "# Data Distribution Audit",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Validation/test data are not used for preprocessing statistics in this audit.",
        "",
    ]
    lines += block("Official SEED Train", official)
    lines += [
        "## Raw External Source",
        f"- source h5: `{EXTERNAL_SOURCE}`",
        f"- mat file count from existing report: `{raw_notes.get('mat_files', raw_notes.get('n_files', 'unknown'))}`",
        f"- trial/window note: source is the pre-converted raw SEED-style 400-point external h5; candidates are derived from it.",
        "",
    ]
    lines += block("External Converted Source", external)
    if supplement:
        lines += block("Existing supplement_seed_like.h5", supplement)
    scale_note = "External and official train are on different raw scales; normalization/alignment should be searched first."
    try:
        official_std_mid = np.mean(official["channel_std_range"])
        external_std_mid = np.mean(external["channel_std_range"])
        if official_std_mid > 0:
            ratio = external_std_mid / official_std_mid
            scale_note = f"External/std to official/std rough ratio is `{ratio:.3f}`, so scale handling is a first-order hypothesis."
    except Exception:
        pass
    lines += ["## Audit Interpretation", f"- {scale_note}", ""]
    audit_path.write_text("\n".join(lines), encoding="utf-8")
    return audit_path


def run_command(cmd: List[str], cwd: Path, log_path: Path, timeout: int = 3600) -> Tuple[int, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    return proc.returncode, proc.stdout


def convert_candidate(c: Candidate, run_dir: Path) -> Path:
    h5_name = (
        f"{c.normalization}__{c.windowing}__spc{c.samples_per_class}"
        f"__stat{int(c.stat_distance_filter)}_{c.stat_distance_ratio:g}__seed{c.seed}.h5"
    )
    out_h5 = run_dir / "converted_h5" / h5_name
    if out_h5.exists():
        return out_h5
    cmd = [
        sys.executable, "external_raw_to_seed_h5.py",
        "--source", str(EXTERNAL_SOURCE),
        "--output-h5", str(out_h5),
        "--report", str(out_h5.with_suffix(".report.md")),
        "--normalization", c.normalization,
        "--windowing", c.windowing,
        "--samples-per-class", str(c.samples_per_class),
        "--max-windows", "0",
        "--seed", str(c.seed),
        "--official-train-h5", str(TRAIN_H5),
    ]
    if c.stat_distance_filter:
        cmd += ["--select-by-stat-distance", "--stat-distance-ratio", str(c.stat_distance_ratio)]
    code, out = run_command(cmd, ROOT, run_dir / "logs" / f"{c.run_id}_convert.log", timeout=1800)
    if code != 0:
        raise RuntimeError(f"conversion failed for {c.run_id}:\n{out[-2000:]}")
    return out_h5


def train_candidate(c: Candidate, run_dir: Path) -> Dict[str, object]:
    out_root = run_dir / "train_runs"
    cmd = [
        sys.executable, "seed_multiscale_crnn_experiment.py",
        "--model", "multiscale_crnn",
        "--epochs", str(c.epochs),
        "--patience", str(c.patience),
        "--batch-size", "64",
        "--seed", str(c.seed),
        "--output-root", str(out_root),
        "--run-name", c.run_id,
    ]
    if c.method in {"supervised_mix", "source_weighted"}:
        c.h5_path = convert_candidate(c, run_dir)
        cmd += ["--use-supplement", "--supplement-h5", str(c.h5_path), "--external-ratio", str(c.external_ratio)]
        if c.method == "source_weighted":
            cmd += ["--source-aware-loss", "--external-loss-weight", str(c.external_loss_weight)]
    elif c.method == "pretrain_finetune":
        c.h5_path = convert_candidate(c, run_dir)
        cmd += [
            "--supplement-h5", str(c.h5_path),
            "--pretrain-type", c.pretrain_type,
            "--pretrain-epochs", str(c.pretrain_epochs),
            "--freeze-backbone-epochs", str(c.freeze_backbone_epochs),
        ]
        if c.backbone_lr > 0:
            cmd += ["--backbone-lr", str(c.backbone_lr)]
        if c.classifier_lr > 0:
            cmd += ["--classifier-lr", str(c.classifier_lr)]
    code, out = run_command(cmd, ROOT, run_dir / "logs" / f"{c.run_id}_train.log", timeout=3600)
    if code != 0:
        raise RuntimeError(f"training failed for {c.run_id}:\n{out[-3000:]}")
    run_dirs = sorted(out_root.glob(f"{c.run_id}_*"), key=lambda p: p.stat().st_mtime)
    if not run_dirs:
        raise RuntimeError(f"training finished but no output dir found for {c.run_id}")
    result_path = run_dirs[-1] / "run_results.json"
    return json.loads(result_path.read_text(encoding="utf-8"))


def result_row(c: Candidate, status: str, result: Optional[Dict[str, object]] = None, error: str = "") -> Dict[str, object]:
    result = result or {}
    val_metrics = result.get("validation_metrics_at_best") or {}
    row = {
        "run_id": c.run_id,
        "stage": c.stage,
        "parent_run_id": c.parent_run_id,
        "hypothesis": c.hypothesis,
        "method": c.method,
        "normalization": c.normalization,
        "windowing": c.windowing,
        "samples_per_class": c.samples_per_class,
        "stat_distance_filter": c.stat_distance_filter,
        "external_ratio": c.external_ratio if c.method != "pretrain_finetune" else "",
        "external_loss_weight": c.external_loss_weight if c.method == "source_weighted" else "",
        "pretrain_type": c.pretrain_type if c.method == "pretrain_finetune" else "",
        "freeze_backbone_epochs": c.freeze_backbone_epochs,
        "backbone_lr": c.backbone_lr,
        "classifier_lr": c.classifier_lr,
        "seed": c.seed,
        "epochs": c.epochs,
        "best_val_acc": result.get("best_val_acc", ""),
        "best_epoch": result.get("best_epoch", ""),
        "final_val_acc": result.get("final_val_acc", ""),
        "macro_f1": result.get("best_val_macro_f1", ""),
        "prediction_distribution": json.dumps(val_metrics.get("prediction_distribution", {}), ensure_ascii=False),
        "output_dir": result.get("output_dir", ""),
        "checkpoint_path": result.get("best_acc_checkpoint", ""),
        "status": status,
        "error_message": error[:800],
        "notes": c.notes,
    }
    return row


def append_results(path: Path, rows: List[Dict[str, object]]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=RESULT_FIELDS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def stage1_candidates() -> List[Candidate]:
    return [
        Candidate("s1_noz_rand200_r001", "stage1", "Test tiny raw-scale supplement: maybe previous ratios were too high.", "supervised_mix", "no_zscore", "random_windows", 200, 0.01),
        Candidate("s1_noz_mid200_r002", "stage1", "Middle windows may be less transition/noisy than arbitrary windows.", "supervised_mix", "no_zscore", "middle_windows", 200, 0.02),
        Candidate("s1_winz_rand200_r002", "stage1", "Per-window channel z-score may remove external amplitude shift.", "supervised_mix", "per_window_zscore", "random_windows", 200, 0.02),
        Candidate("s1_winz_rand500_r005", "stage1", "More normalized external samples with still-small ratio.", "supervised_mix", "per_window_zscore", "random_windows", 500, 0.05),
        Candidate("s1_trialz_nonover200_r002", "stage1", "Trial/window z-score with balanced non-overlap sampling.", "supervised_mix", "per_trial_zscore", "non_overlap_balanced", 200, 0.02),
        Candidate("s1_extstat_rand200_r002", "stage1", "External global z-score may stabilize source scale.", "supervised_mix", "external_train_stat_zscore", "random_windows", 200, 0.02),
        Candidate("s1_align_rand200_r002", "stage1", "Align external channel statistics to official train statistics.", "supervised_mix", "align_to_official_train_stat", "random_windows", 200, 0.02),
        Candidate("s1_winz_stride200_r005", "stage1", "Overlap-style candidate may improve coverage without full dominance.", "supervised_mix", "per_window_zscore", "stride_200_overlap_balanced", 200, 0.05),
        Candidate("s1_w005_winz500_r010", "stage1", "If labels are shifted, keep external in batches but down-weight loss.", "source_weighted", "per_window_zscore", "random_windows", 500, 0.10, 0.05),
        Candidate("s1_w010_align500_r010", "stage1", "Combine statistic alignment with low external loss weight.", "source_weighted", "align_to_official_train_stat", "random_windows", 500, 0.10, 0.10),
        Candidate("s1_pre_mask_winz500", "stage1", "Use external only for masked reconstruction pretrain, not supervised concat.", "pretrain_finetune", "per_window_zscore", "random_windows", 500, 0.0, pretrain_type="masked_reconstruction", pretrain_epochs=2),
        Candidate("s1_pre_sup_winz500", "stage1", "Use external supervised labels only for initialization, then official fine-tune.", "pretrain_finetune", "per_window_zscore", "random_windows", 500, 0.0, pretrain_type="supervised", pretrain_epochs=2),
    ]


def numeric(row: Dict[str, object], key: str, default: float = -1.0) -> float:
    try:
        if row.get(key) == "":
            return default
        return float(row.get(key, default))
    except Exception:
        return default


def make_stage2(stage1: List[Dict[str, object]], control_acc: float) -> List[Candidate]:
    ok = [r for r in stage1 if r.get("status") == "ok"]
    supervised = [r for r in ok if r.get("method") in {"supervised_mix", "source_weighted"}]
    pretrain = [r for r in ok if r.get("method") == "pretrain_finetune"]
    best = max(ok, key=lambda r: numeric(r, "best_val_acc"), default=None)
    best_norm = best.get("normalization", "per_window_zscore") if best else "per_window_zscore"
    best_window = best.get("windowing", "random_windows") if best else "random_windows"
    all_supervised_bad = supervised and max(numeric(r, "best_val_acc") for r in supervised) < control_acc - 1e-9
    ratio_worse = False
    ratios = sorted((numeric(r, "external_ratio", 0), numeric(r, "best_val_acc")) for r in supervised if r.get("external_ratio") != "")
    if len(ratios) >= 2:
        ratio_worse = ratios[-1][1] < ratios[0][1]
    parent = best.get("run_id", "") if best else ""
    cands: List[Candidate] = []
    if all_supervised_bad or ratio_worse:
        cands += [
            Candidate("s2_verylow_winz500_r0005", "stage2", "Stage1 says direct supervised mix is harmful; reduce ratio to 0.005.", "supervised_mix", best_norm, best_window, 500, 0.005, parent_run_id=parent),
            Candidate("s2_verylow_winz500_r001", "stage2", "Check whether a barely-present external regularizer is less harmful.", "supervised_mix", best_norm, best_window, 500, 0.01, parent_run_id=parent),
            Candidate("s2_weight001_500_r005", "stage2", "Treat external labels as weak labels with 0.01 loss weight.", "source_weighted", best_norm, best_window, 500, 0.05, 0.01, parent_run_id=parent),
            Candidate("s2_weight003_500_r005", "stage2", "Slightly higher weak-label weight after harmful concat.", "source_weighted", best_norm, best_window, 500, 0.05, 0.03, parent_run_id=parent),
        ]
    else:
        cands += [
            Candidate("s2_refine_r001", "stage2", "Refine the best normalization/windowing at lower ratio.", "supervised_mix", best_norm, best_window, 500, 0.01, parent_run_id=parent),
            Candidate("s2_refine_r002", "stage2", "Refine the best normalization/windowing at the apparent useful ratio.", "supervised_mix", best_norm, best_window, 1000, 0.02, parent_run_id=parent),
        ]
    cands += [
        Candidate("s2_statdist020_weight003", "stage2", "Keep only external windows statistically closest to official train and weakly weight them.", "source_weighted", best_norm, best_window, 1000, 0.05, 0.03, stat_distance_filter=True, stat_distance_ratio=0.2, parent_run_id=parent),
        Candidate("s2_statdist050_r001", "stage2", "Closest-half stat filter plus very low ratio tests distribution-shift hypothesis.", "supervised_mix", "external_train_stat_zscore", best_window, 1000, 0.01, stat_distance_filter=True, stat_distance_ratio=0.5, parent_run_id=parent),
        Candidate("s2_pre_mask_freeze3", "stage2", "If pretrain is less harmful, freeze pretrained backbone for warmup.", "pretrain_finetune", best_norm, best_window, 500, 0.0, pretrain_type="masked_reconstruction", pretrain_epochs=3, freeze_backbone_epochs=3, parent_run_id=parent),
        Candidate("s2_pre_sup_lowlr", "stage2", "External supervised pretrain with lower backbone LR during official fine-tune.", "pretrain_finetune", best_norm, best_window, 500, 0.0, pretrain_type="supervised", pretrain_epochs=3, backbone_lr=1e-4, classifier_lr=5e-4, parent_run_id=parent),
    ]
    return cands[:10]


def run_candidates(candidates: Iterable[Candidate], run_dir: Path, results_csv: Path) -> List[Dict[str, object]]:
    rows = []
    failed = run_dir / "failed_runs.txt"
    for c in candidates:
        print(f"\n=== {c.run_id}: {c.hypothesis}")
        try:
            result = train_candidate(c, run_dir)
            row = result_row(c, "ok", result=result)
        except Exception as exc:
            err = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            failed.write_text((failed.read_text(encoding="utf-8") if failed.exists() else "") + f"\n## {c.run_id}\n{err}\n", encoding="utf-8")
            row = result_row(c, "failed", error=err)
        rows.append(row)
        append_results(results_csv, [row])
        print(f"{c.run_id}: {row['status']} best={row['best_val_acc']} f1={row['macro_f1']}")
    return rows


def confirmation_candidates(rows: List[Dict[str, object]], control_acc: float) -> List[Candidate]:
    ok = [r for r in rows if r.get("status") == "ok" and numeric(r, "best_val_acc") > control_acc]
    ok = sorted(ok, key=lambda r: numeric(r, "best_val_acc"), reverse=True)[:3]
    cands: List[Candidate] = []
    for r in ok:
        for seed in [3407, 2026]:
            cands.append(Candidate(
                f"confirm_{r['run_id']}_seed{seed}",
                "confirm",
                f"Confirm whether {r['run_id']} beats no_external beyond seed 42.",
                method=str(r["method"]),
                normalization=str(r["normalization"]),
                windowing=str(r["windowing"]),
                samples_per_class=int(r["samples_per_class"]),
                external_ratio=float(r["external_ratio"] or 0.0),
                external_loss_weight=float(r["external_loss_weight"] or 1.0),
                pretrain_type=str(r["pretrain_type"] or "none"),
                pretrain_epochs=3 if r["method"] == "pretrain_finetune" else 0,
                freeze_backbone_epochs=int(r["freeze_backbone_epochs"] or 0),
                backbone_lr=float(r["backbone_lr"] or 0.0),
                classifier_lr=float(r["classifier_lr"] or 0.0),
                seed=seed,
                epochs=20,
                patience=8,
                stat_distance_filter=str(r["stat_distance_filter"]).lower() == "true",
                parent_run_id=str(r["run_id"]),
            ))
    return cands


def write_reports(run_dir: Path, rows: List[Dict[str, object]], audit_path: Path, control_acc: float) -> Tuple[Path, Path]:
    ok = [r for r in rows if r.get("status") == "ok"]
    best = max(ok, key=lambda r: numeric(r, "best_val_acc"), default=None)
    supervised = [r for r in ok if r.get("method") in {"supervised_mix", "source_weighted"}]
    pretrain = [r for r in ok if r.get("method") == "pretrain_finetune"]
    best_sup = max(supervised, key=lambda r: numeric(r, "best_val_acc"), default=None)
    best_pre = max(pretrain, key=lambda r: numeric(r, "best_val_acc"), default=None)
    harmful = [r for r in ok if numeric(r, "best_val_acc") <= 0.35]
    improved = [r for r in ok if numeric(r, "best_val_acc") > control_acc]

    def line_for(r: Optional[Dict[str, object]]) -> str:
        if not r:
            return "none"
        return f"`{r['run_id']}` {r['method']} norm={r['normalization']} window={r['windowing']} acc={numeric(r, 'best_val_acc'):.4f} f1={numeric(r, 'macro_f1'):.4f}"

    stage1 = [r for r in ok if r.get("stage") == "stage1"]
    stage2 = [r for r in ok if r.get("stage") == "stage2"]
    confirm = [r for r in ok if r.get("stage") == "confirm"]
    report = run_dir / "auto_search_report.md"
    lines = [
        "# External Auto Search Report",
        "",
        f"- audit: `{audit_path}`",
        f"- no_external control best val acc used for decisions: `{control_acc:.4f}`",
        "",
        "## Why direct concat likely failed",
        "- The external source is much larger than official train, so even small mistakes in sampling/scale can dominate gradients.",
        "- Window-level random sampling can overrepresent subject/session-specific external artifacts.",
        "- External labels are formally compatible but may encode a different subject/session distribution than the course split.",
        "- Previous ratio 0.1/0.3 runs collapsed toward weak validation behavior, consistent with distribution shift or label noise.",
        "",
        "## Search Logic",
        "- Stage 1 covered normalization, windowing, sampling, tiny ratios, source-weighted weak labels, and pretrain-only variants.",
        "- Stage 2 was generated from Stage 1: if supervised mix was below control, it moved to very-low-ratio, weak source loss, stat-distance filtering, and pretrain/freeze variants.",
        "- Early stopping always came from official SEED validation through the training script.",
        "",
        "## Stage 1 Results",
    ]
    lines += [f"- {line_for(r)}" for r in sorted(stage1, key=lambda r: numeric(r, "best_val_acc"), reverse=True)]
    lines += ["", "## Adjustment After Stage 1"]
    if supervised and max(numeric(r, "best_val_acc") for r in supervised) < control_acc:
        lines.append("- Supervised external remained below control, so Stage 2 reduced ratio/loss weight and tried distribution-distance filtering and pretrain-only.")
    else:
        lines.append("- At least one supervised candidate approached/exceeded control, so Stage 2 refined its normalization/windowing family.")
    lines += ["", "## Stage 2 Results"]
    lines += [f"- {line_for(r)}" for r in sorted(stage2, key=lambda r: numeric(r, "best_val_acc"), reverse=True)]
    lines += ["", "## Confirmation Results"]
    if confirm:
        lines += [f"- {line_for(r)}" for r in sorted(confirm, key=lambda r: numeric(r, "best_val_acc"), reverse=True)]
    else:
        lines.append("- No candidate exceeded the no_external control, so multi-seed confirmation was not triggered.")
    lines += [
        "",
        "## Best Configs",
        f"- Best overall: {line_for(best)}",
        f"- Best supervised external: {line_for(best_sup)}",
        f"- Best pretrain: {line_for(best_pre)}",
        "",
        "## External Helped?",
        f"- {'Yes' if improved else 'No'}. {len(improved)} run(s) exceeded the no_external control." if improved else "- No. None of the completed external strategies exceeded the no_external control.",
        "",
        "## Clearly Harmful / Drop",
    ]
    if harmful:
        lines += [f"- Drop {line_for(r)}; validation accuracy is near/chance or prediction distribution is likely collapsed." for r in harmful[:8]]
    else:
        lines.append("- No completed run was catastrophically below 0.35, but any below control should not be used as final.")
    lines += [
        "",
        "## Next Steps",
        "- External supervised mixing is not worth expanding unless subject/session provenance can be matched to the course split.",
        "- If continuing, prioritize pretrain-only or domain-adaptation losses over more supervised concat ratios.",
        "- A better external filter would use subject/session metadata from raw mat files rather than only window statistics.",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")

    best_strategy = run_dir / "best_strategy.md"
    chosen = best if best and numeric(best, "best_val_acc") >= control_acc else best
    seed_txt = ""
    if chosen and chosen.get("output_dir"):
        seed_txt = str(Path(str(chosen["output_dir"])) / "SEED.txt")
    bs = [
        "# Best Strategy",
        "",
        f"- Final selected config: {line_for(chosen)}",
        f"- Why selected: highest completed validation result while respecting no leakage; {'it exceeded control' if chosen and numeric(chosen, 'best_val_acc') > control_acc else 'it is the least harmful external attempt but does not beat no_external'}.",
        f"- Exceeds no_external: `{bool(chosen and numeric(chosen, 'best_val_acc') > control_acc)}`",
        f"- Reproduce from this run: use the matching row in `{run_dir / 'results.csv'}` and rerun the command logged under `{run_dir / 'logs'}`.",
        f"- Checkpoint path: `{chosen.get('checkpoint_path', '') if chosen else ''}`",
        f"- SEED.txt path: `{seed_txt}`",
        "",
        "## Practical Recommendation",
    ]
    if chosen and numeric(chosen, "best_val_acc") > control_acc:
        bs.append("- Use this external strategy and confirm once more with a longer final run before submission.")
    else:
        bs.append("- Use no_external as the final learned model. External data is currently best treated as an analysis/pretraining source, not supervised supplement.")
    best_strategy.write_text("\n".join(bs), encoding="utf-8")
    return report, best_strategy


def main() -> None:
    run_dir = OUTPUT_ROOT / f"run_{now_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "converted_h5").mkdir()
    (run_dir / "train_runs").mkdir()
    (run_dir / "logs").mkdir()
    results_csv = run_dir / "results.csv"
    print(f"run_dir: {run_dir}")

    audit_path = write_audit(run_dir)
    print(f"audit: {audit_path}")

    control = Candidate("control_no_external", "control", "Official train only control for this auto-search.", "no_external", epochs=7, patience=3)
    control_result = train_candidate(control, run_dir)
    control_row = result_row(control, "ok", result=control_result)
    append_results(results_csv, [control_row])
    control_acc = numeric(control_row, "best_val_acc", CONTROL_TARGET)
    print(f"control best val acc: {control_acc:.4f}")

    rows: List[Dict[str, object]] = [control_row]
    s1_rows = run_candidates(stage1_candidates(), run_dir, results_csv)
    rows += s1_rows
    s2 = make_stage2(s1_rows, control_acc)
    s2_rows = run_candidates(s2, run_dir, results_csv)
    rows += s2_rows
    conf = confirmation_candidates(s1_rows + s2_rows, control_acc)
    if conf:
        rows += run_candidates(conf, run_dir, results_csv)
    report, best_strategy = write_reports(run_dir, rows, audit_path, control_acc)
    print(f"results: {results_csv}")
    print(f"report: {report}")
    print(f"best_strategy: {best_strategy}")


if __name__ == "__main__":
    main()
