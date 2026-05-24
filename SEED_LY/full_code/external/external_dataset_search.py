#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Search/inspect public EEG emotion datasets for SEED compatibility.

The script uses a conservative curated search table plus lightweight URL
checks. It does not download huge archives by default; it records whether a
dataset is directly usable, pretrain-only, or rejected.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs_preprocessing_forensics"


DATASETS: List[Dict[str, object]] = [
    {
        "dataset_name": "FACED / NeMAR NM000112",
        "source_url": "https://eegdash.org/api/dataset/eegdash.dataset.NM000112.html",
        "download_url": "https://openneuro.org/datasets/ds005219",
        "license": "CC-BY-4.0 per EEGDash/NeMAR page",
        "file_format": "BIDS EEG, raw BDF converted to BIDS",
        "eeg_shape": "123 subjects, long recordings; trial spans from event markers",
        "channel_number": "32",
        "channel_names": "standardized International 10-20 names, 32 channels",
        "sampling_rate": "250 or 1000 Hz",
        "label_meaning": "9 emotion categories; binary positive/negative/neutral available in BIDS metadata",
        "subject_session_trial_structure": "123 subjects, 28 video clips, events for video onset/offset and emotion labels",
        "seed_label_mapping": "negative emotions->0, neutral->1, positive emotions->2 is natural",
        "channel_alignment": "partial overlap with SEED 62 channels; not full 62-channel alignment",
        "window_400": "possible after resampling to 200 Hz and taking 2s windows",
        "domain_shift_risk": "medium-high: 32-channel BioSemi/FACED vs 62-channel SEED, different stimuli and labels",
        "decision": "use_pretrain_or_partial_channel_experiment_only",
        "reason": "Best public candidate, but current model expects 62 channels; supervised concat would require a new channel-mapping architecture or imputation.",
    },
    {
        "dataset_name": "DEAP",
        "source_url": "https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html",
        "download_url": "https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html",
        "license": "research access via request server",
        "file_format": "BDF original or Matlab/Python preprocessed",
        "eeg_shape": "32 subjects x 40 trials; 32 EEG channels plus peripheral channels",
        "channel_number": "32 EEG",
        "channel_names": "BioSemi 10-20 subset",
        "sampling_rate": "512 Hz original, 128 Hz preprocessed",
        "label_meaning": "valence/arousal/dominance/liking 1-9 ratings",
        "subject_session_trial_structure": "32 subjects, 40 one-minute music video trials",
        "seed_label_mapping": "valence low/high can map negative/positive, but neutral class is ambiguous",
        "channel_alignment": "partial overlap only",
        "window_400": "possible after resampling, but 128 Hz preprocessed is not 400 points per 2s",
        "domain_shift_risk": "high",
        "decision": "reject_for_supervised_current_run",
        "reason": "Requires request credentials and lacks a natural 3-class negative/neutral/positive mapping without arbitrary thresholds.",
    },
    {
        "dataset_name": "DREAMER",
        "source_url": "https://doi.org/10.1109/JBHI.2017.2688239",
        "download_url": "https://zenodo.org/search?q=DREAMER%20EEG%20emotion%20dataset",
        "license": "varies by mirror; original dataset commonly distributed for research",
        "file_format": "Matlab; EEG/ECG segments",
        "eeg_shape": "23 subjects x 18 stimuli",
        "channel_number": "14 EEG",
        "channel_names": "Emotiv EPOC 14-channel montage",
        "sampling_rate": "128 Hz",
        "label_meaning": "valence/arousal/dominance 1-5 ratings",
        "subject_session_trial_structure": "subject-level baseline/stimulus segments",
        "seed_label_mapping": "positive/negative valence possible; neutral class is unreliable",
        "channel_alignment": "small 14-channel subset only",
        "window_400": "requires resampling to 200 Hz; 14-to-62 channel mismatch",
        "domain_shift_risk": "high",
        "decision": "use_pretrain_only_if_downloaded",
        "reason": "Emotion EEG but too few channels and no robust SEED 3-class supervised mapping.",
    },
    {
        "dataset_name": "Mendeley DEAP/SEED frequency-domain feature collection",
        "source_url": "https://data.mendeley.com/datasets/h75kvfbr73/1",
        "download_url": "https://data.mendeley.com/datasets/h75kvfbr73/1",
        "license": "CC BY 4.0 on Mendeley page",
        "file_format": "processed windows and DE frequency-domain features",
        "eeg_shape": "DEAP/SEED processed feature windows, not raw 62x400 EEG",
        "channel_number": "unclear per file; features rather than raw signal",
        "channel_names": "not guaranteed in downloaded metadata",
        "sampling_rate": "not directly relevant after feature extraction",
        "label_meaning": "emotion labels extracted for datasets/dimensions",
        "subject_session_trial_structure": "processed from DEAP and SEED",
        "seed_label_mapping": "may contain SEED-derived labels, but feature representation mismatches model input",
        "channel_alignment": "not usable for current raw CRNN input",
        "window_400": "no, feature-domain data",
        "domain_shift_risk": "medium for features, impossible for current raw model",
        "decision": "reject_for_current_raw_model",
        "reason": "It is not raw 62x400 EEG and may duplicate SEED-derived features; useful only for separate feature-model baselines.",
    },
]


def check_url(url: str, timeout: int = 15) -> Dict[str, object]:
    req = urllib.request.Request(url, method="GET", headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read(2048)
            return {
                "reachable": True,
                "status": int(resp.status),
                "content_type": resp.headers.get("content-type", ""),
                "content_length": resp.headers.get("content-length", ""),
                "sample_bytes": len(data),
                "error": "",
            }
    except Exception as exc:
        return {"reachable": False, "status": "", "content_type": "", "content_length": "", "sample_bytes": 0, "error": str(exc)}


def write_outputs(run_dir: Path, rows: List[Dict[str, object]]) -> Path:
    csv_path = run_dir / "external_dataset_compatibility.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        fields = list(rows[0].keys())
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    report = run_dir / "external_dataset_search_report.md"
    lines = [
        "# External Dataset Search Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Searched public EEG emotion-recognition candidates with priority on label/channel/sampling compatibility. Large archives were not downloaded blindly.",
        "",
    ]
    for row in rows:
        lines += [
            f"## {row['dataset_name']}",
            f"- source URL: {row['source_url']}",
            f"- download/check URL: {row['download_url']}",
            f"- URL check: reachable=`{row['reachable']}` status=`{row['status']}` content_type=`{row['content_type']}` error=`{row['error']}`",
            f"- license / usage note: {row['license']}",
            f"- file format: {row['file_format']}",
            f"- EEG shape: {row['eeg_shape']}",
            f"- channels: {row['channel_number']} ({row['channel_names']})",
            f"- sampling rate: {row['sampling_rate']}",
            f"- labels: {row['label_meaning']}",
            f"- subject/session/trial: {row['subject_session_trial_structure']}",
            f"- SEED label mapping: {row['seed_label_mapping']}",
            f"- channel alignment: {row['channel_alignment']}",
            f"- 400-point windows: {row['window_400']}",
            f"- domain shift risk: {row['domain_shift_risk']}",
            f"- decision: `{row['decision']}`",
            f"- reason: {row['reason']}",
            "",
        ]
    lines += [
        "## Summary Decision",
        "- FACED is the most promising public external source because it has explicit positive/negative/neutral metadata and event spans, but it has only 32 channels and is large (about 31 GB), so it should not be directly concatenated into the current 62-channel CRNN without an explicit channel-intersection or imputation experiment.",
        "- DEAP and DREAMER are emotion EEG datasets, but their channel count and label schemes make supervised SEED 0/1/2 mapping risky.",
        "- The Mendeley feature collection is not raw 62x400 EEG, so it is excluded from the current raw-model supplement path.",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--run-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or (args.output_root / f"dataset_search_{time.strftime('%Y%m%d_%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in DATASETS:
        checked = dict(item)
        checked.update(check_url(str(item["source_url"])))
        rows.append(checked)
    report = write_outputs(run_dir, rows)
    print(f"report: {report}")
    print(f"compatibility_csv: {run_dir / 'external_dataset_compatibility.csv'}")


if __name__ == "__main__":
    main()
