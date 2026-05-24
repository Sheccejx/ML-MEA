"""
Final probability fusion for the SEED EEG emotion recognition task.

The script uses four saved candidate probability files in ../probs/test,
applies fixed validation-selected weights, and writes submit/SEED.txt.
No hidden test labels are used.
"""

from pathlib import Path

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROB_DIR = PACKAGE_DIR / "probs"
OUT_DIR = PACKAGE_DIR / "submit"

FINAL_WEIGHTS = np.array([
    0.5221866239430111,
    0.12563498149116575,
    0.2907072375990765,
    0.06147115696674659,
])

SAFE_WEIGHTS = np.array([
    0.5332091377732898,
    0.09347676439736376,
    0.31606850533019126,
    0.05724559249915526,
])


MODEL_FILES = ["model1.npy", "model2.npy", "model3.npy", "model4.npy"]


def load_probs(split_name):
    folder = PROB_DIR / split_name
    probs = []
    for file_name in MODEL_FILES:
        path = folder / file_name
        probs.append(np.load(path))
    return probs


def weighted_average(prob_list, weights):
    result = np.zeros_like(prob_list[0], dtype=float)
    for prob, weight in zip(prob_list, weights):
        result += prob * weight
    return result / result.sum(axis=1, keepdims=True)


def write_txt(labels, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for label in labels:
            f.write(str(int(label)) + "\n")


def count_labels(labels):
    counts = {}
    for label in labels:
        counts[int(label)] = counts.get(int(label), 0) + 1
    return counts


def main():
    test_probs = load_probs("test")

    final_prob = weighted_average(test_probs, FINAL_WEIGHTS)
    safe_prob = weighted_average(test_probs, SAFE_WEIGHTS)

    final_labels = final_prob.argmax(axis=1)
    safe_labels = safe_prob.argmax(axis=1)

    write_txt(final_labels, OUT_DIR / "SEED.txt")
    write_txt(safe_labels, OUT_DIR / "SEED_safe.txt")

    print("final label counts:", count_labels(final_labels))
    print("safe label counts:", count_labels(safe_labels))
    print("final rows:", len(final_labels))
    print("safe rows:", len(safe_labels))


if __name__ == "__main__":
    main()
