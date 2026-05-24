"""
Check prediction txt files before submission.

The script verifies row count, label range, class counts, and whether
submit/SEED.txt matches the result regenerated from the packaged probability
files. It does not use hidden test labels.
"""

from hashlib import sha256
from pathlib import Path
import subprocess
import sys


PACKAGE_DIR = Path(__file__).resolve().parents[1]
SUBMIT_DIR = PACKAGE_DIR / "submit"
EXPECTED_FINAL_HASH = "9769fb53aa14317b5b9bd1e5d892aa02680e57c5d23a77b266bb3be6a4ec4d90"


def read_labels(path):
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(int(line))
    return labels


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def check_one(path):
    labels = read_labels(path)
    ok_len = len(labels) == 450
    ok_label = all(x in [0, 1, 2] for x in labels)

    counts = {}
    for x in labels:
        counts[x] = counts.get(x, 0) + 1

    print(path.name)
    print("  rows:", len(labels), "OK" if ok_len else "BAD")
    print("  labels only 0/1/2:", ok_label)
    print("  counts:", counts)


def main():
    final_file = SUBMIT_DIR / "SEED.txt"
    safe_file = SUBMIT_DIR / "SEED_safe.txt"
    balanced_file = SUBMIT_DIR / "SEED_balanced.txt"

    check_one(final_file)
    check_one(safe_file)
    check_one(balanced_file)

    final_hash = file_hash(final_file)
    print("\nfinal hash:", final_hash)
    print("matches expected final hash:", final_hash == EXPECTED_FINAL_HASH)

    print("\nregenerating final file from packaged probabilities...")
    subprocess.run([sys.executable, str(PACKAGE_DIR / "code" / "final_fusion.py")], check=True)
    regenerated_hash = file_hash(final_file)
    print("hash after regeneration:", regenerated_hash)
    print("regenerated file matches expected final hash:", regenerated_hash == EXPECTED_FINAL_HASH)


if __name__ == "__main__":
    main()
