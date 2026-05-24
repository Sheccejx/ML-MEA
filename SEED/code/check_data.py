"""
Simple data check for SEED train/validation/test_x_only files.

The purpose is only to check whether test_x_only looks very different
from train and validation. It does not use hidden test labels.
"""

from pathlib import Path

import h5py
import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = PACKAGE_DIR.parent
DATA_DIR = PROJECT_DIR / "data_seed_link"


def read_x(file_name):
    path = DATA_DIR / file_name
    with h5py.File(path, "r") as f:
        x = f["X"][:]
        y = f["y"][:] if "y" in f else None
    return x, y


def show_basic_info(name, x, y=None):
    flat = x.reshape(x.shape[0], -1)
    sample_std = flat.std(axis=1)
    channel_std = x.std(axis=(0, 2))

    print("\n[" + name + "]")
    print("shape:", x.shape)
    print("dtype:", x.dtype)
    print("nan:", np.isnan(x).sum(), "inf:", np.isinf(x).sum())
    print("global mean:", float(x.mean()))
    print("global std:", float(x.std()))
    print("min/max:", float(x.min()), float(x.max()))
    print("sample std median:", float(np.median(sample_std)))
    print("sample std q05/q95:", float(np.quantile(sample_std, 0.05)), float(np.quantile(sample_std, 0.95)))
    print("channel std mean:", float(channel_std.mean()))

    if y is not None:
        labels, counts = np.unique(y, return_counts=True)
        print("labels:", dict(zip(labels.tolist(), counts.tolist())))


def compare_two(name_a, x_a, name_b, x_b):
    flat_a = x_a.reshape(x_a.shape[0], -1)
    flat_b = x_b.reshape(x_b.shape[0], -1)

    sample_std_a = flat_a.std(axis=1)
    sample_std_b = flat_b.std(axis=1)
    channel_std_a = x_a.std(axis=(0, 2))
    channel_std_b = x_b.std(axis=(0, 2))

    std_ratio = float(x_b.std() / x_a.std())
    sample_std_ratio = float(np.median(sample_std_b) / np.median(sample_std_a))
    channel_corr = float(np.corrcoef(channel_std_a, channel_std_b)[0, 1])

    print("\n[" + name_a + " vs " + name_b + "]")
    print("global std ratio:", std_ratio)
    print("sample std median ratio:", sample_std_ratio)
    print("channel std correlation:", channel_corr)


def main():
    x_train, y_train = read_x("train.h5")
    x_val, y_val = read_x("val.h5")
    x_test, _ = read_x("test_x_only.h5")

    show_basic_info("train", x_train, y_train)
    show_basic_info("validation", x_val, y_val)
    show_basic_info("test_x_only", x_test)

    compare_two("train", x_train, "validation", x_val)
    compare_two("train", x_train, "test_x_only", x_test)
    compare_two("validation", x_val, "test_x_only", x_test)


if __name__ == "__main__":
    main()


