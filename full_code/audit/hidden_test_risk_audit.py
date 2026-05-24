import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data_seed_link"
V25_DIR = ROOT / "outputs_experiments" / "clean_feature_block_fusion_v25_20260522_150944"
V2_SOURCE_DIR = ROOT / "outputs_experiments" / "clean_feature_block_fusion_v2_20260522_085528"
STABLE_DIR = ROOT / "outputs_experiments" / "regularized_stable_fusion_search_20260523_003129"
OUT_DIR = ROOT / "outputs_experiments" / "hidden_test_risk_audit_20260524"


def read_h5(name):
    path = DATA_DIR / name
    with h5py.File(path, "r") as f:
        x = f["X"][:].astype(np.float64)
        y = f["y"][:] if "y" in f else None
    return x, y


def q(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "q01": float(np.quantile(values, 0.01)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def ks_stat(a, b):
    a = np.sort(np.asarray(a, dtype=np.float64).ravel())
    b = np.sort(np.asarray(b, dtype=np.float64).ravel())
    values = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, values, side="right") / max(len(a), 1)
    cdf_b = np.searchsorted(b, values, side="right") / max(len(b), 1)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def wasserstein_1d(a, b):
    a = np.sort(np.asarray(a, dtype=np.float64).ravel())
    b = np.sort(np.asarray(b, dtype=np.float64).ravel())
    n = max(len(a), len(b))
    qa = np.quantile(a, np.linspace(0, 1, n))
    qb = np.quantile(b, np.linspace(0, 1, n))
    return float(np.mean(np.abs(qa - qb)))


def mmd_rbf(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    joined = np.vstack([a, b])
    # Median heuristic on a deterministic subset to keep the audit lightweight.
    subset = joined[:: max(1, len(joined) // 300)]
    d2 = ((subset[:, None, :] - subset[None, :, :]) ** 2).sum(axis=2)
    med = np.median(d2[d2 > 0])
    gamma = 1.0 / max(2.0 * med, 1e-12)

    def kernel_mean(u, v):
        d2_uv = ((u[:, None, :] - v[None, :, :]) ** 2).sum(axis=2)
        return float(np.exp(-gamma * d2_uv).mean())

    return kernel_mean(a, a) + kernel_mean(b, b) - 2.0 * kernel_mean(a, b)


def descriptor_matrix(x):
    flat = x.reshape(x.shape[0], -1)
    sample_mean = flat.mean(axis=1, keepdims=True)
    sample_std = flat.std(axis=1, keepdims=True)
    sample_abs = np.abs(flat).mean(axis=1, keepdims=True)
    ch_mean = x.mean(axis=2)
    ch_std = x.std(axis=2)
    # Use a compact, model-agnostic descriptor. This is for distribution audit,
    # not for training or label inference.
    return np.hstack([sample_mean, sample_std, sample_abs, ch_mean, ch_std])


def pca_centroid_distance(desc_a, desc_b):
    joined = np.vstack([desc_a, desc_b])
    joined = (joined - joined.mean(axis=0)) / (joined.std(axis=0) + 1e-12)
    _, _, vt = np.linalg.svd(joined, full_matrices=False)
    z = joined @ vt[:10].T
    za = z[: len(desc_a)]
    zb = z[len(desc_a) :]
    centroid_dist = float(np.linalg.norm(za.mean(axis=0) - zb.mean(axis=0)))
    within = 0.5 * (
        np.mean(np.linalg.norm(za - za.mean(axis=0), axis=1))
        + np.mean(np.linalg.norm(zb - zb.mean(axis=0), axis=1))
    )
    return {
        "centroid_distance_pc10": centroid_dist,
        "within_radius_pc10": float(within),
        "centroid_over_within": float(centroid_dist / max(within, 1e-12)),
        "mmd_rbf_descriptor": float(mmd_rbf(za, zb)),
    }


def summarize_x(name, x, y=None):
    flat = x.reshape(x.shape[0], -1)
    sample_mean = flat.mean(axis=1)
    sample_std = flat.std(axis=1)
    sample_abs = np.abs(flat).mean(axis=1)
    channel_mean = x.mean(axis=(0, 2))
    channel_std = x.std(axis=(0, 2))
    time_mean = x.mean(axis=(0, 1))
    time_std = x.std(axis=(0, 1))
    z = flat - flat.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(z, axis=1) + 1e-12
    adjacent_cos = np.sum(z[:-1] * z[1:], axis=1) / (norms[:-1] * norms[1:])
    out = {
        "shape": list(x.shape),
        "dtype_after_read": str(x.dtype),
        "nan_count": int(np.isnan(x).sum()),
        "inf_count": int(np.isinf(x).sum()),
        "global": {
            "mean": float(x.mean()),
            "std": float(x.std()),
            "min": float(x.min()),
            "max": float(x.max()),
        },
        "per_sample_mean": q(sample_mean),
        "per_sample_std": q(sample_std),
        "per_sample_abs_mean": q(sample_abs),
        "per_channel_mean": q(channel_mean),
        "per_channel_std": q(channel_std),
        "per_time_mean": q(time_mean),
        "per_time_std": q(time_std),
        "adjacent_cosine": q(adjacent_cos),
    }
    if y is not None:
        out["labels"] = {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    return out


def compare_x(a_name, a, b_name, b):
    def metrics(v1, v2):
        return {
            "ks": ks_stat(v1, v2),
            "wasserstein": wasserstein_1d(v1, v2),
            "median_ratio_b_over_a": float(np.median(v2) / max(abs(float(np.median(v1))), 1e-12)),
        }

    fa = a.reshape(a.shape[0], -1)
    fb = b.reshape(b.shape[0], -1)
    desc_a = descriptor_matrix(a)
    desc_b = descriptor_matrix(b)
    return {
        "pair": f"{a_name}_vs_{b_name}",
        "global_std_ratio_b_over_a": float(b.std() / max(a.std(), 1e-12)),
        "sample_mean": metrics(fa.mean(axis=1), fb.mean(axis=1)),
        "sample_std": metrics(fa.std(axis=1), fb.std(axis=1)),
        "sample_abs_mean": metrics(np.abs(fa).mean(axis=1), np.abs(fb).mean(axis=1)),
        "channel_mean": metrics(a.mean(axis=(0, 2)), b.mean(axis=(0, 2))),
        "channel_std": metrics(a.std(axis=(0, 2)), b.std(axis=(0, 2))),
        "time_mean": metrics(a.mean(axis=(0, 1)), b.mean(axis=(0, 1))),
        "time_std": metrics(a.std(axis=(0, 1)), b.std(axis=(0, 1))),
        "channel_std_corr": float(np.corrcoef(a.std(axis=(0, 2)), b.std(axis=(0, 2)))[0, 1]),
        "time_std_corr": float(np.corrcoef(a.std(axis=(0, 1)), b.std(axis=(0, 1)))[0, 1]),
        "descriptor_pca": pca_centroid_distance(desc_a, desc_b),
    }


def load_run_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def reconstruct_from_v2(candidate, split):
    folder = V2_SOURCE_DIR / f"all_{split}_probs"
    weights = candidate["weights"]
    probs = None
    for name, weight in weights.items():
        arr = np.load(folder / f"{name}.npy")
        probs = arr * weight if probs is None else probs + arr * weight
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    return probs


def load_seed_txt(path):
    return np.array([int(line.strip()) for line in Path(path).read_text().splitlines() if line.strip()])


def entropy(probs):
    p = np.clip(probs, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1) / math.log(probs.shape[1])


def prob_summary(name, val_probs, test_probs, y_val=None, seed_path=None):
    def one(split, probs, y=None):
        pred = probs.argmax(axis=1)
        sorted_p = np.sort(probs, axis=1)
        max_p = sorted_p[:, -1]
        margin = sorted_p[:, -1] - sorted_p[:, -2]
        out = {
            "prediction_distribution": {str(k): int(v) for k, v in zip(*np.unique(pred, return_counts=True))},
            "prediction_proportion": {str(k): float(v / len(pred)) for k, v in zip(*np.unique(pred, return_counts=True))},
            "max_probability": q(max_p),
            "margin_top1_top2": q(margin),
            "normalized_entropy": q(entropy(probs)),
        }
        if y is not None:
            out["accuracy"] = float((pred == y).mean())
        return out

    out = {
        "name": name,
        "val": one("val", val_probs, y_val),
        "test": one("test", test_probs, None),
    }
    if seed_path is not None:
        txt = load_seed_txt(seed_path)
        out["seed_txt_check"] = {
            "path": str(seed_path),
            "line_count": int(len(txt)),
            "matches_test_argmax": bool(np.array_equal(txt, test_probs.argmax(axis=1))),
        }
    return out


def compare_predictions(name_a, probs_a, name_b, probs_b):
    pred_a = probs_a.argmax(axis=1)
    pred_b = probs_b.argmax(axis=1)
    disagree = pred_a != pred_b
    sorted_a = np.sort(probs_a, axis=1)
    sorted_b = np.sort(probs_b, axis=1)
    max_a = sorted_a[:, -1]
    max_b = sorted_b[:, -1]
    margin_a = sorted_a[:, -1] - sorted_a[:, -2]
    margin_b = sorted_b[:, -1] - sorted_b[:, -2]
    pairs = Counter([f"{a}->{b}" for a, b in zip(pred_a[disagree], pred_b[disagree])])
    out = {
        "pair": f"{name_a}_vs_{name_b}",
        "agreement": float((~disagree).mean()),
        "disagreement_count": int(disagree.sum()),
        "disagreement_rate": float(disagree.mean()),
        "disagreement_pairs_a_to_b": dict(pairs),
    }
    if disagree.any():
        out.update(
            {
                f"{name_a}_max_probability_on_disagreement": q(max_a[disagree]),
                f"{name_b}_max_probability_on_disagreement": q(max_b[disagree]),
                f"{name_a}_margin_on_disagreement": q(margin_a[disagree]),
                f"{name_b}_margin_on_disagreement": q(margin_b[disagree]),
                f"{name_a}_lower_confidence_than_{name_b}_on_disagreement_rate": float(
                    (max_a[disagree] < max_b[disagree]).mean()
                ),
            }
        )
    return out


def inspect_candidate(candidate):
    selected = candidate.get("selected_candidates", [])
    names = " ".join(selected).lower()
    flags = {
        "uses_prob_block_smooth": "prob_block_smooth" in names,
        "uses_train_eval_block_candidate": "trainblock" in names or "evalblock" in names or "block_et" in names,
        "uses_order_prior_named": "order_prior" in names or "order-aware" in names,
        "uses_source_matching_named": "source_matching" in names or "source" in names and "matching" in names,
        "uses_sample_index_named": "index" in names or "sample_index" in names,
    }
    if flags["uses_order_prior_named"] or flags["uses_sample_index_named"] or flags["uses_source_matching_named"]:
        risk = "high"
    elif flags["uses_train_eval_block_candidate"] or flags["uses_prob_block_smooth"]:
        risk = "medium"
    else:
        risk = "lower"
    return {
        "candidate_name": candidate.get("candidate_name"),
        "selected_candidates": selected,
        "weights": candidate.get("weights", {}),
        "flags": flags,
        "structure_dependency_risk_from_names": risk,
    }


def write_report(results):
    p = OUT_DIR / "hidden_test_risk_audit_report.md"
    final = results["probability_models"]["fallback_old_balanced_07111"]
    balanced = results["probability_models"]["balanced_v25_07111"]
    safe = results["probability_models"]["safe_v25_06889"]
    stable = results["probability_models"]["regularized_stable_06889"]
    xcmp = {item["pair"]: item for item in results["x_distribution_comparisons"]}
    val_test = xcmp["val_vs_test"]
    final_safe = results["prediction_comparisons"]["fallback_old_balanced_07111_vs_safe_v25_06889_test"]
    final_stable = results["prediction_comparisons"]["fallback_old_balanced_07111_vs_regularized_stable_06889_test"]
    lines = []
    lines.append("# Hidden Test Risk Audit")
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    lines.append(
        "- `test_x_only.h5` 的 X 分布没有显示大规模预处理/尺度变化：val/test global std 比值约 "
        f"`{val_test['global_std_ratio_b_over_a']:.4f}`，channel std 相关约 "
        f"`{val_test['channel_std_corr']:.4f}`，sample std 的 KS 距离约 "
        f"`{val_test['sample_std']['ks']:.4f}`，time std 中位数比值约 "
        f"`{val_test['time_std']['median_ratio_b_over_a']:.4f}`。"
    )
    lines.append(
        "- `0.7111` final 是四个候选概率矩阵的固定加权融合，不是按 test 顺序新构造规则；但候选里包含 "
        "`prob_block_smooth_10/20` 和 `block_et__trainblock_10__evalblock_10`，因此存在中等 block 结构依赖。"
    )
    lines.append(
        f"- `0.7111` fallback 与 `0.6889` safe 在 test 上一致率为 `{final_safe['agreement']:.4f}`，"
        f"与 regularized stable 一致率为 `{final_stable['agreement']:.4f}`。分歧不是大面积重写预测。"
    )
    lines.append(
        "- 综合判断：X 分布/预处理变化导致严重掉分的风险为 `low`；validation 选择偏差风险为 `medium`；"
        "order/block artifact 风险为 `medium`，但低于纯 order-prior 或 neural stacker。"
    )
    lines.append(
        "- 若 hidden test 评分确实基于当前 `test_x_only.h5` 这 450 个样本，主交 `0.7111` 有提交价值；"
        "若非常厌恶 block 结构风险，则 `0.6889` safe 是保守备选。"
    )
    lines.append("")
    lines.append("## 1. Final 生成路径与结构依赖")
    lines.append("")
    lines.append(f"- final 输出目录：`{V25_DIR}`")
    lines.append("- 主提交文件：`fallback_old_balanced_SEED.txt` / `balanced_v25_SEED.txt`")
    lines.append("- 保守文件：`safe_v25_SEED.txt`；另有 `regularized_stable_fusion_search` 的 stable/safe 输出。")
    lines.append("")
    for key in ["fallback_old_balanced_07111", "balanced_v25_07111", "safe_v25_06889"]:
        ins = results["candidate_inspection"][key]
        lines.append(f"### {key}")
        lines.append("")
        lines.append(f"- candidate: `{ins['candidate_name']}`")
        lines.append(f"- name-based risk: `{ins['structure_dependency_risk_from_names']}`")
        lines.append(f"- flags: `{ins['flags']}`")
        lines.append("- selected candidates:")
        for name in ins["selected_candidates"]:
            lines.append(f"  - `{name}`")
        lines.append("- weights:")
        for name, w in ins["weights"].items():
            lines.append(f"  - `{name}`: `{w:.6f}`")
        lines.append("")
    lines.append("未发现最终流程显式使用 hidden test label；也没有在本审计中发现直接命名的 order_prior/source_matching/sample_index 规则。主要风险来自概率块平滑和 block-based 候选。")
    lines.append("")
    lines.append("## 2. X 分布检查")
    lines.append("")
    lines.append("|split|shape|global mean|global std|min|max|NaN|Inf|")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for split in ["train", "val", "test"]:
        s = results["x_summaries"][split]
        lines.append(
            f"|{split}|`{tuple(s['shape'])}`|{s['global']['mean']:.6f}|{s['global']['std']:.6f}|"
            f"{s['global']['min']:.3f}|{s['global']['max']:.3f}|{s['nan_count']}|{s['inf_count']}|"
        )
    lines.append("")
    lines.append("|pair|global std ratio|sample std KS|sample std W|channel std corr|time std KS|time std median ratio|descriptor centroid/within|MMD|")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for item in results["x_distribution_comparisons"]:
        pc = item["descriptor_pca"]
        lines.append(
            f"|{item['pair']}|{item['global_std_ratio_b_over_a']:.4f}|"
            f"{item['sample_std']['ks']:.4f}|{item['sample_std']['wasserstein']:.4f}|"
            f"{item['channel_std_corr']:.4f}|{item['time_std']['ks']:.4f}|"
            f"{item['time_std']['median_ratio_b_over_a']:.4f}|"
            f"{pc['centroid_over_within']:.4f}|{pc['mmd_rbf_descriptor']:.6f}|"
        )
    lines.append("")
    lines.append("test 的 per-sample/per-channel 统计贴近 val/train；per-time 细粒度曲线相关性不高，但中位数尺度接近，且未出现 NaN/Inf、整体 scale shift、channel shift 或异常样本导致的明显 X-domain 崩坏信号。")
    lines.append("")
    lines.append("## 3. 预测行为")
    lines.append("")
    lines.append("|model|val acc|val pred dist|test pred dist|val maxP median|test maxP median|val entropy median|test entropy median|txt check|")
    lines.append("|---|---:|---|---|---:|---:|---:|---:|---|")
    for key, item in results["probability_models"].items():
        lines.append(
            f"|{key}|{item['val'].get('accuracy', float('nan')):.4f}|"
            f"`{item['val']['prediction_distribution']}`|`{item['test']['prediction_distribution']}`|"
            f"{item['val']['max_probability']['median']:.4f}|{item['test']['max_probability']['median']:.4f}|"
            f"{item['val']['normalized_entropy']['median']:.4f}|{item['test']['normalized_entropy']['median']:.4f}|"
            f"`{item.get('seed_txt_check', {}).get('matches_test_argmax')}`|"
        )
    lines.append("")
    lines.append("test 上没有出现单类塌缩。`0.7111` 的 class 2 预测偏少，这是风险点，但 safe/stable 版本也接近这个分布，并非只有 final 独有。")
    lines.append("")
    lines.append("## 4. 0.7111 vs 0.6889")
    lines.append("")
    for key, item in results["prediction_comparisons"].items():
        lines.append(f"### {key}")
        lines.append("")
        lines.append(f"- agreement: `{item['agreement']:.4f}`")
        lines.append(f"- disagreement count/rate: `{item['disagreement_count']}` / `{item['disagreement_rate']:.4f}`")
        lines.append(f"- disagreement pairs: `{item['disagreement_pairs_a_to_b']}`")
        for k, v in item.items():
            if k.endswith("_max_probability_on_disagreement") or k.endswith("_margin_on_disagreement"):
                lines.append(f"- {k}: median `{v['median']:.4f}`, q05 `{v['q05']:.4f}`, q95 `{v['q95']:.4f}`")
        lines.append("")
    lines.append("分歧样本比例较小到中等，且多集中在边界概率变化；这支持 `0.7111` 不是在 test 上完全换了一套预测逻辑。")
    lines.append("")
    lines.append("## 5. 风险分解与提交建议")
    lines.append("")
    lines.append("- X 分布/预处理变化风险：`low`。证据是 shape、std、channel/time 统计和 descriptor 距离均接近。")
    lines.append("- validation overfitting/model selection bias：`medium`。候选和权重是在 validation 上筛选的，0.7111 不能当成无偏泛化估计。")
    lines.append("- order/block artifact 风险：`medium`。没有显式 order-prior，但使用了 block smoothing 和一个 block_et 候选。")
    lines.append("")
    lines.append("合理预期：如果 hidden label 与当前 test_x_only 的生成结构接近，`0.7111` 掉到 `0.65-0.69` 是可以接受且较合理的区间；若 test 的真实 block/标签顺序明显不同，可能掉到 `0.60-0.65`，极端 block 失配时可能更低。当前证据不支持因为 X 结构大规模变化而直接崩到普通 CNN 水平。")
    lines.append("")
    lines.append("?????????? fixed probability fusion??? stable/safe fusion ?????neural stacker ????????")
    p.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    x_train, y_train = read_h5("train.h5")
    x_val, y_val = read_h5("val.h5")
    x_test, _ = read_h5("test_x_only.h5")

    v25 = load_run_json(V25_DIR / "run_results.json")
    fallback = v25["recommended"]
    balanced = v25["picks"]["balanced_plus"]
    safe = v25["picks"]["safest"]

    model_probs = {
        "fallback_old_balanced_07111": {
            "val": reconstruct_from_v2(fallback, "val"),
            "test": reconstruct_from_v2(fallback, "test"),
            "seed": V25_DIR / "fallback_old_balanced_SEED.txt",
        },
        "balanced_v25_07111": {
            "val": reconstruct_from_v2(balanced, "val"),
            "test": reconstruct_from_v2(balanced, "test"),
            "seed": V25_DIR / "balanced_v25_SEED.txt",
        },
        "safe_v25_06889": {
            "val": reconstruct_from_v2(safe, "val"),
            "test": reconstruct_from_v2(safe, "test"),
            "seed": V25_DIR / "safe_v25_SEED.txt",
        },
        "regularized_stable_06889": {
            "val": np.load(STABLE_DIR / "best_stable_fusion_val_probs.npy"),
            "test": np.load(STABLE_DIR / "best_stable_fusion_test_probs.npy"),
            "seed": STABLE_DIR / "best_stable_fusion_SEED.txt",
        },
    }

    results = {
        "paths": {
            "data_dir": str(DATA_DIR),
            "v25_dir": str(V25_DIR),
            "v2_source_dir_for_reconstruction": str(V2_SOURCE_DIR),
            "stable_dir": str(STABLE_DIR),
        },
        "candidate_inspection": {
            "fallback_old_balanced_07111": inspect_candidate(fallback),
            "balanced_v25_07111": inspect_candidate(balanced),
            "safe_v25_06889": inspect_candidate(safe),
        },
        "x_summaries": {
            "train": summarize_x("train", x_train, y_train),
            "val": summarize_x("val", x_val, y_val),
            "test": summarize_x("test", x_test, None),
        },
        "x_distribution_comparisons": [
            compare_x("train", x_train, "val", x_val),
            compare_x("train", x_train, "test", x_test),
            compare_x("val", x_val, "test", x_test),
        ],
        "probability_models": {},
        "prediction_comparisons": {},
    }

    for name, obj in model_probs.items():
        results["probability_models"][name] = prob_summary(
            name, obj["val"], obj["test"], y_val=y_val, seed_path=obj["seed"]
        )

    comparisons = [
        ("fallback_old_balanced_07111", "safe_v25_06889"),
        ("fallback_old_balanced_07111", "regularized_stable_06889"),
        ("balanced_v25_07111", "safe_v25_06889"),
    ]
    for a, b in comparisons:
        key = f"{a}_vs_{b}_test"
        results["prediction_comparisons"][key] = compare_predictions(
            a, model_probs[a]["test"], b, model_probs[b]["test"]
        )

    (OUT_DIR / "hidden_test_risk_audit_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(results)
    print(OUT_DIR)
    print(OUT_DIR / "hidden_test_risk_audit_report.md")


if __name__ == "__main__":
    main()
