# SEED EEG Emotion Recognition

本项目用于 SEED 三分类 EEG 情绪识别任务。输入为 EEG 片段，单个样本形状为 `62 x 400`，输出类别为 `0/1/2`。本仓库整理了最终提交文件、结果复现代码、候选模型概率、实验报告、notebook，以及完整探索阶段中使用过的训练、融合、审计和外部数据尝试脚本。

本项目的最终思路不是单纯训练一个模型，而是围绕课程给定的 SEED 数据完成了一套完整的机器学习工程流程：先做数据核验和基线复现，再尝试深度模型、传统 EEG 特征、外部数据、特征融合和概率融合，最后对高分结果进行风险审计，并筛选出最终提交文件。

## 目录结构

```text
code/
  final_fusion.py      最终概率融合脚本
  check_submit.py      提交文件检查脚本
  check_data.py        数据格式与分布检查脚本

submit/
  SEED.txt             最终提交文件
  SEED_safe.txt        保守备选结果
  SEED_balanced.txt    平衡备选结果

probs/
  val/                 验证集候选模型概率
  test/                测试集候选模型概率

notebook/
  SEED_project.ipynb   项目主要流程 notebook

doc/
  report.md            项目方法与结果说明

full_code/
  train/               训练相关脚本
  fusion/              融合相关脚本
  baseline/            基线模型脚本
  audit/               结果检查脚本
  external/            外部数据尝试脚本
```

## 最终提交文件

正式提交文件为：

```text
submit/SEED.txt
```

文件检查结果：

```text
行数: 450
标签范围: 0/1/2
预测分布: 0=190, 1=200, 2=60
```

`submit/SEED_safe.txt` 和 `submit/SEED_balanced.txt` 是保守备选或平衡备选结果，用于和主提交结果进行对照。

## 方法简述

项目首先对官方 HDF5 数据进行格式和分布检查，确认训练集、验证集和测试集的输入形状一致，并检查缺失值、异常值、标签范围、类别分布和基础统计量。课程数据中，训练集为 `900 x 62 x 400`，验证集和测试集各为 `450 x 62 x 400`，标签为 `0/1/2` 三分类。

在模型路线选择上，先尝试了直接使用 raw EEG 的深度模型，包括 CNN、1D-CNN、CRNN、EEGNet / FBCNet 风格模型和 MLP 等。实验发现，在当前小样本设置下，单一 raw EEG 深度模型在 validation 上不够稳定，容易出现类别偏置，部分模型对 neutral 类召回不稳定。

随后项目转向传统 EEG 特征和机器学习分类器。尝试了 PSD、DE、log-bandpower、通道均值与方差、原始信号统计特征、协方差矩阵特征、PCA 降维特征、common average reference、per-sample channel z-score 等方法，并结合 SVM、RandomForest、ExtraTrees、HistGradientBoosting 和 MLP 等分类器。传统特征路线整体更稳定，但单一模型仍然存在性能上限。

最终方案采用多模型概率融合。具体做法是：保存多个候选模型在 validation 和 test_x_only 上输出的三分类概率，再对若干较稳定候选概率进行固定权重加权平均，最后对融合概率取 `argmax` 得到最终类别。最终提交版本不是单一模型的直接输出，而是多个候选模型概率的融合结果。

## 探索阶段与迭代过程

### 1. 数据核验与任务边界确认

项目最开始先确认官方数据结构与任务边界。检查了 `train.h5`、`val.h5` 和 `test_x_only.h5` 的 HDF5 键、输入 shape、标签范围、类别数量、缺失值、异常值、loader shuffle、loss 输入、checkpoint 保存和 test 顺序等基本契约，避免后续结果建立在数据读取错误或顺序错乱上。

这一阶段确认：训练集为 900 个样本，验证集和测试集各 450 个样本；单个样本为 62 个 EEG 通道、400 个时间点；训练和验证标签为 `0/1/2`，且类别整体较均衡。这个阶段的重点不是提高准确率，而是保证后续所有实验使用的是正确的数据和一致的任务定义。

### 2. 早期深度学习基线

随后从已有 notebook 和 FBCNet / CNN 类模型出发，建立 raw EEG 深度学习基线。早期深度模型大多可以正常训练，但 validation accuracy 多数停留在约 `0.36-0.44`。这说明直接用小样本 raw EEG 训练深度网络并不稳定，模型容易受到随机种子、类别分布和训练细节影响。

之后继续扩展 CNN 路线，包括数据增强、测试时增强、weighted cross entropy、focal loss、类别校准、多 seed sweep、simple 1D-CNN、CRNN 和 neural MLP baseline 等方法。部分方法可以带来小幅提升，最好推进到约 `0.45-0.46`，但仍然暴露出类别偏置和召回不稳定的问题。

### 3. 传统特征与机器学习模型

在 raw 深度模型效果不够稳定后，项目转向更可解释的 EEG 特征路线。尝试了 PSD、DE、log-bandpower、channel statistics、band features、covariance PCA 等特征，并结合 SVM、RandomForest、ExtraTrees、HistGradientBoosting 和 MLP 等分类器。

这一阶段证明传统特征能够稳定跑通任务，但早期单模型结果仍然有限，验证集表现大约在 `0.41` 左右。随后通过组合 CAR、通道统计、频带特征、协方差 PCA 和不同分类器，形成了多个候选模型，为后续概率融合提供了基础。

### 4. 外部数据与迁移尝试

为了突破官方训练集样本量较小的问题，尝试接入外部 SEED-style raw EEG、feature-only 数据、external pretrain、neutral recovery、normalization alignment、two-stream 模型和 MultiScale CRNN 等路线。

这些路线大多没有稳定提升。主要原因是外部数据与课程官方 split 在归一化尺度、subject/session 来源、窗口切分、标签映射和分布上存在差异，直接 supervised concat 反而可能导致模型偏类或退化到接近随机。因此，外部数据最终没有作为主结果来源，而是作为 domain shift 和失败分析的一部分保留在 `full_code/external/` 中。

### 5. 概率融合与分阶段提升

在单模型效果有限的情况下，项目重点转向多模型概率融合。保存多个候选模型在 validation 和 test_x_only 上的三分类概率，并比较不同特征、预处理方式和分类器组合的验证集表现。

融合阶段尝试了 soft voting、class-wise fusion、teacher distillation、probability pool super fusion、targeted distillation pool expansion 和 regularized stable fusion 等方法。通过逐步扩展候选概率池、调整融合方式并进行稳定性检查，validation 表现从早期单模型水平逐步提升，最终形成 `fallback_old_balanced / balanced_v25` 主提交候选，以及 `safe_v25 / regularized stable fusion` 保守备选。

### 6. Artifact 检查与风险审计

实验中后期发现 validation 中存在明显 block/order 结构。单独使用 order-prior 可以得到很高的 validation 结果，甚至达到 `1.0000`，但这类方法本质上利用的是样本顺序规律，而不是可靠 EEG 分类能力。因此，这些结果只作为 artifact 分析保留，没有作为主提交模型。

随后专门做了 clean low-risk 路线，避免 order prior 和 source matching，得到约 `0.6222` 的较干净结果。最后阶段又对高分结果进行可靠性审计：neural probability stacker 的 OOF 结果较高，但嵌套 holdout 和 block-wise 检查显示明显 validation-overfit，因此只保留为高风险消融，不作为主提交。

最终主提交候选 `fallback_old_balanced / balanced_v25` 在 validation 上约为 `0.7111`。审计中没有发现 hidden test label 泄漏、显式 sample index 规则、source matching 或直接按固定区间人工改标签的问题。但该方案仍包含一定 validation-based model selection 和 block-level 处理，因此被定位为“经过审计但仍带有验证集优化风险”的主提交候选。`safe_v25 / regularized stable fusion` 在 validation 上约为 `0.6889`，作为更保守备选保留。

进一步对 `test_x_only.h5` 的 X 分布进行只读审计后发现，train、validation 和 test_x_only 均为 `(N, 62, 400)` 格式，未发现 NaN/Inf；validation 和 test 的 global std、sample-level std、channel-level std 等统计量接近，说明 test_x_only 没有明显预处理尺度或通道结构变化。主提交结果与保守版本在 test_x_only 上预测一致率较高，说明最终高分版本并没有在 hidden test 上表现出完全不同的激进预测模式。

## 最终版本选择

最终采用 `submit/SEED.txt` 作为主提交文件。该文件由 `code/final_fusion.py` 读取 `probs/test/` 中的候选概率文件并完成固定权重融合后生成。

最终版本的定位是：

```text
经过数据核验、特征工程、多模型融合和风险审计后的高分提交候选。
```

需要说明的是，`submit/SEED.txt` 并不代表一个已经严格证明能泛化到所有 EEG 情绪识别场景的模型，而是在课程给定 split 和 hidden test 设置下，综合 validation 表现、复现性和风险审计结果后选择的提交版本。项目同时保留 `SEED_safe.txt` 作为更保守结果，用于说明主提交并非唯一可运行方案。

## 运行方式

生成最终提交文件：

```bash
python code/final_fusion.py
```

检查提交文件：

```bash
python code/check_submit.py
```

如果本地有原始 HDF5 数据，也可以检查数据格式和分布：

```bash
python code/check_data.py
```

## 文件说明

- `code/final_fusion.py`：读取 `probs/test/` 中的候选概率，融合后生成 `submit/SEED.txt`。
- `code/check_submit.py`：检查提交文件行数、标签范围和类别分布，并重新生成结果进行一致性校验。
- `code/check_data.py`：检查 train、validation 和 test_x_only 的数据形状与基础统计量。
- `notebook/SEED_project.ipynb`：整理项目主要流程。
- `doc/report.md`：说明实验流程、模型方法、最终结果和风险审计。
- `full_code/`：保存实验过程中使用的训练、融合、分析、外部数据尝试和审计脚本。

## 数据说明

仓库不包含原始 HDF5 数据和大型中间缓存。运行 `check_data.py` 时，需要将课程提供的原始数据放在代码指定的数据目录中。

`probs/` 目录中保存的是复现最终融合结果所需的候选概率文件，不包含 hidden test 标签。

## 总结

整体来看，本项目完成的不只是一个最终预测文件，而是一条较完整的实验链：先确认数据可信，再建立传统和深度基线，随后尝试模型增强、损失函数、校准、蒸馏、外部数据、特征工程、概率池、块平滑和集成，最后反向审计哪些高分可能来自真实模式，哪些可能来自验证集结构或选择偏差。

失败尝试主要集中在 raw 深度模型小样本不稳、外部数据域不匹配、order prior 泛化风险高、neural stacker 验证集过拟合等方面。最终结果是在“分数导向”和“可靠性导向”之间做出的折中：主提交选择 validation 表现最高且审计后风险可接受的融合版本，同时保留更保守的 safe fusion 作为备选。
