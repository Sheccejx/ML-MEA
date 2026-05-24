# SEED EEG 情绪识别项目

本仓库用于提交 SEED 三分类 EEG 情绪识别任务的代码、说明文档和预测结果。

## 目录结构

```text
code/
  final_fusion.py      最终概率融合脚本
  check_data.py        数据格式与分布检查脚本
  check_submit.py      提交文件检查脚本

doc/
  report.md            项目方法与结果说明

notebook/`r`n  SEED_project.ipynb   项目流程 notebook`r`n`r`nfull_code/`r`n  README.md            完整实验脚本说明`r`n  train/               训练相关脚本`r`n  fusion/              融合相关脚本`r`n  baseline/            基线模型脚本`r`n  audit/               审计检查脚本`r`n  external/            外部数据相关脚本`r`n`r`nprobs/
  val/                 验证集候选模型概率
  test/                测试集候选模型概率

submit/
  SEED.txt             最终提交文件
  SEED_safe.txt        保守备选结果
  SEED_balanced.txt    平衡备选结果
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

## 方法简述

整体工作流程并非训练单一模型，而是围绕 SEED 三分类 EEG 情绪识别任务，经历了从数据核验、基线复现、模型改进、外部数据尝试、概率融合、风险审计到最终提交文件筛选的一整套迭代流程。最开始确认官方数据结构与任务边界：训练集为 900 x 62 x 400，验证集和测试集各 450 x 62 x 400，标签为 0/1/2 且三类均衡，同时检查了 HDF5 键、标签映射、test 顺序、loader shuffle、loss 输入、checkpoint 保存等基本契约，避免后续结果建立在数据读取错误或顺序错乱上。随后从已有 notebook 和 FBCNet/CNN 类模型出发，做了早期深度学习基线，验证准确率大多停留在 0.36-0.44，说明直接用小样本 raw EEG 训练深度网络并不稳定；接着尝试传统 PSD、DE、log-bandpower、SVM、soft voting 等可解释特征方法，结果约 0.41，证明传统特征能跑通但表达能力有限。之后系统扩展 CNN 路线，包括数据增强、TTA、weighted CE、focal loss、类别校准、多 seed sweep 等，最好推进到约 0.45-0.46，但也暴露出模型容易偏向某些类别、尤其 neutral 类召回不稳定的问题。为了突破单模型上限，转向特征融合和概率融合：构造 CAR、channel statistics、band features、covariance PCA、HistGradientBoosting、RandomForest、MLP 等多种候选，再做 class-wise fusion、teacher distillation、probability pool super fusion、targeted distillation pool expansion，分阶段把验证结果从 0.4667、0.4689、0.4822 推到约 0.4911/0.4933。同时也尝试接入外部 SEED-style raw EEG、feature-only 数据、external pretrain、neutral recovery、normalization alignment、two-stream 模型和 MultiScale CRNN，但这些路线大多失败或没有稳定提升，主要原因是外部数据与课程官方 split 在归一化尺度、subject/session 来源、窗口切分、标签映射和分布上存在差异，直接 supervised concat 反而导致模型偏类或退化到接近随机，因此外部数据最终更多作为失败分析和可拓展方向，而不是主结果来源。中后期发现验证集存在明显 block/order 结构，order-prior 可以得到 1.0000，order-aware soft prior 也能推到很高，但这类结果本质上利用了样本顺序规律，不是可靠 EEG 分类器，因此被标记为高风险或 artifact 分析；随后又专门做了 clean low-risk 路线，避免 order prior 和 source matching，形成了 0.6222 左右的较干净结果。最后进一步扩展 v2/v25、regularized stable fusion、neural MLP baseline、raw-like EEG feasibility、simple 1D-CNN、neural probability stacker，并对高分结果做可靠性审计：神经 stacker 的 OOF 可达 0.7667，但嵌套 holdout 和 block-wise 检查显示明显 validation-overfit，只保留为高风险消融；fallback_old_balanced / balanced_v25 的 0.7111 不是明显 bug，也没有发现 test label 泄漏，但其权重和候选选择仍依赖同一验证集，block-wise selection-holdout 会下降，因此最终被表述为“经过审计但验证集优化”的主提交候选，而 safe_v25 或 regularized stable fusion 约 0.6889 作为更保守备选。整体来看，本项目完成的不只是一个最终文件，而是一条完整实验链：先确认数据可信，再建立传统和深度基线，逐步尝试增强、损失函数、校准、蒸馏、外部数据、图模型、特征工程、概率池、块平滑和集成，最后再反向审计哪些高分来自真实模式、哪些可能来自验证集结构或选择偏差；失败尝试主要集中在 raw 深度模型小样本不稳、外部数据域不匹配、order prior 泛化风险高、neural stacker 验证集过拟合，而最终结果则是在“分数导向”和“可靠性导向”之间做出的折中。

## 文件说明

- `code/final_fusion.py`：读取 `probs/test/` 中的候选概率，按固定权重融合并生成 `SEED.txt`。
- `code/check_data.py`：检查 train、validation、test_x_only 的形状、均值、标准差、NaN/Inf 和通道统计。
- `code/check_submit.py`：检查提交文件行数、标签范围、类别分布，并用包内概率文件重新生成最终提交结果进行校验。
- `doc/report.md`：说明实验流程、最终模型、风险分析和提交文件。
- `notebook/SEED_project.ipynb`：展示数据检查、概率融合和提交文件检查流程。

## 运行方式

```bash
python code/final_fusion.py
python code/check_submit.py
```

如需检查原始 HDF5 数据分布：

```bash
python code/check_data.py
```

## 不包含的数据

仓库不包含原始 HDF5 数据和大型训练缓存。以下文件不应上传：

```text
*.h5
*.zip
feature_cache/
outputs_experiments/
data_seed_link/
__pycache__/
```

