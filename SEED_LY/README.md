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

项目先对官方 HDF5 数据进行格式检查，然后提取 EEG 统计特征、频带特征和协方差 PCA 特征。分类器包括 HistGradientBoosting、RandomForest、MLP 等。最终结果采用多个候选模型的概率加权融合，并输出测试集预测标签。

`probs/` 中保存的是最终融合所需的候选模型概率文件，体积较小，用于复现 `submit/SEED.txt`。这些文件不包含测试集标签。

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

