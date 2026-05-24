# 完整实验脚本说明

本目录保存项目中的主要训练、融合、基线和审计脚本，用于说明完整实验过程。

最终结果的最短复现入口在：

```text
../code/final_fusion.py
../code/check_submit.py
```

本目录下脚本多数依赖原始 HDF5 数据、历史实验输出或中间缓存，因此不建议在没有完整数据环境时逐个运行。

## 目录说明

```text
train/
```

训练相关脚本，包括 raw EEG 模型训练、预处理候选训练和辅助工具。

```text
fusion/
```

特征融合、概率融合、稳定融合和最终融合搜索脚本。

```text
baseline/
```

基线模型脚本，包括 raw 1D-CNN 和 clean feature MLP。

```text
audit/
```

数据检查、最终结果可靠性检查、hidden test 风险检查和融合稳定性分析脚本。

```text
external/
```

外部数据搜索、格式转换和外部数据接入相关脚本。

## 推荐阅读顺序

1. `fusion/feature_fusion_v25.py`
2. `fusion/stable_fusion_search.py`
3. `audit/hidden_test_risk_audit.py`
4. `audit/final_fusion_reliability_audit.py`
5. `baseline/raw_1dcnn_baseline.py`
6. `baseline/clean_feature_mlp_baseline.py`
