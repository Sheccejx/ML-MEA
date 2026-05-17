SEED 实验记录

--------------------------------------------------

0. 基本情况

SEED
三分类
输入大概是 62 channels × 400 time points
目前主要用 EEGNetClassifier

--------------------------------------------------

1. 最开始的版本（原始 EEGNet baseline）

f1 = 8
d = 2
pk1 = 4
pk2 = 8
dropout = 0.5
norm = Identity
loss = CrossEntropyLoss
optimizer = Adam
lr = 1e-3
weight_decay = 1e-4

结果：
best val acc 0.4378
final val acc 0.4156

仅略高于随机三分类

--------------------------------------------------

2. 加入训练记录功能

train acc
best val model
best epoch
best_state
最后 load best_state
保存 SEED.txt 前 assert 数量


--------------------------------------------------

3. 自动读取 shape

读出：
CHANNELS = 62
patch_size = 400
CLASSES = 3

--------------------------------------------------

4. 加标准化 + 调参

加 EEGNormalize：
class EEGNormalize(nn.Module):
    def forward(self, x):
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + 1e-6)
        return x
f1: 16
dropout: 0.3
AdamW
lr: 3e-4
weight_decay: 1e-3
CrossEntropyLoss(label_smoothing=0.05)
加 ReduceLROnPlateau
epochs = 50

结果：
best val acc = 0.3467
best epoch = 22
final val acc = 0.3356

train acc 后面能到 0.54~0.57 左右
val acc 基本一直在 0.33~0.35，接近随机三分类，炸缸
val loss 后面也没有变好
不能作为最后提交版本。

怀疑是标准化把有用信息抹掉了
--------------------------------------------------

5. 回滚，去掉标准化，参数调回原始EEGNet

弃用EEGNormalize
norm = Identity
f1 = 8
dropout = 0.5
optimizer = Adam
lr = 1e-3
weight_decay = 1e-4
loss = CrossEntropyLoss
scheduler = None

结果：

best val acc = 0.4578
best epoch = 36
final val acc = 0.4244
train acc 后期最高到 0.63 左右
val acc 在 0.42~0.45 附近波动
val loss 后期开始升高，最后到 1.16 左右

删除标准化之后效果恢复并略有提升
后期val loss变差，有点过拟合
--------------------------------------------------

6. 降低lr

lr: 1e-3 -> 5e-4
其他保持不变

结果：
best val acc = 0.4267
best epoch = 24
final val acc = 0.3889
train acc 0.55左右
val acc 0.39~0.42
final val acc 0.3889

效果不如v5

--------------------------------------------------

7. 回滚lr，提高dropout

lr = 1e-3
dp = 0.6

结果：
best val acc = 0.4378
best epoch = 38
final val acc = 0.4133
前期val acc从0.33升到0.38
但best val acc仅0.4378
比当前最好0.4578低

没有提升泛化，可能正则太强，模型学习能力被压低
--------------------------------------------------

8. 尝试更温和正则化

dp 调回0.5
weight_decay: 1e-4 -> 5e-4

结果：
best val acc = 0.4400
best epoch = 26
final val acc = 0.4244
val loss 后期明显上升
train acc 上升时 val acc 没有同步提升
loss 曲线里 train loss 和 val loss 分开得比较明显

接近当前最好版本

--------------------------------------------------

9. weight_decay=3e-4，f1=16

结果：
best val acc = 0.4200
best epoch = 36
final val acc = 0.3889

单纯调参已经没意义了，要想要更好的结果就必须改模型架构了

--------------------------------------------------

10. Multi-scale EEGNet

在 EEGNet 前半部分加入多尺度 temporal convolution。
用多个不同时间卷积核提取 EEG 时间特征，
然后把不同分支的特征 concat，再进入后续 spatial filtering 和 classifier。

Best Val Acc: Epoch 33, 0.4556
Final Val Acc: 0.4311

目前更高的验证集准确率，说明多尺度时间特征提取对当前raw EEG输入是有帮助的。
后期仍然存在一定过拟合现象,训练准确率持续上升到约0.65，但验证集准确率在 0.45 左右震荡，验证集loss也有上升趋势,但泛化能力仍然不够稳定。

--------------------------------------------------

11. 尝试更换其他主流EEG模型

前面几版 raw EEG 模型虽然逐步提升，但最高验证集准确率仍只有0.4556。训练过程中还可以看到 train accuracy 持续上升，而 validation accuracy 在 0.45 左右震荡，说明继续只调 dropout、weight decay 等超参数，可能很难继续大幅提升。

因此，后续不再只对原有模型做小幅修改，而是尝试引入 EEG 领域更常用的模型结构，例如 ShallowConvNet、DeepConvNet 或标准 EEGNet。这样做的目的不是直接调用现成 SOTA，而是测试更符合 EEG 信号特点的 temporal convolution、spatial convolution 和 band-power-like feature extraction 是否能更好地处理当前 SEED raw EEG 输入。

Course project 指导文件中没有禁止更换模型，评分维度也包括 model design reasoning、ablation study、failure analysis 和方法改进。因此，只要保留对照实验并说明换模型原因，更换模型是合理的。

--------------------------------------------------

8. 目前结论

现在最重要的是先跑回滚版。
也就是删除标准化，参数回到原始 EEGNet，只保留 best model、train acc、assert 这些记录功能。

第 4 版结果太低，暂时不用。
后面每次只改一个变量，慢慢看哪一个真的有效。
