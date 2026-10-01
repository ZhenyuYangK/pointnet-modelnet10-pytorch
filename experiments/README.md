# 实验记录

已完成训练流程诊断、完整 baseline 训练与官方测试集评估。数据增强组合训练与验证对比已完成，增强模型的官方测试评估待进行。每次运行独立记录，不覆盖历史结果。

## 2026-10-01：单 batch 过拟合诊断

- 实验编号：`overfit_20261001_seed42`。
- 目的：检查模型能否记住固定的一批样本，以及权重能否正确保存和重新加载。
- 数据：仅使用官方训练集，每类随机选 2 个模型，共 20 个样本；采样后固定张量，不重复采样，无增强。
- 模型：基础 PointNet，max pooling，Dropout 设为 0，保留 BatchNorm。
- 设置：seed 42，256 点，batch size 20，Adam，学习率 0.001，200 次参数更新。
- 设备：CPU，4 个计算线程；本次进程未检测到可用 CUDA。
- 同批 eval loss：2.303629 → 0.001788；同批 eval 准确率：10% → 100%。
- 读取、训练及权重检查耗时：约 13.04 秒，不含绘图时间。
- 验收：同批 eval 准确率 100%、loss < 0.05 且下降；权重重新加载后的输出一致，均通过。
- 验证集准确率 / 测试准确率：本实验不评估。
- [指标、样本名单及完整训练记录](../results/metrics/overfit_20261001_seed42.json)。
- [loss 和准确率曲线](../results/figures/overfit_20261001_seed42.png)。
- 本地权重：`checkpoints/overfit_20261001_seed42/model.pth`，不纳入 Git。
- 观察：训练模式下的准确率更早达到 100%；评估模式的曲线前期滞后，随后也达到 100%。BatchNorm 在两种模式下的统计量不同，因此需要同时检查最终 eval 结果。
- 结论：训练更新、固定样本学习和权重保存流程已跑通；这一结果不能代表模型在未见样本上的识别能力。

复现：`python scripts/overfit_batch.py --device cpu`。默认生成新名称，保留本次记录。

## 2026-10-01：2 epoch 小规模训练／验证

- 实验编号：`train_small_20261001_seed42`。
- 目的：验收独立验证集、完整 epoch 循环、逐轮日志及最佳权重保存／重载流程。
- 完整划分：只使用官方 train，按类别取约 20% 验证，得到训练 3193、验证 798；没有重叠。
- 实际用量：从上述两边分别每类取 8／2 个，共训练 80、验证 20。官方 test 未参与。
- 模型与设置：基础 PointNet，max pooling，Dropout 0.3，256 点，batch size 8，Adam，学习率 0.001，2 epoch，seed 42。
- 采样：每个网格按固定 seed 采样，缓存在内存；训练逐轮打乱样本顺序，无数据增强。
- 设备：CPU，4 个计算线程，DataLoader worker 0；总运行约 10.01 秒。

| Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.9422 | 31.25% | 2.6403 | 10% |
| 2 | 1.1756 | 63.75% | 2.9860 | 15% |

- 最佳 epoch：2，按验证准确率优先、相同时按较低验证 loss 选择。
- 保存权重重新加载后，在相同验证点云上的 loss／accuracy 与记录一致。
- 结论：训练流程验收通过。验证集仅 20 个样本，准确率 15% 为 3/20；验证 loss 上升，不应视为模型性能已达标。该次实验时尚未开展完整 baseline 和最终测试。
- [指标与配置](../results/metrics/train_small_20261001_seed42.json)、[完整及实际划分名单](../results/metrics/train_small_20261001_seed42_split.json)、[曲线](../results/figures/train_small_20261001_seed42.png)。
- 本地权重：`checkpoints/train_small_20261001_seed42/best_model.pth`，不纳入 Git。

复现（默认生成新实验名称）：

```bash
python train.py --epochs 2 --batch-size 8 --num-points 256 \
  --train-per-class 8 --val-per-class 2
```

## 2026-10-01：完整 baseline（已完成）

- 实验编号：`baseline_20261001_seed42_r2`，RTX 4060 Laptop GPU。
- 基础 PointNet，无 T-Net，max pooling，Dropout 0.3，无数据增强。
- 1024 点，batch size 32，Adam 学习率 0.001，50 epoch，seed 42，DataLoader worker 0。
- 使用完整训练／验证划分 3193／798，文件名单与小规模运行所依据的完整划分一致。
- 按验证准确率选最佳权重，准确率相同时比较验证 loss。官方测试集不参与训练或权重选择，独立评估结果见下一节。
- 已完成全部 50 epoch；每轮实际训练 3193 个、验证 798 个样本。
- 最佳 epoch：49，训练准确率 99.37%，验证准确率 94.99%（758/798），验证 loss 0.179686。
- 最后 epoch 50：训练准确率 98.40%，验证准确率 93.23%，验证 loss 0.279129。
- 总耗时约 342.25 秒；首轮约 274.28 秒，包含首次网格读取与点云缓存。
- 重新加载最佳模型后，验证 loss 与 accuracy 与原记录一致；权重 SHA256 和训练源码文件 SHA256 已记录。
- 观察：训练 loss 整体下降；验证曲线有明显波动，部分轮次出现 loss 峰值。第 50 轮弱于第 49 轮，因此采用按验证集选定的最佳模型。
- 结论：完整训练和权重选择流程完成，94.99% 为验证结果。官方测试结果在下一节单独记录。
- [完整指标](../results/metrics/baseline_20261001_seed42_r2.json)、[划分](../results/metrics/baseline_20261001_seed42_r2_split.json)、[曲线](../results/figures/baseline_20261001_seed42_r2.png)。
- 本地权重：`checkpoints/baseline_20261001_seed42_r2/best_model.pth`，不纳入 Git。
- 本地运行日志：`logs/baseline_20261001_seed42_r2.log`。
- 首次尝试 `baseline_20261001_seed42` 在第 1 个 epoch 完成前中断，没有保存权重；原记录保留为 interrupted，新实验使用相同配置从头开始。

## 2026-10-01：baseline 官方测试集评估

- 实验编号：`test_baseline_20261001_seed42`。
- 固定权重：`baseline_20261001_seed42_r2` 的第 49 轮，按验证表现预先选定。
- 数据：全部 908 个官方测试样本，沿用训练配置的 1024 点、seed 42；每个模型固定采样一次，无增强或多次投票。
- 设备与耗时：RTX 4060 Laptop GPU，batch size 32，约 52.28 秒，包含读取网格和绘图。
- 总准确率：817／908 = 89.97797%（报告保留两位小数为 89.98%）。
- 平均类别准确率：89.80233%；测试交叉熵：0.402902。
- 每类准确率：bathtub 94.00%、bed 98.00%、chair 100.00%、desk 81.40%、dresser 89.53%、monitor 96.00%、night_stand 72.09%、sofa 97.00%、table 76.00%、toilet 94.00%。
- 主要混淆：table → desk 23，desk → table 8，night_stand → dresser 17，dresser → night_stand 8；这两对类别占 56／91 个错误。
- 核验：测试文件名单、CSV 的 908 条唯一预测、矩阵行合计及对角线正确数一致；权重哈希匹配，模型状态和权重文件在评估前后保持不变。
- [完整指标](../results/metrics/test_baseline_20261001_seed42.json)、[逐样本预测](../results/metrics/test_baseline_20261001_seed42_predictions.csv)、[混淆矩阵](../results/figures/test_baseline_20261001_seed42_confusion_matrix.png)、[点云预测示例](../results/figures/test_baseline_20261001_seed42_examples.png)。
- 结论：baseline 全流程已完成。当前测试结果固定归属于这组预先选定的权重；后续实验预先定义方案，通过验证集调参与选权重。

复现：

```bash
python test.py --checkpoint checkpoints/baseline_20261001_seed42_r2/best_model.pth --device cuda
```

## 2026-10-01：数据增强组合对比（方案预先固定）

- 实验编号：`augmentation_20261001_seed42`；参照 `baseline_20261001_seed42_r2`。
- 目的：研究缩放、平移、抖动组合对未增强验证集的识别效果。
- 唯一实验因素：训练时开启 `scale_shift_jitter`；每个物体依次进行整体等比例缩放 U(0.8, 1.2)、逐轴平移 U(-0.1, 0.1)、逐点逐坐标高斯噪声（标准差 0.01，截断到 ±0.05）。增强后不再归一化，否则会抵消缩放和平移。
- 点云先固定采样、归一化并缓存，再在每个训练 batch 上实时增强；使用非原地运算保护原始点云，标签不变。验证与测试均不增强。
- 固定条件：基础 PointNet，无 T-Net，max pooling，Dropout 0.3，1024 点，batch size 32，Adam 学习率 0.001，50 epoch，seed 42，worker 0，原训练／验证名单 3193／798。
- 增强使用独立随机数生成器 seed 42，不消耗模型 Dropout 或 DataLoader 的随机数流。
- 选择标准：最高验证准确率，相同则选最低验证 loss。训练结束后与固定 baseline 比较，不根据测试成绩改变增强参数。
- 本次仅运行上述一组完整训练；它衡量三种增强的组合效果，不能归因于单独一种方法，也不能代表多个随机种子的平均效果。
- 当前状态：已完成 50 轮训练；每轮训练 3193 个、验证 798 个。RTX 4060 Laptop GPU，总耗时 355.85 秒，首轮含网格读取与缓存耗时 281.79 秒。
- 最佳权重：第 20 轮，增强训练准确率 96.30%，固定验证准确率 95.86466%（765/798），验证 loss 0.106661。
- 对比 baseline：第 49 轮验证 758/798 = 94.98747%；增强组合净多答对 7 个，增加 0.87719 个百分点（报告取 0.88）。
- 最后第 50 轮验证准确率 94.24%，仍保存按验证标准选出的第 20 轮权重。
- 最佳权重重载后验证 loss／accuracy 与保存时一致；对比脚本确认配置、划分、模型及数据源码、设备与 PyTorch 版本一致。
- 权重：`checkpoints/augmentation_20261001_seed42/best_model.pth`；SHA256 `80c3c54249f09eb8f823379884e07a25b32a8396de961ab29910c2751611c1ef`。
- [完整训练指标](../results/metrics/augmentation_20261001_seed42.json)、[划分](../results/metrics/augmentation_20261001_seed42_split.json)、[训练曲线](../results/figures/augmentation_20261001_seed42.png)、[增强示意图](../results/figures/augmentation_preview.png)。
- [对比数据](../results/metrics/augmentation_comparison_20261001_seed42.json)、[验证曲线对比](../results/metrics/augmentation_comparison_20261001_seed42.png)。
- 结论：当前种子与划分下，三种增强的组合提高了最佳验证准确率；两组曲线均存在波动，不能断言每一轮都提升，也不能据此确定单种增强的贡献。
- 验证范围：运行了完整实验及训练流程内置的权重重载验证；本阶段未新增或运行单元测试。
- 官方测试准确率：尚未评估，留到固定权重后单独进行。

复现（自动生成新名称）：

```bash
python train.py --augmentation scale_shift_jitter --device cuda
python scripts/visualize_augmentation.py --output results/figures/augmentation_preview_new.png
```

## 记录模板

- 实验编号与日期：
- 实验目的：
- 模型与配置：
- 数据划分与随机种子：
- 点数、batch size、学习率、epoch 数：
- 数据增强与 Pooling 方式：
- 最佳验证准确率：
- 测试准确率：
- 训练耗时：
- 权重、指标与图表路径：
- 观察与结论：
