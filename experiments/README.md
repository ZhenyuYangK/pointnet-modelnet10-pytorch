# 实验记录

已完成训练流程诊断与完整 baseline 训练；官方测试集评估和对比实验尚未开始。每次运行独立记录，不覆盖历史结果。

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
- 按验证准确率选最佳权重，准确率相同时比较验证 loss。官方测试集保留到下一阶段。
- 已完成全部 50 epoch；每轮实际训练 3193 个、验证 798 个样本。
- 最佳 epoch：49，训练准确率 99.37%，验证准确率 94.99%（758/798），验证 loss 0.179686。
- 最后 epoch 50：训练准确率 98.40%，验证准确率 93.23%，验证 loss 0.279129。
- 总耗时约 342.25 秒；首轮约 274.28 秒，包含首次网格读取与点云缓存。
- 重新加载最佳模型后，验证 loss 与 accuracy 与原记录一致；权重 SHA256 和训练源码文件 SHA256 已记录。
- 观察：训练 loss 整体下降；验证曲线有明显波动，部分轮次出现 loss 峰值。第 50 轮弱于第 49 轮，因此采用按验证集选定的最佳模型。
- 结论：完整训练和权重选择流程完成，94.99% 为验证结果。官方测试结果将在下一阶段单独记录；尚不能将其写成测试准确率。
- [完整指标](../results/metrics/baseline_20261001_seed42_r2.json)、[划分](../results/metrics/baseline_20261001_seed42_r2_split.json)、[曲线](../results/figures/baseline_20261001_seed42_r2.png)。
- 本地权重：`checkpoints/baseline_20261001_seed42_r2/best_model.pth`，不纳入 Git。
- 本地运行日志：`logs/baseline_20261001_seed42_r2.log`。
- 首次尝试 `baseline_20261001_seed42` 在第 1 个 epoch 完成前中断，没有保存权重；原记录保留为 interrupted，新实验使用相同配置从头开始。

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
