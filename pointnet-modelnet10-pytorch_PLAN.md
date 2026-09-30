# PointNet-ModelNet10-PyTorch 项目实现计划书

## 1. 项目名称

**pointnet-modelnet10-pytorch**

项目目标：使用 **PyTorch** 从零实现一个用于三维点云分类的 **PointNet**，在 **ModelNet10** 数据集上完成训练、验证、测试与实验分析，并形成可复现的课程作业项目。

---

# 2. 项目最终目标

完成一个完整、可复现、结构清晰的深度学习项目，能够：

1. 下载并预处理 ModelNet10 数据集；
2. 将三维模型转换/采样为固定数量的点云；
3. 使用 PyTorch 实现 PointNet 分类网络；
4. 完成训练、验证和测试；
5. 输出训练曲线、测试准确率和混淆矩阵；
6. 可视化部分点云样本及预测结果；
7. 完成至少 2~3 组对比/消融实验；
8. 编写 README，保证其他人能够复现；
9. 为后续课程报告提供实验结果、图表和数据；
10. 代码保持模块化，便于后续扩展 PointNet++、DGCNN 等模型。

---

# 3. 技术栈

建议环境：

- Python >= 3.10
- PyTorch
- NumPy
- Matplotlib
- tqdm
- scikit-learn
- trimesh（用于三维模型读取与表面采样，可选）
- pandas（实验结果整理，可选）
- TensorBoard（可选）

硬件：

- NVIDIA RTX 4060 8GB
- CUDA 可用时优先使用 GPU
- CPU 模式也应能够运行小规模测试

---

# 4. Agent 总体执行原则

Agent 实现本项目时必须遵循以下原则。

## 4.1 先跑通最小闭环，再增加功能

严格按照：

```text
数据读取
  ↓
单样本可视化
  ↓
DataLoader
  ↓
PointNet forward
  ↓
单 batch 训练
  ↓
完整训练
  ↓
测试
  ↓
实验
  ↓
可视化与文档
```

不得一开始同时开发所有功能。

---

## 4.2 每一个阶段都必须可验证

完成一个阶段后，必须给出对应验证命令。

例如：

```bash
python scripts/check_dataset.py
```

或：

```bash
python train.py --epochs 2 --num-points 256
```

只有当前阶段验证通过后，才进入下一阶段。

---

## 4.3 不允许把大型数据和训练产物提交到 Git

以下内容默认不得进入 Git：

```text
data/
datasets/
checkpoints/
runs/
logs/
outputs/
__pycache__/
*.pth
*.pt
*.ckpt
```

必要的实验结果图片可有选择地提交到：

```text
assets/
results/
```

---

## 4.4 优先保证代码易读

代码目标是“课程项目 + GitHub 项目”，不是竞赛式代码。

要求：

- 模块职责明确；
- 函数名清晰；
- 关键逻辑有注释；
- 避免一个文件包含全部代码；
- 训练参数统一通过 argparse 或 config 管理；
- 固定随机种子，保证实验可复现。

---

# 5. 推荐目录结构

Agent 应逐步构建如下结构：

```text
pointnet-modelnet10-pytorch/
├── README.md
├── requirements.txt
├── .gitignore
├── train.py
├── test.py
├── configs/
│   └── default.yaml
├── datasets/
│   ├── __init__.py
│   └── modelnet10.py
├── models/
│   ├── __init__.py
│   └── pointnet.py
├── utils/
│   ├── __init__.py
│   ├── seed.py
│   ├── metrics.py
│   ├── visualization.py
│   └── checkpoint.py
├── scripts/
│   ├── check_dataset.py
│   ├── visualize_sample.py
│   └── run_experiments.py
├── experiments/
│   └── README.md
├── results/
│   ├── figures/
│   └── metrics/
├── checkpoints/
├── data/
└── docs/
    └── report_notes.md
```

说明：

- `data/`：本地数据，不提交 Git；
- `checkpoints/`：模型权重，不提交 Git；
- `results/`：保存实验曲线、混淆矩阵等；
- `docs/report_notes.md`：同步记录课程报告所需结论。

---

# 6. 阶段一：项目初始化

## 目标

建立项目基础结构并确保 Python 环境可运行。

## Agent 任务

1. 创建基础目录；
2. 创建 `.gitignore`；
3. 创建 `requirements.txt`；
4. 创建一个最简 README；
5. 添加环境检查代码；
6. 检查 PyTorch 是否识别 CUDA。

## 验收标准

执行：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

能够正常输出版本号。

如果 CUDA 可用，应输出：

```text
True
```

---

# 7. 阶段二：ModelNet10 数据准备

## 目标

完成 ModelNet10 数据集读取和点云采样。

## 数据流程

```text
ModelNet10 3D Mesh
        ↓
读取 mesh
        ↓
从模型表面采样点
        ↓
N × 3 点云
        ↓
归一化
        ↓
Tensor
```

建议初始设置：

```text
num_points = 1024
```

每个输入：

```text
shape = [1024, 3]
```

---

## 点云归一化

对点云进行中心化：

\[
p_i' = p_i - \bar p
\]

其中：

\[
\bar p = \frac{1}{N}\sum_{i=1}^{N}p_i
\]

随后缩放到单位球附近：

\[
p_i'' = \frac{p_i'}{\max_i ||p_i'||_2}
\]

---

## Agent 任务

实现：

```text
datasets/modelnet10.py
```

要求 Dataset 至少返回：

```python
points, label
```

其中：

```text
points.shape = [N, 3]
label = int
```

同时创建：

```text
scripts/check_dataset.py
scripts/visualize_sample.py
```

---

## 验收标准

执行：

```bash
python scripts/check_dataset.py
```

应输出类似：

```text
Dataset size: xxxx
Point shape: torch.Size([1024, 3])
Label: 3
```

执行：

```bash
python scripts/visualize_sample.py
```

能够显示一个三维点云。

---

# 8. 阶段三：实现基础 PointNet

## 目标

实现最基础的 PointNet 分类网络。

暂时不要一开始实现复杂 T-Net。

第一版网络建议：

```text
Input
[N, 3]
   ↓
Shared MLP
3 → 64
   ↓
Shared MLP
64 → 128
   ↓
Shared MLP
128 → 1024
   ↓
Global Max Pooling
   ↓
1024-D global feature
   ↓
FC
1024 → 512
   ↓
FC
512 → 256
   ↓
FC
256 → 10
```

可使用：

- BatchNorm
- ReLU
- Dropout

---

## 关键要求

网络输入推荐统一为：

```text
[B, N, 3]
```

进入 Conv1d 前转换：

```text
[B, 3, N]
```

Shared MLP 可以使用：

```python
nn.Conv1d(..., kernel_size=1)
```

实现。

Global Max Pooling：

```python
torch.max(x, dim=2)
```

---

## Agent 任务

实现：

```text
models/pointnet.py
```

同时写简单的 shape test。

---

## 验收标准

执行类似：

```bash
python -m models.pointnet
```

或者独立测试脚本，输入：

```text
[8, 1024, 3]
```

输出必须为：

```text
[8, 10]
```

---

# 9. 阶段四：训练闭环

## 目标

完成完整训练流程。

## 默认训练设置

初始可设置：

```text
optimizer: Adam
learning_rate: 0.001
batch_size: 32
epochs: 50
num_points: 1024
loss: CrossEntropyLoss
```

后续允许调整。

---

## train.py 必须包含

1. 读取配置；
2. 设置随机种子；
3. 创建 train / validation DataLoader；
4. 初始化模型；
5. 自动选择 CPU / CUDA；
6. forward；
7. 计算 loss；
8. backward；
9. optimizer.step；
10. 计算 accuracy；
11. 保存最佳模型；
12. 输出 epoch 日志；
13. 保存训练历史。

日志类似：

```text
Epoch [10/50]
Train Loss: 0.5231
Train Acc: 83.2%
Val Loss: 0.6114
Val Acc: 80.7%
```

---

## 快速 Debug 模式

Agent 必须支持快速运行，例如：

```bash
python train.py \
  --epochs 2 \
  --batch-size 8 \
  --num-points 256
```

用于快速验证，而不是每次完整训练。

---

## 验收标准

2 epoch 小规模训练能够正常运行：

- 无 shape error；
- loss 能反向传播；
- 模型可以保存；
- accuracy 可计算；
- GPU 可用时模型运行在 GPU。

---

# 10. 阶段五：完整训练与模型测试

## 目标

完成基础 PointNet baseline。

执行完整训练后保存：

```text
checkpoints/best_model.pth
```

测试程序：

```text
test.py
```

必须输出：

```text
Overall Accuracy
Per-Class Accuracy
Confusion Matrix
```

示例：

```text
Test Accuracy: 89.3%
```

---

## 输出文件

建议：

```text
results/
├── figures/
│   ├── loss_curve.png
│   ├── accuracy_curve.png
│   └── confusion_matrix.png
└── metrics/
    └── baseline.json
```

---

# 11. 阶段六：数据增强

## 目标

研究数据增强对 PointNet 分类性能的影响。

实现以下增强中的至少 3 个：

### Random Rotation

绕竖直轴随机旋转：

\[
R(\theta)
\]

### Random Scaling

例如：

```text
scale ∈ [0.8, 1.2]
```

### Random Translation

例如：

```text
[-0.1, 0.1]
```

### Jitter

加入小高斯噪声：

\[
p' = p + \epsilon
\]

---

## 对比实验

至少比较：

```text
Baseline PointNet
vs
PointNet + Data Augmentation
```

结果记录：

```text
results/metrics/augmentation.json
```

---

# 12. 阶段七：点数实验

## 目标

研究点云采样数量对分类性能的影响。

建议实验：

```text
256
512
1024
2048
```

如果训练时间有限，可使用：

```text
256
512
1024
```

统一其他训练参数。

记录：

| Num Points | Test Accuracy | Training Time |
|---:|---:|---:|
| 256 | | |
| 512 | | |
| 1024 | | |
| 2048 | | |

---

# 13. 阶段八：Pooling 消融实验

## 目标

验证 Global Max Pooling 的作用。

对比：

```text
Max Pooling
vs
Average Pooling
```

保持其余网络结构不变。

需要解释：

PointNet 输入点云不存在固定顺序，因此需要具有置换不变性的对称聚合函数。

对于点集：

\[
\{p_1,p_2,...,p_n\}
\]

即使顺序改变：

\[
\{p_3,p_1,p_n,...\}
\]

分类结果也不应该改变。

---

# 14. 可选阶段：加入 T-Net

如果基础项目全部完成且时间充足，可以进一步实现原论文中的：

```text
Input Transform
Feature Transform
```

但该部分属于增强项。

Agent 不得在 baseline 未跑通前优先实现。

可以形成：

```text
Simple PointNet
vs
PointNet + T-Net
```

对比实验。

---

# 15. 阶段九：结果可视化

至少生成以下图表：

## 训练 Loss 曲线

```text
Epoch → Train Loss / Val Loss
```

## Accuracy 曲线

```text
Epoch → Train Acc / Val Acc
```

## Confusion Matrix

展示 ModelNet10 各类别分类情况。

## 点云预测结果

至少展示：

- 预测正确样本；
- 预测错误样本。

例如：

```text
Ground Truth: chair
Prediction: chair
Confidence: 92.3%
```

---

# 16. 阶段十：README

README 至少包括：

```text
1. 项目简介
2. PointNet 简介
3. 项目结构
4. 环境安装
5. 数据集准备
6. 训练方法
7. 测试方法
8. 实验结果
9. 可视化
10. 后续工作
```

建议 README 开头：

```markdown
# PointNet on ModelNet10 with PyTorch

A PyTorch implementation of PointNet for 3D point cloud
classification on the ModelNet10 dataset.
```

---

# 17. 课程报告同步记录

Agent 在项目开发过程中，应同步更新：

```text
docs/report_notes.md
```

每完成一次重要实验，记录：

```text
实验目的
实验设置
实验结果
观察到的现象
原因分析
可以写入报告的结论
```

这样最终不用重新回忆整个开发过程。

---

# 18. 最终实验矩阵

项目完成后，至少应得到如下实验：

## Experiment 1：Baseline

```text
PointNet
1024 points
Max Pooling
No Augmentation
```

---

## Experiment 2：Data Augmentation

```text
PointNet
1024 points
Max Pooling
With Augmentation
```

---

## Experiment 3：Point Number

```text
256 / 512 / 1024 / 2048
```

---

## Experiment 4：Pooling Ablation

```text
Max Pooling
vs
Average Pooling
```

如果时间有限，优先：

```text
Baseline
+
Data Augmentation
+
Pooling Ablation
```

---

# 19. Agent 必须维护的实验记录

建议使用：

```text
experiments/README.md
```

记录每次正式实验。

格式：

```markdown
## EXP-001 Baseline

Date:

Model:
PointNet

Num Points:
1024

Batch Size:
32

Learning Rate:
0.001

Epochs:
50

Augmentation:
False

Pooling:
Max

Best Val Accuracy:

Test Accuracy:

Checkpoint:

Notes:
```

正式实验不得覆盖历史结果。

---

# 20. Git 提交建议

每完成一个阶段进行一次 Git commit。

例如：

```text
chore: initialize project structure

feat: add ModelNet10 dataset loader

feat: implement point cloud visualization

feat: implement PointNet classifier

feat: add training pipeline

feat: add evaluation metrics

feat: add data augmentation

exp: compare different point counts

exp: compare max and average pooling

docs: update README and experiment results
```

不要等整个项目完成后一次性提交。

---

# 21. Agent 执行顺序

Agent 应严格按照以下顺序工作：

```text
Phase 0
项目初始化
    ↓
Phase 1
ModelNet10 数据读取
    ↓
Phase 2
点云可视化
    ↓
Phase 3
PointNet forward
    ↓
Phase 4
单 batch overfit 测试
    ↓
Phase 5
小规模训练
    ↓
Phase 6
完整 baseline
    ↓
Phase 7
测试 + 混淆矩阵
    ↓
Phase 8
数据增强实验
    ↓
Phase 9
点数实验
    ↓
Phase 10
Pooling 消融
    ↓
Phase 11
README + 报告素材整理
```

---

# 22. 特别重要：单 Batch Overfit Test

完整训练前，Agent 必须执行一个 sanity check：

只选择一个很小 batch，例如：

```text
8~32 个样本
```

重复训练该 batch。

理想情况下模型应该能够明显过拟合，训练准确率逐渐接近：

```text
100%
```

如果无法 overfit 一个小 batch，应优先检查：

- label 是否正确；
- loss 是否正确；
- tensor shape；
- optimizer；
- model forward；
- normalization；
- 数据处理。

这个步骤能够避免浪费大量 GPU 训练时间。

---

# 23. 最终验收标准

项目完成时必须满足以下条件。

## 数据

- [ ] ModelNet10 能正常读取
- [ ] 每个样本可转换为固定点数点云
- [ ] 点云完成中心化和尺度归一化
- [ ] 点云可视化正常

## 模型

- [ ] PointNet 可独立 forward
- [ ] 输入 `[B, N, 3]`
- [ ] 输出 `[B, 10]`
- [ ] 支持 CUDA

## 训练

- [ ] loss 正常下降
- [ ] 可以保存 checkpoint
- [ ] 可以恢复/加载 checkpoint
- [ ] 可以记录 train / val accuracy

## 测试

- [ ] 输出 overall accuracy
- [ ] 输出 per-class accuracy
- [ ] 生成 confusion matrix

## 实验

- [ ] baseline
- [ ] augmentation 对比
- [ ] 点数对比
- [ ] pooling 消融

## 工程

- [ ] `.gitignore` 正确
- [ ] README 完整
- [ ] requirements.txt 完整
- [ ] 代码目录清晰
- [ ] 大型数据未提交 Git
- [ ] checkpoint 未提交 Git

---

# 24. MVP 最小完成版本

如果课程截止时间临近，只完成以下部分也必须保证项目完整：

```text
ModelNet10
    ↓
1024 点采样
    ↓
PointNet
    ↓
训练
    ↓
测试
    ↓
Accuracy
    ↓
训练曲线
```

对应：

```text
Dataset
Model
Train
Test
README
```

然后再依次增加：

```text
Data Augmentation
Pooling Ablation
Point Number Experiment
T-Net
```

---

# 25. Agent 的第一轮任务

Agent 接手项目后，不要立即编写全部模型。

第一轮只完成：

1. 检查当前 `pointnet-modelnet10-pytorch` 文件夹；
2. 初始化标准目录；
3. 创建 `.gitignore`；
4. 创建 `requirements.txt`；
5. 创建 README 骨架；
6. 创建 Python 环境检查脚本；
7. 确认 PyTorch / CUDA；
8. 给出当前目录树；
9. 给出下一步 ModelNet10 数据准备方案。

完成后停止，等待用户确认，再继续数据集阶段。

---

# 26. 给 Agent 的总指令

> 你正在实现一个课程级、但具备标准工程结构的 PointNet 项目。
>
> 不要追求一次完成所有功能，也不要过早进行架构优化。
>
> 首先保证 Dataset → Model → Train → Test 的最小闭环完全可运行。
>
> 每开发一个模块，都必须提供独立验证方式。
>
> 每一个正式实验必须保存参数和结果。
>
> 不要将数据集、checkpoint、日志等大型文件提交 Git。
>
> 所有实现应优先考虑可读性、可复现性和报告可分析性。
>
> 当一个阶段验证完成后，再进入下一阶段。
