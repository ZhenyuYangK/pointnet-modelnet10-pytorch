# PointNet on ModelNet10 with PyTorch

使用 PyTorch 实现面向 ModelNet10 数据集的 PointNet 三维点云分类项目。


## 当前进度

已完成项目框架、ModelNet10 原始数据准备和本机 Python / PyTorch CUDA 环境配置。已实现 OFF 读取、表面采样、点云归一化和 Dataset，并提供数据检查、DataLoader 批次检查及点云可视化脚本。数据管线已通过全部 4899 个模型的读取与数值检查，点云可视化和 DataLoader 批次检查通过。基础 PointNet 已通过 CPU / CUDA 上的前向、点顺序不变性和反向传播检查。单 batch 过拟合验证已通过：20 个固定训练样本达到 100% 同批准确率，权重保存、重载结果一致。下一阶段是划分验证集并跑通小规模训练，尚无验证集或测试集准确率。


## 项目结构

```text
.
├── README.md
├── requirements.txt         # 项目 Python 依赖清单
├── .gitignore
├── configs/                 # 后续训练配置
├── datasets/                # 数据集读取源码，纳入 Git
├── models/                  # PointNet 模型源码
├── utils/                   # 随机种子、指标、可视化、权重管理
├── scripts/                 # 检查、可视化、实验脚本
├── experiments/
│   └── README.md            # 正式实验记录
├── results/
│   ├── figures/             # 可选择提交的结果图片
│   └── metrics/             # 实验指标
├── checkpoints/             # 本地模型权重，不纳入 Git
└── data/                    # 本地数据，不纳入 Git
```

空源码及结果目录使用 `.gitkeep` 保留。`data/`、`checkpoints/` 仅在本地创建，克隆仓库后按需重新创建。

## 环境与依赖

### 已验证的环境

- 系统：Ubuntu 22.04
- Python：3.10.12
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU
- NVIDIA 驱动：580.126.09（`nvidia-smi` 显示 CUDA 13.0）
- PyTorch：2.13.0+cu130，CUDA 13.0
- 已安装并验证：NumPy 2.2.6、Matplotlib 3.10.9、tqdm 4.70.1、scikit-learn 1.7.2
- 虚拟环境位于仓库根目录 `.venv/`，已加入 `.gitignore`。CUDA 张量运算验证通过。

### 创建环境

在 Linux 上从仓库根目录执行：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
```

`requirements.txt` 包含 PyTorch 和项目 Python 依赖。上面的 PyTorch 安装命令选用 CUDA 13.0 GPU 轮子；PyTorch 轮子自带 CUDA 运行时，本机 NVIDIA 驱动负责与 GPU 通信，无需另行安装 CUDA Toolkit。若使用其他操作系统、CPU 或不同的 NVIDIA 驱动，请先在 [PyTorch 安装页面](https://pytorch.org/get-started/locally/)选择相应安装命令，再安装 `requirements.txt`。

激活环境后，可用以下命令确认 PyTorch 是否识别 GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 数据集准备

使用 [Princeton ModelNet10](https://modelnet.cs.princeton.edu/download.html) 原始 OFF 网格，保留官方训练 / 测试划分。官方下载较慢，本次使用 [Kaggle 原始数据镜像](https://www.kaggle.com/datasets/balraj98/modelnet10-princeton-3d-object-dataset)，其中全部 OFF 文件的名称、字节数和 CRC32 已与官方 ZIP 文件目录逐一比对一致。

```text
data/
├── ModelNet10.zip              # 下载的镜像压缩包
├── modelnet10_manifest.json    # 来源、压缩包 SHA256 和逐文件校验信息
└── ModelNet10/
    ├── bathtub/
    │   ├── train/*.off
    │   └── test/*.off
    └── ...                    # 共 10 类，每类均有 train 和 test
```

| 类别 | 训练集 | 测试集 |
| --- | ---: | ---: |
| bathtub | 106 | 50 |
| bed | 515 | 100 |
| chair | 889 | 100 |
| desk | 200 | 86 |
| dresser | 200 | 86 |
| monitor | 465 | 100 |
| night_stand | 200 | 86 |
| sofa | 680 | 100 |
| table | 392 | 100 |
| toilet | 344 | 100 |
| **合计** | **3991** | **908** |

在项目根目录重新准备数据（需要 `curl` 和 `unzip`）：

```bash
mkdir -p data
curl --fail --location --retry 3 --continue-at - \
  'https://www.kaggle.com/api/v1/datasets/download/balraj98/modelnet10-princeton-3d-object-dataset' \
  --output data/ModelNet10.zip
unzip -tq data/ModelNet10.zip
unzip -nq data/ModelNet10.zip 'ModelNet10/*' -d data
```

官方备用下载地址：<https://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip>。官方和镜像的 ZIP 打包方式不同，压缩包哈希不相同；本次按 OFF 文件逐一核验。

`data/` 已由 `.gitignore` 排除。Dataset 在读取时将原始网格采样成点云并归一化。验证集尚未划分，后续应从训练集中划分，测试集保留用于最终评估。

## 点云读取与检查

一个 OFF 文件表示一个三维物体，包含顶点坐标和由顶点组成的面。
`datasets/modelnet10.py` 按下面的流程，把它转换成模型需要的数据：

1. `read_off`：读取网格的顶点和面。
2. `sample_surface_points`：按面面积分配采样概率，从表面取出 1024 个点。
3. `normalize_points`：减去点云中心，再除以最远点的距离，使点云中心接近原点、最大半径为 1。
4. `ModelNet10`：返回点云张量和类别编号，类别按文件夹名称排序，编号为 0～9。

单个点云的形状是 `[1024, 3]`，每行是一个点的 `(x, y, z)`。
DataLoader 把 8 个物体放在同一批时，点云形状是 `[8, 1024, 3]`，标签形状是 `[8]`。

从项目根目录运行：

```bash
source .venv/bin/activate
# 快速检查：训练集、测试集各取每类一个模型，并检查一个 batch。
python scripts/check_dataset.py
# 完整检查：逐一读取全部 4899 个模型，保存验收结果。
python scripts/check_dataset.py --all --output results/metrics/dataset_check.json
# 生成一张椅子的点云图，默认保存到 results/figures/chair_0001.png。
python scripts/visualize_sample.py
# 生成十类点云总览；有桌面环境时可加 --show 打开交互窗口。
python scripts/visualize_sample.py --all-classes
```

检查脚本核对官方样本数、类别、张量形状、有限坐标、中心、单位半径、固定随机种子的重复性及 DataLoader 输出。
这些是数据管线检查，没有进行模型训练或测试准确率评估。完整检查会读取所有网格，耗时明显多于快速检查。

本次全量检查通过：训练集 3991 个、测试集 908 个模型；批次形状为 `[8, 1024, 3]`。
另已检查 256 点、batch size 4、2 个 DataLoader worker 的配置。
验收记录见 [dataset_check.json](results/metrics/dataset_check.json)，
示例见 [椅子点云](results/figures/chair_0001.png) 和 [十类总览](results/figures/modelnet10_train_classes.png)。

可视化可用 `--class-name table --index 1` 选择某一类别的第 2 个模型，
用 `--num-points 2048` 改变采样点数。图中颜色表示 z 坐标，类别名称来自文件夹标签。
两个脚本默认固定 `seed=42`，相同参数会得到相同点云；Dataset 未传入 `seed` 时，每次读取会重新随机采样。

## 基础 PointNet

模型位于 `models/pointnet.py`，类名为 `PointNetClassifier`，当前是没有 T-Net 的基础版本。
输入为 `[B, N, 3]`，输出为 `[B, 10]`。`B` 是一批物体的数量，`N` 是每个物体的点数。

| 步骤 | 做什么 | 张量形状 |
| --- | --- | --- |
| 输入 | 每个点有 x、y、z 坐标 | `[B, N, 3]` |
| 转置 | 调整成 Conv1d 要求的维度顺序 | `[B, 3, N]` |
| 共享 MLP | 对所有点使用同一套计算，特征数 3 → 64 → 128 → 1024 | `[B, 1024, N]` |
| 全局最大池化 | 对每个特征取所有点中的最大值，得到整个物体的特征 | `[B, 1024]` |
| 分类层 | 全连接层 1024 → 512 → 256 → 10 | `[B, 10]` |

共享 MLP 使用 `Conv1d(kernel_size=1)`，逐点处理，不依赖点之间的排列位置。
最大池化的结果不随点的顺序变化。中间的 1024 是**特征数量**，和默认采样的 1024 个点是两个不同概念；模型也能处理其他点数。

输出的 10 个数是类别分数（logits），列顺序对应 `dataset.class_to_idx`。
模型没有在输出端加 Softmax，后续训练会把这些分数直接交给 `CrossEntropyLoss`。
新创建的模型参数是随机初始化的；形状检查中的分数不能用于判断识别效果。

从项目根目录运行：

```bash
source .venv/bin/activate
# 使用随机点云观察输入、输出形状，无需读取数据集。
python -m models.pointnet
# 使用真实训练样本检查模型，默认检查 CPU 以及可用的 CUDA。
python scripts/check_model.py --output results/metrics/model_check.json
```

第一条模型命令输出输入 `[8, 1024, 3]`、输出 `[8, 10]`，参数量为 801354。
检查脚本验证真实批次形状、有限输出、打乱点顺序后输出一致，以及不同点数和单样本推理。
它还计算一次交叉熵并调用 `backward()`，确认所有参数能获得有限梯度；没有执行优化器更新。
检查中的 loss 仅用于确认计算正常，不是训练成果。
验收记录见 [model_check.json](results/metrics/model_check.json)。

网络使用 BatchNorm 和 Dropout，推理或比较点顺序时需要调用 `model.eval()`；训练时调用 `model.train()`。
分类层的 BatchNorm 要求训练批次至少包含 2 个物体；评估模式支持只输入 1 个物体。
后续训练 DataLoader 需要避免最后一个批次只有 1 个样本。

## 训练与测试

### 单 batch 过拟合验证（已完成）

这一步故意让模型记住一小批训练样本，用来确认数据、标签、梯度和参数更新能够配合工作。
`scripts/overfit_batch.py` 从官方训练集每类选 2 个模型，共 20 个物体，每个物体采样 256 个点。
点云在训练前只读取一次，后面 200 次参数更新使用同一批张量。

每次更新的核心流程是：

```python
optimizer.zero_grad(set_to_none=True)  # 清除旧梯度
logits = model(points)                # 前向预测
loss = criterion(logits, labels)      # 交叉熵衡量预测误差
loss.backward()                       # 计算梯度
optimizer.step()                      # 更新模型参数
```

实验使用 Adam、学习率 0.001、seed 42。为排除随机干扰，此诊断将 Dropout 设为 0，且不进行数据增强。
每步更新后切换到 `eval()`，重新评估同一批训练点云。这里的 eval 指模型运行模式，数据依然来自这 20 个训练样本。
BatchNorm 在训练和评估模式下使用不同的统计量，因此前期的两条曲线可能有差距。

```bash
source .venv/bin/activate
# 默认自动选择可用 CUDA，否则使用 CPU；每次生成独立的结果名称。
python scripts/overfit_batch.py
# 也可以明确选择设备、点数和更新次数。
python scripts/overfit_batch.py --device cpu --num-points 256 --steps 200
```

验收要求：最终同批评估准确率为 100%，交叉熵小于 0.05 且比初始值下降；保存的权重重新加载后，输出保持一致。
2026-10-01 的 CPU 实验通过：eval loss 从 2.303629 降到 0.001788，同批准确率从 10% 升到 100%。
本次运行环境未检测到可用 CUDA，脚本自动使用 CPU；此前的模型 CUDA 检查记录仍保留。

- [本次指标与逐步记录](results/metrics/overfit_20261001_seed42.json)
- [loss 和准确率曲线](results/figures/overfit_20261001_seed42.png)
- 本地权重：`checkpoints/overfit_20261001_seed42/model.pth`，已由 Git 忽略；其中也保存了固定点云、标签和模型配置。

重复运行时，指标、图片分别写入 `results/metrics/<run_name>.json` 和 `results/figures/<run_name>.png`，
权重写入 `checkpoints/<run_name>/model.pth`。可以用 `--run-name` 指定新名称，已有同名实验会拒绝覆盖。
这项结果证明模型能记住这批样本；对未见过的物体的识别能力，需要后续用独立验证集和测试集衡量。

### 后续训练

待实现：从官方训练集划分验证集、小规模训练、完整训练及最终测试。

## 实验与可视化

已生成单 batch 诊断的 loss 和准确率曲线。待实现：完整 baseline、数据增强、点数对比、Pooling 消融，以及正式训练曲线、混淆矩阵和点云预测可视化。

正式实验记录见 [experiments/README.md](experiments/README.md)。

## 后续工作

数据管线、基础 PointNet 和单 batch 过拟合验证已完成。下一步从官方训练集中划分验证集，建立训练与验证循环，先用少量样本跑通 2 个 epoch，再进入完整训练。官方测试集保留到最终评估。
