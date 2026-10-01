# PointNet on ModelNet10 with PyTorch

使用 PyTorch 实现面向 ModelNet10 数据集的 PointNet 三维点云分类项目。


## 当前进度

baseline 训练与官方测试集评估均已完成。基础 PointNet 在 RTX 4060 上训练 50 epoch，按验证集选定第 49 轮，验证准确率为 94.99%。该固定权重在官方 908 个测试样本上答对 817 个，测试总准确率 **89.98%**，平均类别准确率 **89.80%**。已保存混淆矩阵、每类指标和逐样本预测。数据增强组合的 50 轮训练也已完成，最佳验证准确率为 95.86%（765/798），比 baseline 高 0.88 个百分点。下一步是对已固定的增强模型进行官方测试集评估。


## 项目结构

```text
.
├── README.md
├── requirements.txt         # 项目 Python 依赖清单
├── .gitignore
├── train.py                 # 训练入口，参数通过命令行配置
├── test.py                  # 固定权重的官方测试集评估
├── configs/                 # 后续训练配置
├── datasets/                # 数据集读取源码，纳入 Git
├── models/                  # PointNet 模型源码
├── utils/                   # 随机种子、指标、可视化、权重管理
├── scripts/                 # 检查、可视化、实验脚本
├── tests/                   # 划分隔离、指标计算及训练更新检查
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

`data/` 已由 `.gitignore` 排除。Dataset 在读取时将原始网格采样成点云并归一化。`train.py` 默认从每个类别的官方训练样本中划出约 20% 作为验证集，得到 3193／798 个训练／验证样本；原始文件不移动，划分名单随实验保存。官方测试集保留用于最终评估。

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
模型没有在输出端加 Softmax，训练时把这些分数直接交给 `CrossEntropyLoss`。
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
训练 DataLoader 在尾批恰好只有 1 个样本时丢弃尾批；验证保留全部样本。

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

### 训练／验证循环（小规模运行已通过）

`train.py` 每个 epoch 先学习训练集，再评估验证集。一个 epoch 表示遍历所选训练数据一遍。
训练调用 `model.train()` 并更新参数；验证调用 `model.eval()` 和 `torch.no_grad()`，不修改参数或 BatchNorm 统计量。
loss 按实际样本数加权，准确率为正确预测数除以样本数。

划分与采样约定：

- 先从官方 train 按类别划分 3193 个训练样本、798 个验证样本，默认 `--val-fraction 0.2 --seed 42`。
- 再从各自划分内抽取调试子集，保证训练与验证的文件不重叠。官方 test 不参与训练、验证或权重选择。
- 每个模型按固定 seed 采样一次，点云缓存在 CPU 内存，后续 epoch 重用；本阶段没有数据增强。
- 训练顺序每个 epoch 打乱；验证顺序与采样固定，以便不同 epoch 的指标可比较。
- 若启用多个 DataLoader worker，每个 worker 有各自的内存缓存。

复现本阶段的小规模运行：

```bash
source .venv/bin/activate
python train.py --epochs 2 --batch-size 8 --num-points 256 \
  --train-per-class 8 --val-per-class 2
```

这会使用每类 8 个训练样本、2 个验证样本，即 80／20 个；Adam 学习率默认 0.001，Dropout 恢复为 0.3。
`--train-per-class` 和 `--val-per-class` 默认都是 0，表示使用各自完整划分。`--device auto` 优先用可用 CUDA，否则使用 CPU；
也可指定 `--device cpu` 或 `--device cuda`。`--num-workers` 默认 0。
训练参数全部由 argparse 管理，可通过 `python train.py --help` 查看。

每个实验有独立名称；已有同名实验会拒绝覆盖。输出包括：

- `results/metrics/<run_name>.json`：配置、逐 epoch 指标、最佳 epoch 和权重重载检查结果。
- `results/metrics/<run_name>_split.json`：完整划分及本次实际使用的文件名单。
- `results/figures/<run_name>.png`：训练／验证曲线。
- `checkpoints/<run_name>/best_model.pth`：最佳模型权重、类别映射、配置和划分记录路径，不纳入 Git。

最佳模型优先按验证准确率选择，准确率相同时选择验证 loss 更低的一轮。运行结束后加载最佳权重，重新验证并核对结果。

2026-10-01 的 CPU 小规模运行结果：

| Epoch | 训练 loss | 训练准确率 | 验证 loss | 验证准确率 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.9422 | 31.25% | 2.6403 | 10% |
| 2 | 1.1756 | 63.75% | 2.9860 | 15% |

本次最佳为第 2 轮，保存与重载检查通过。
验收记录见 [训练指标](results/metrics/train_small_20261001_seed42.json)、
[划分名单](results/metrics/train_small_20261001_seed42_split.json) 和
[训练／验证曲线](results/figures/train_small_20261001_seed42.png)。

这里仅评估了 20 个验证样本，15% 表示答对 3 个，不能代表完整验证集或测试集表现。
验证准确率略升而 loss 上升并不矛盾：准确率只看选中的类别是否正确，交叉熵还取决于模型给真实类别分配的概率。
上述 2 轮小规模实验用于验收训练流程，完整 baseline 结果见下文。

基础检查：`python -m unittest discover -s tests -v`，覆盖划分不重叠、调试子集边界、验证不更新模型、指标加权和训练参数更新。

### 完整 baseline（已完成）

baseline 是后续数据增强、点数对比和 Pooling 消融的比较基准。
本次使用基础 PointNet（无 T-Net）、1024 点、batch size 32、Adam 学习率 0.001、Dropout 0.3、50 epoch、seed 42。
使用完整的 3193／798 训练／验证划分，固定点云采样，不加数据增强，按验证准确率保存最佳模型。

```bash
python train.py --device cuda --epochs 50 --batch-size 32 --num-points 1024 \
  --learning-rate 0.001 --dropout 0.3 --val-fraction 0.2 --seed 42 --num-workers 0
```

首轮需要读取网格、采样并填充内存缓存，后续轮次会快很多。首轮每 25 个训练 batch 打印一次进度。
此前在应用沙箱内无法访问 NVIDIA 驱动而回退至 CPU；已确认本机 RTX 4060 在可访问 GPU 的环境下正常工作。

实验 `baseline_20261001_seed42_r2` 共运行约 342.25 秒（5 分 42 秒），首轮读取网格及训练／验证约 274.28 秒。

| 权重 | Epoch | 训练准确率 | 验证准确率 | 验证 loss |
| --- | ---: | ---: | ---: | ---: |
| 验证最佳 | 49 | 99.37% | 94.99% | 0.1797 |
| 最后一轮 | 50 | 98.40% | 93.23% | 0.2791 |

保存的是第 49 轮。重新加载后完整验证集结果保持一致。
训练 loss 整体下降，验证曲线在部分轮次明显波动，因此不直接使用最后一轮权重。
上述数值属于验证集；同一权重的官方测试结果见下节。

- [完整指标与代码文件哈希](results/metrics/baseline_20261001_seed42_r2.json)
- [训练／验证曲线](results/figures/baseline_20261001_seed42_r2.png)
- [完整划分名单](results/metrics/baseline_20261001_seed42_r2_split.json)
- 本地最佳权重：`checkpoints/baseline_20261001_seed42_r2/best_model.pth`，不纳入 Git。
- 本地运行日志：`logs/baseline_20261001_seed42_r2.log`，不纳入 Git。

### 官方测试集评估（已完成）

`test.py` 加载按验证集选定的权重，沿用其中的点数与采样种子。
当前 baseline 使用第 49 轮权重、1024 点、seed 42，每个测试模型只进行一次固定表面采样，不加增强、不做多次投票。
脚本核对类别映射、官方测试样本数，并在有对应训练报告时核对最佳轮次和已记录的权重哈希。
预测使用 `model.eval()` 和 `torch.inference_mode()`；结束后检查模型状态和权重文件没有变化。

复现（默认生成独立结果名称）：

```bash
source .venv/bin/activate
python test.py --checkpoint checkpoints/baseline_20261001_seed42_r2/best_model.pth \
  --device cuda
```

2026-10-01 在 RTX 4060 上评估全部 908 个官方测试样本，耗时约 52.28 秒，包含网格读取和绘图：

- 总准确率：817／908 = **89.98%**，按所有样本计算。
- 平均类别准确率：**89.80%**，先计算每类准确率，再对 10 个类别取平均。
- 测试 loss：0.402902。
- 权重哈希与已选定的 baseline 一致；评估前后模型参数、BatchNorm 状态和权重文件均未改变。

| 类别 | 正确／总数 | 准确率 |
| --- | ---: | ---: |
| bathtub（浴缸） | 47／50 | 94.00% |
| bed（床） | 98／100 | 98.00% |
| chair（椅子） | 100／100 | 100.00% |
| desk（书桌） | 70／86 | 81.40% |
| dresser（抽屉柜） | 77／86 | 89.53% |
| monitor（显示器） | 96／100 | 96.00% |
| night_stand（床头柜） | 62／86 | 72.09% |
| sofa（沙发） | 97／100 | 97.00% |
| table（桌子） | 76／100 | 76.00% |
| toilet（马桶） | 94／100 | 94.00% |

混淆矩阵的**行是真实类别，列是预测类别**，对角线表示预测正确。
主要错误包括 table → desk 23 个、night_stand → dresser 17 个，反向错误分别为 8 个和 8 个。
这两对类别的互相混淆合计占 56／91 个错误。预测示例用于观察现象，尚不能据此断言错误原因。

- [完整测试指标和混淆矩阵数值](results/metrics/test_baseline_20261001_seed42.json)
- [逐样本预测 CSV](results/metrics/test_baseline_20261001_seed42_predictions.csv)
- [混淆矩阵图片](results/figures/test_baseline_20261001_seed42_confusion_matrix.png)
- [正确／错误点云示例](results/figures/test_baseline_20261001_seed42_examples.png)

点云示例按数据集顺序，选取至多三个不同真实类别的首个正确／错误案例；标题绿／红色区分正误，点颜色仅表示 z 坐标。
CSV 中的 confidence 为模型对预测类别给出的 Softmax 分数，不参与样本筛选或权重选择。
验证集 94.99% 和测试集 89.98% 分别来自不同数据：验证集用于选择模型，测试集用于记录选定模型的独立评估结果。
本次 baseline 的权重选择已经固定；后续对比实验应预先确定方案，使用验证集调参与选权重。

## 实验与可视化

已完成完整 baseline 训练与测试，生成训练曲线、测试混淆矩阵和预测示例。待开展：数据增强、点数对比和 Pooling 消融。

正式实验记录见 [experiments/README.md](experiments/README.md)。

## 后续工作

baseline 的数据准备、训练、验证选模、测试评估和可视化已形成完整流程。下一步预先定义对比实验，每次改变一个因素，用验证集比较与选择配置，保留当前 baseline 的全部记录。

## 数据增强对比

数据增强是在训练时对输入作随机变化，标签保持不变。本阶段实现三种组合：

| 方法 | 具体设置 | 直观含义 |
| --- | --- | --- |
| 等比例缩放 | 每个物体一个随机因子，范围 0.8～1.2 | 物体变大或变小 |
| 平移 | 每个物体一个随机向量，各轴 −0.1～0.1 | 物体位置偏移 |
| 抖动 | 每点每坐标独立高斯噪声，标准差 0.01，截断到 ±0.05 | 点的位置有少量误差 |

数据路径：固定采样与归一化 → 缓存原始点云 → 训练 batch 随机增强 → PointNet。每次访问产生新增强；非原地运算保护缓存。增强后不重新归一化，以保留缩放和平移效果。验证与测试仍使用原始固定点云。

[同一把椅子的增强示意图](results/figures/augmentation_preview.png)。四张图使用相同坐标范围；这里只是展示，CPU 示意图的随机数不要求与 CUDA 训练逐位相同。

代码阅读顺序：

1. `datasets/augmentation.py`：`points * scale + shift + noise` 实现三种变化。
2. `utils/training.py`：仅在训练前向传播前调用增强。
3. `train.py`：命令行开关、独立增强随机数生成器与参数记录。

完整实验（默认 50 epoch、1024 点、seed 42，自动创建新名称）：

```bash
source .venv/bin/activate
python train.py --augmentation scale_shift_jitter --device cuda
```

无增强仍是默认行为，可显式传入 `--augmentation none`。完整训练以相同 3193／798 划分、网络、优化器等条件对比；只通过验证集选最佳权重。组合实验不能解释单种增强的贡献，一次 seed 的结果也不能代表重复实验平均水平。

小规模运行命令（用于自行检查接入流程，不用于性能结论）：

```bash
python train.py --augmentation scale_shift_jitter --epochs 2 \
  --batch-size 8 --num-points 256 --train-per-class 8 --val-per-class 2
```

查看增强示例（新输出路径）与生成验证对比报告：

```bash
python scripts/visualize_augmentation.py --output results/figures/augmentation_preview_new.png
python scripts/compare_augmentation.py \
  --baseline results/metrics/baseline_20261001_seed42_r2.json \
  --augmented results/metrics/augmentation_20261001_seed42.json \
  --output results/metrics/augmentation_comparison_new.json
```

比较脚本读取已完成记录，要求两次训练配置、数据划分和模型源码一致，输出 JSON 与同名 PNG 验证曲线。

本次完整对比结果：

| 配置 | 最佳轮次 | 验证正确数 | 最佳验证准确率 | 对应验证 loss | 总训练用时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 无增强 baseline | 49 | 758/798 | 94.99% | 0.179686 | 342.25 秒 |
| 缩放 + 平移 + 抖动 | 20 | 765/798 | 95.86% | 0.106661 | 355.85 秒 |

增强组合在本次验证集上净多答对 7 个物体，准确率增加约 0.88 个百分点。时间包含读取网格、训练、最佳权重重载及绘图，单次计时不用于严格性能基准。已完成 50 轮实际训练及内置最佳权重重载验证；本阶段没有新增或运行单元测试。

[对比报告](results/metrics/augmentation_comparison_20261001_seed42.json) · [验证曲线对比](results/metrics/augmentation_comparison_20261001_seed42.png) · [增强训练完整记录](results/metrics/augmentation_20261001_seed42.json) · [增强训练曲线](results/figures/augmentation_20261001_seed42.png)。

增强模型权重：`checkpoints/augmentation_20261001_seed42/best_model.pth`。官方测试准确率尚未评估，不能将 95.86% 作为测试成绩。
