# PointNet on ModelNet10 with PyTorch

使用 PyTorch 实现面向 ModelNet10 数据集的 PointNet 三维点云分类课程项目。


## 当前进度

已完成项目框架、ModelNet10 原始数据准备和本机 Python / PyTorch CUDA 环境配置。已实现 OFF 读取、表面采样、点云归一化和 Dataset，并提供数据检查、DataLoader 批次检查及点云可视化脚本。数据管线已通过全部 4899 个模型的读取与数值检查，点云可视化和 DataLoader 批次检查通过。下一阶段是基础 PointNet，模型和训练流程尚未实现。


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

### 已验证的本机环境

- 系统：Ubuntu Linux x86_64
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

## 训练与测试

待实现：基础 PointNet、单 batch 过拟合验证、小规模训练、完整训练及测试。

## 实验与可视化

待实现：baseline、数据增强、点数对比、Pooling 消融，以及训练曲线、混淆矩阵和点云预测可视化。

正式实验记录见 [experiments/README.md](experiments/README.md)。

## 后续工作

数据管线验收已完成。下一步实现基础 PointNet，先检查输入 `[B, N, 3]` 能否得到输出 `[B, 10]`，再进入单 batch 训练及完整训练流程。
