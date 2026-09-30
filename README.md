# PointNet on ModelNet10 with PyTorch

使用 PyTorch 实现面向 ModelNet10 数据集的 PointNet 三维点云分类课程项目。


## 当前进度

目前已建立项目框架，导入 ModelNet10 原始训练集与测试集，并配置、验证本机 Python / PyTorch CUDA 环境。模型、数据加载和点云采样代码尚未实现。


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
├── data/                    # 本地数据，不纳入 Git
└── docs/
    └── report_notes.md      # 课程报告素材
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

`data/` 已由 `.gitignore` 排除。当前数据是原始网格，后续需实现表面采样和归一化才能输入 PointNet。验证集尚未划分，后续应从训练集中划分，测试集保留用于最终评估。

## 训练与测试

待实现：基础 PointNet、单 batch 过拟合验证、小规模训练、完整训练及测试。

## 实验与可视化

待实现：baseline、数据增强、点数对比、Pooling 消融，以及训练曲线、混淆矩阵和点云预测可视化。

正式实验记录见 [experiments/README.md](experiments/README.md)，报告素材见 [docs/report_notes.md](docs/report_notes.md)。

## 后续工作

后续实现 OFF 网格读取、点云采样与归一化，再实现 PointNet 训练和测试。
