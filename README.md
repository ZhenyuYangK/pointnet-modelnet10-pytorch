# PointNet on ModelNet10 with PyTorch

使用 PyTorch 实现面向 ModelNet10 数据集的 PointNet 三维点云分类课程项目。

## 当前进度

目前仅建立项目框架，尚未编写 Python 代码或训练配置，未安装依赖、检查 PyTorch / CUDA 或准备数据。后续实现等待进一步规划。

## 项目结构

```text
.
├── README.md
├── requirements.txt         # 初步依赖清单，版本待环境验证
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

计划使用 Python >= 3.10 和 PyTorch。`requirements.txt` 为初步依赖清单，安装方式、版本及 CUDA 配置待后续确认与验证。

## 数据集准备

待规划：将 ModelNet10 原始数据存放于 `data/`，确认目录与类别后实现网格读取、点云采样和归一化。

## 训练与测试

待实现：基础 PointNet、单 batch 过拟合验证、小规模训练、完整训练及测试。

## 实验与可视化

待实现：baseline、数据增强、点数对比、Pooling 消融，以及训练曲线、混淆矩阵和点云预测可视化。

正式实验记录见 [experiments/README.md](experiments/README.md)，报告素材见 [docs/report_notes.md](docs/report_notes.md)。

## 后续工作

等待后续规划，再开展环境验证、数据准备及代码实现。
