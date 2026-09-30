"""Basic PointNet classifier: shared point features, max pooling, class logits."""

import torch
from torch import nn


class PointNetClassifier(nn.Module):
    """Classify point clouds of shape [batch_size, num_points, 3].

    This first baseline uses shared MLPs and global max pooling without T-Nets.
    The output is raw class scores (logits), suitable for CrossEntropyLoss.
    Training requires at least two samples per batch for classifier BatchNorm;
    evaluation also supports a single sample.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.3) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.num_classes = num_classes

        # kernel_size=1：每个点使用同一套权重，不混合相邻位置的点。
        self.point_features = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        # 汇总后的每个物体只有一条 1024 维特征，再映射到各类别分数。
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(f"Expected points with shape [B, N, 3], got {tuple(points.shape)}")
        if points.shape[0] == 0 or points.shape[1] == 0:
            raise ValueError("A point cloud batch must have at least one sample and one point")
        if self.training and points.shape[0] < 2:
            raise ValueError("Training BatchNorm requires batch_size >= 2; use model.eval() for single-sample inference")

        # Conv1d 的输入顺序是 [batch, channels, length]。
        features = self.point_features(points.transpose(1, 2))  # [B, 1024, N]
        global_features = features.max(dim=2).values           # [B, 1024]
        return self.classifier(global_features)                # [B, num_classes]


if __name__ == "__main__":
    torch.manual_seed(42)
    model = PointNetClassifier().eval()
    points = torch.randn(8, 1024, 3)
    with torch.no_grad():
        logits = model(points)
    print(f"Input shape: {list(points.shape)}")
    print(f"Output shape: {list(logits.shape)}")
    print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    print("These are scores from an untrained model, not classification results.")
