"""Training-only point cloud augmentation, applied after fixed surface sampling."""

import torch


class PointCloudAugmentation:
    """Scale, translate, then jitter each [B, N, 3] cloud without modifying input.

    A separate generator keeps augmentation draws independent of model dropout
    and DataLoader shuffling. The same seed/device/batch order reproduces draws.
    """

    def __init__(self, seed, device):
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.config = {
            "name": "scale_shift_jitter",
            "order": ["isotropic_scale", "translation", "jitter"],
            "scale_range": [0.8, 1.2],
            "translation_range": [-0.1, 0.1],
            "jitter_std": 0.01,
            "jitter_clip": 0.05,
            "seed": seed,
            "renormalize_after": False,
            "scope": "training batches only; fresh draws each visit",
        }

    def __call__(self, points):
        if points.ndim != 3 or points.shape[-1] != 3 or not points.is_floating_point():
            raise ValueError("Expected floating point clouds of shape [B, N, 3]")
        options = {"device": points.device, "dtype": points.dtype,
                   "generator": self.generator}
        # 一个物体共享一个缩放因子和一个平移向量，各点分别添加噪声。
        scale = torch.rand((len(points), 1, 1), **options) * 0.4 + 0.8
        shift = torch.rand((len(points), 1, 3), **options) * 0.2 - 0.1
        noise = (torch.randn(points.shape, **options) * 0.01).clamp(-0.05, 0.05)
        return points * scale + shift + noise
