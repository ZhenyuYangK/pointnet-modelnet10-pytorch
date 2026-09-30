"""Read ModelNet10 OFF meshes and sample normalized surface point clouds."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_off(path: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    """Read an OFF mesh and return its vertices and polygon faces.

    Faces are returned as arrays of vertex indices. Polygons with more than
    three vertices are later split into triangles using a fan from vertex 0.
    """
    tokens: list[str] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            # OFF permits comments beginning with '#'.
            line = line.split("#", maxsplit=1)[0]
            tokens.extend(line.split())

    if not tokens or tokens[0] != "OFF":
        raise ValueError(f"{path} is not a valid OFF file (missing OFF header)")

    cursor = 1
    if len(tokens) < cursor + 3:
        raise ValueError(f"{path} has an incomplete OFF count header")

    vertex_count, face_count, _ = map(int, tokens[cursor : cursor + 3])
    cursor += 3

    expected_vertex_tokens = vertex_count * 3
    if len(tokens) < cursor + expected_vertex_tokens:
        raise ValueError(f"{path} ends before all vertex coordinates are read")

    vertices = np.asarray(
        tokens[cursor : cursor + expected_vertex_tokens], dtype=np.float32
    ).reshape(vertex_count, 3)
    cursor += expected_vertex_tokens

    faces: list[np.ndarray] = []
    for face_number in range(face_count):
        if cursor >= len(tokens):
            raise ValueError(f"{path} ends before face {face_number} is read")

        polygon_size = int(tokens[cursor])
        cursor += 1
        if polygon_size < 3 or len(tokens) < cursor + polygon_size:
            raise ValueError(f"{path} contains an invalid face {face_number}")

        face = np.asarray(tokens[cursor : cursor + polygon_size], dtype=np.int64)
        cursor += polygon_size
        if np.any(face < 0) or np.any(face >= vertex_count):
            raise ValueError(f"{path} face {face_number} has an invalid vertex index")
        faces.append(face)

    if not np.isfinite(vertices).all():
        raise ValueError(f"{path} contains a non-finite vertex coordinate")
    return vertices, faces


def sample_surface_points(
    vertices: np.ndarray,
    faces: list[np.ndarray],
    num_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample points from mesh triangles, weighted by triangle surface area."""
    if num_points <= 0:
        raise ValueError("num_points must be greater than zero")

    # Turn polygon faces into triangles. ModelNet10 is triangulated, while the
    # fan split also handles simple polygon faces in other OFF files.
    triangles = [
        [face[0], face[index], face[index + 1]]
        for face in faces
        for index in range(1, len(face) - 1)
    ]
    if not triangles:
        raise ValueError("The mesh has no faces to sample")

    triangles_array = np.asarray(triangles, dtype=np.int64)
    triangle_vertices = vertices[triangles_array]
    edge_a = triangle_vertices[:, 1] - triangle_vertices[:, 0]
    edge_b = triangle_vertices[:, 2] - triangle_vertices[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(edge_a, edge_b), axis=1)

    # Zero-area faces cannot contribute surface points.
    valid = areas > 0
    triangle_vertices = triangle_vertices[valid]
    areas = areas[valid]
    if len(areas) == 0:
        raise ValueError("The mesh has no non-degenerate triangle faces")

    triangle_indices = rng.choice(
        len(areas), size=num_points, replace=True, p=areas / areas.sum()
    )
    chosen = triangle_vertices[triangle_indices]

    # Uniform sampling inside each triangle using reflected barycentric values.
    uv = rng.random((num_points, 2))
    reflected = uv.sum(axis=1) > 1
    uv[reflected] = 1 - uv[reflected]
    points = (
        chosen[:, 0]
        + uv[:, :1] * (chosen[:, 1] - chosen[:, 0])
        + uv[:, 1:] * (chosen[:, 2] - chosen[:, 0])
    )
    return points.astype(np.float32, copy=False)


def normalize_points(points: np.ndarray) -> np.ndarray:
    """Center a point cloud and scale it to fit inside a unit sphere."""
    points = points - points.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(points, axis=1).max()
    if radius > 0:
        points = points / radius
    return points.astype(np.float32, copy=False)


class ModelNet10(Dataset):
    """ModelNet10 classification dataset.

    Args:
        split: ``"train"`` or ``"test"``; uses the official directory split.
        root: Dataset directory containing the ten class folders. Defaults to
            ``<project root>/data/ModelNet10``.
        num_points: Number of surface points returned for each mesh.
        seed: Optional seed for repeatable point sampling per item. If omitted,
            points are resampled each time an item is read.

    Each item is ``(points, label)`` where ``points`` has shape ``[N, 3]`` and
    ``label`` is an integer in ``[0, 9]``.
    """

    def __init__(
        self,
        split: str = "train",
        root: Optional[str | Path] = None,
        num_points: int = 1024,
        seed: Optional[int] = None,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError("split must be either 'train' or 'test'")
        if num_points <= 0:
            raise ValueError("num_points must be greater than zero")

        self.root = Path(root) if root is not None else PROJECT_ROOT / "data" / "ModelNet10"
        self.split = split
        self.num_points = num_points
        self.seed = seed

        if not self.root.is_dir():
            raise FileNotFoundError(f"ModelNet10 directory not found: {self.root}")

        self.classes = sorted(path.name for path in self.root.iterdir() if path.is_dir())
        if len(self.classes) != 10:
            raise ValueError(
                f"Expected 10 class folders in {self.root}, found {len(self.classes)}"
            )
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}

        self.samples: list[tuple[Path, int]] = []
        for class_name in self.classes:
            split_dir = self.root / class_name / split
            if not split_dir.is_dir():
                raise FileNotFoundError(f"Missing split directory: {split_dir}")
            for path in sorted(split_dir.glob("*.off")):
                self.samples.append((path, self.class_to_idx[class_name]))

        if not self.samples:
            raise ValueError(f"No OFF files found for split '{split}' in {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        mesh_path, label = self.samples[index]
        vertices, faces = read_off(mesh_path)

        if self.seed is None:
            rng = np.random.default_rng()
        else:
            # The same item always gets the same sample when a seed is supplied.
            rng = np.random.default_rng(self.seed + index)

        points = sample_surface_points(vertices, faces, self.num_points, rng)
        points = normalize_points(points)
        return torch.from_numpy(points), label
