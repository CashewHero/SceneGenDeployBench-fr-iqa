from __future__ import annotations

"""Load Graphdeco 3DGS PLY files and render equirectangular camera views."""

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from equilib import cube2equi
from gsplat.rendering import rasterization
from plyfile import PlyData


def _properties(vertex: Any, names: list[str]) -> np.ndarray:
    return np.stack([np.asarray(vertex[name], dtype=np.float32) for name in names], axis=-1)


def _numbered_properties(property_names: set[str], prefix: str) -> list[str]:
    names = [name for name in property_names if name.startswith(prefix)]
    return sorted(names, key=lambda name: int(name.removeprefix(prefix)))


def load_graphdeco_ply(path: Path, device: torch.device) -> dict[str, torch.Tensor | int]:
    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        raise ValueError("3DGS PLY does not contain a vertex element")
    vertex = ply["vertex"].data
    names = set(vertex.dtype.names or ())
    required = {
        "x",
        "y",
        "z",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
    }
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"unsupported 3DGS PLY; missing properties: {', '.join(missing)}")

    rest_names = _numbered_properties(names, "f_rest_")
    if len(rest_names) % 3:
        raise ValueError("unsupported 3DGS PLY; f_rest properties are not RGB-aligned")
    coefficient_count = 1 + len(rest_names) // 3
    sh_degree = math.isqrt(coefficient_count) - 1
    if (sh_degree + 1) ** 2 != coefficient_count:
        raise ValueError("unsupported 3DGS PLY; spherical-harmonic coefficient count is invalid")

    means = _properties(vertex, ["x", "y", "z"])
    scales = np.exp(_properties(vertex, ["scale_0", "scale_1", "scale_2"]))
    opacities = 1.0 / (1.0 + np.exp(-np.asarray(vertex["opacity"], dtype=np.float32)))
    quats = _properties(vertex, ["rot_0", "rot_1", "rot_2", "rot_3"])
    quats /= np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12)

    dc = _properties(vertex, ["f_dc_0", "f_dc_1", "f_dc_2"])[:, None, :]
    if rest_names:
        # Graphdeco stores channel-major coefficients; gsplat expects coefficient-major RGB.
        rest = _properties(vertex, rest_names).reshape(len(vertex), 3, -1).transpose(0, 2, 1)
        colors = np.concatenate((dc, rest), axis=1)
    else:
        colors = dc

    return {
        "means": torch.from_numpy(means).to(device),
        "quats": torch.from_numpy(quats).to(device),
        "scales": torch.from_numpy(scales).to(device),
        "opacities": torch.from_numpy(opacities).to(device),
        "colors": torch.from_numpy(colors).to(device),
        "sh_degree": sh_degree,
    }


def _rotation_x(degrees: float, device: torch.device) -> torch.Tensor:
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    return torch.tensor(
        [[1, 0, 0, 0], [0, cosine, -sine, 0], [0, sine, cosine, 0], [0, 0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )


def _rotation_y(degrees: float, device: torch.device) -> torch.Tensor:
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    return torch.tensor(
        [[cosine, 0, sine, 0], [0, 1, 0, 0], [-sine, 0, cosine, 0], [0, 0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )


def _canonical_cube_viewmats(device: torch.device) -> list[torch.Tensor]:
    # Render cubemap faces in canonical panorama order using Graphdeco camera axes.
    cube_views = [
        # Equilib order: front, right, back, left, up, down.
        _rotation_y(270, device),
        torch.eye(4, dtype=torch.float32, device=device),
        _rotation_y(90, device),
        _rotation_y(180, device),
        _rotation_x(90, device) @ _rotation_y(270, device),
        _rotation_x(-90, device) @ _rotation_y(270, device),
    ]
    graphdeco_camera_axes = torch.diag(
        torch.tensor([-1, -1, 1, 1], dtype=torch.float32, device=device)
    )
    return [graphdeco_camera_axes @ view for view in cube_views]


@torch.inference_mode()
def render_panorama_outputs(
    splats: dict[str, torch.Tensor | int],
    width: int,
    height: int,
    device: torch.device,
    world_to_camera: torch.Tensor | None = None,
    *,
    include_depth: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if width != height * 2:
        raise ValueError(
            f"reference image must be a 2:1 equirectangular panorama, got {width}x{height}"
        )
    if width % 8 or height % 8:
        raise ValueError("reference panorama width and height must be divisible by 8")

    if world_to_camera is None:
        world_to_camera = torch.eye(4, dtype=torch.float32, device=device)
    else:
        world_to_camera = world_to_camera.to(device=device, dtype=torch.float32)
    if tuple(world_to_camera.shape) != (4, 4):
        raise ValueError("world_to_camera must be a 4x4 matrix")

    face_size = width // 4
    focal = face_size / 2.0
    intrinsic = torch.tensor(
        [[focal, 0, face_size / 2.0], [0, focal, face_size / 2.0], [0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )[None]
    rgb_faces: list[torch.Tensor] = []
    depth_faces: list[torch.Tensor] = []
    alpha_faces: list[torch.Tensor] = []
    pixel_coordinates = (
        torch.arange(face_size, dtype=torch.float32, device=device) + 0.5
    )
    normalized = (pixel_coordinates - face_size / 2.0) / focal
    normalized_x, normalized_y = torch.meshgrid(normalized, normalized, indexing="xy")
    radial_depth_factor = torch.sqrt(
        1.0 + normalized_x.square() + normalized_y.square()
    )
    for cube_viewmat in _canonical_cube_viewmats(device):
        viewmat = cube_viewmat @ world_to_camera
        render, alpha, _ = rasterization(
            means=splats["means"],
            quats=splats["quats"],
            scales=splats["scales"],
            opacities=splats["opacities"],
            colors=splats["colors"],
            viewmats=viewmat[None],
            Ks=intrinsic,
            width=face_size,
            height=face_size,
            sh_degree=int(splats["sh_degree"]),
            packed=True,
            render_mode="RGB+ED" if include_depth else "RGB",
        )
        rgb = render[0, ..., :3] + (1.0 - alpha[0])
        rgb_faces.append(rgb.permute(2, 0, 1).clamp(0, 1))
        if include_depth:
            # gsplat's expected depth is along the cube-face optical axis. Convert
            # it to radial camera distance before assembling the panorama.
            depth_faces.append(
                (render[0, ..., 3] * radial_depth_factor).unsqueeze(0)
            )
            alpha_faces.append(alpha[0, ..., 0].unsqueeze(0).clamp(0, 1))

    panorama = cube2equi(
        cubemap=rgb_faces,
        cube_format="list",
        height=height,
        width=width,
        mode="bilinear",
    )
    if not include_depth:
        return panorama.clamp(0, 1), None, None
    depth = cube2equi(
        cubemap=depth_faces,
        cube_format="list",
        height=height,
        width=width,
        mode="bilinear",
    )[0]
    alpha = cube2equi(
        cubemap=alpha_faces,
        cube_format="list",
        height=height,
        width=width,
        mode="bilinear",
    )[0]
    return panorama.clamp(0, 1), depth, alpha.clamp(0, 1)


def render_panorama(
    splats: dict[str, torch.Tensor | int],
    width: int,
    height: int,
    device: torch.device,
    world_to_camera: torch.Tensor | None = None,
) -> torch.Tensor:
    image, _, _ = render_panorama_outputs(
        splats,
        width,
        height,
        device,
        world_to_camera,
    )
    return image
