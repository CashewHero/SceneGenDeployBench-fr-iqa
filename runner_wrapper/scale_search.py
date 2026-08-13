from __future__ import annotations

"""Small, dependency-free helpers for signed 3DGS scale searches."""

import math
from typing import Any


HIGH_LOSS_TRIM_FRACTION = 0.05


def upper_trim_count(
    item_count: int,
    trim_fraction: float = HIGH_LOSS_TRIM_FRACTION,
) -> int:
    if item_count < 0:
        raise ValueError("item_count must be non-negative")
    if not 0.0 <= trim_fraction < 1.0:
        raise ValueError("trim_fraction must be at least 0 and less than 1")
    return math.floor(item_count * trim_fraction)


def scene_scale(output_metadata: Any) -> float:
    if output_metadata is None:
        return 1.0
    if not isinstance(output_metadata, dict):
        raise ValueError("job.primary_output_metadata must be an object")
    value = output_metadata.get("scene_scale", 1.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            "job.primary_output_metadata.scene_scale must be a nonzero finite number"
        )
    result = float(value)
    if not math.isfinite(result) or result == 0:
        raise ValueError(
            "job.primary_output_metadata.scene_scale must be a nonzero finite number"
        )
    return result


def resolve_scene_scale(
    metadata_scale: float,
    override: Any,
    parameter_name: str = "initial_scene_scale_overwrite",
) -> float:
    if override is None:
        return metadata_scale
    if isinstance(override, bool) or not isinstance(override, (int, float)):
        raise ValueError(
            f"job.parameters.{parameter_name} must be a finite number"
        )
    result = float(override)
    if not math.isfinite(result):
        raise ValueError(
            f"job.parameters.{parameter_name} must be a finite number"
        )
    return metadata_scale if result == 0 else result


def initial_scale(
    metadata_scale: float,
    depth_matched_scale: float | None,
    mode: str,
    hybrid_depth_weight: float,
) -> float:
    if mode == "depth":
        if depth_matched_scale is None:
            raise ValueError("depth mode requires usable ground-truth depth")
        return depth_matched_scale
    if mode == "hybrid":
        if depth_matched_scale is None:
            raise ValueError("hybrid mode requires usable ground-truth depth")
        return (
            depth_matched_scale * hybrid_depth_weight
            + metadata_scale * (1.0 - hybrid_depth_weight)
        )
    return metadata_scale


def upper_trimmed_weighted_mean(
    weighted_losses: list[tuple[float, float]],
    trim_fraction: float = HIGH_LOSS_TRIM_FRACTION,
) -> float | None:
    """Return a weighted mean after dropping the highest-loss fraction by count."""
    if not weighted_losses:
        return None
    trim_count = upper_trim_count(len(weighted_losses), trim_fraction)
    retained = sorted(weighted_losses, key=lambda item: item[0])[
        : len(weighted_losses) - trim_count
    ]
    weight_sum = math.fsum(weight for _, weight in retained)
    if weight_sum <= 0.0:
        raise ValueError("weighted losses must have a positive total weight")
    return math.fsum(loss * weight for loss, weight in retained) / weight_sum


def logarithmic_points(lower: float, upper: float, count: int) -> list[float]:
    if lower == 0 or upper == 0 or upper <= lower or (lower < 0) != (upper < 0):
        raise ValueError("scale search range must be nonzero, same-sign, and increasing")
    if count < 3:
        raise ValueError("search_points_per_round must be at least 3")
    sign = -1.0 if lower < 0 else 1.0
    log_lower = math.log(abs(lower))
    step = (math.log(abs(upper)) - log_lower) / (count - 1)
    return [sign * math.exp(log_lower + index * step) for index in range(count)]


def relative_accuracy(lower: float, upper: float, center: float) -> float:
    if center == 0:
        raise ValueError("scale search center must be nonzero")
    return abs(upper - lower) / abs(center)


def next_range(
    points: list[float],
    best_index: int,
    scale_range_factor: float,
) -> tuple[float, float, bool]:
    if len(points) < 3:
        raise ValueError("scale search requires at least three points")
    if not 0 <= best_index < len(points):
        raise ValueError("best scale index is outside the search points")
    if scale_range_factor <= 1:
        raise ValueError("scale_range_factor must be greater than 1")
    if best_index == 0:
        edge = (
            points[0] * scale_range_factor
            if points[0] < 0
            else points[0] / scale_range_factor
        )
        return edge, points[1], True
    if best_index == len(points) - 1:
        edge = (
            points[-1] / scale_range_factor
            if points[-1] < 0
            else points[-1] * scale_range_factor
        )
        return points[-2], edge, True
    return points[best_index - 1], points[best_index + 1], False
