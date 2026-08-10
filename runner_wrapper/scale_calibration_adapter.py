from __future__ import annotations

"""Calibrate generated-scene scale from nearby image and depth references."""

import csv
import logging
import math
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from runner_wrapper.fr_iqa_adapter import (
    METRIC_ALIASES,
    SUPPORTED_METRICS,
    create_image_quality_evaluator,
    _evaluate,
    _load_rgb_image,
    _normalize_inputs,
    _timestamp,
    _variant_key,
    _write_metrics_file,
    event_message,
)
from runner_wrapper.gsplat_renderer import load_graphdeco_ply, render_panorama_outputs
from runner_wrapper.job_logging import tee_job_output
from runner_wrapper.measurements import ResourceMonitor
from runner_wrapper.render_3dgs_adapter import (
    _max_distance,
    _max_references,
    _normalized_world_to_camera,
    _required_file,
    _save_image,
)
from runner_wrapper.scale_search import (
    initial_scale,
    logarithmic_points,
    next_range,
    relative_accuracy,
    resolve_scene_scale,
    scene_scale,
)

logger = logging.getLogger("runner_wrapper.scale_calibration_adapter")

SCALE_FIELDS_PREFIX = [
    "sample", "mode", "initial_scale", "scale_range_factor", "target_accuracy",
    "view_distance_weight_half_life",
    "round", "point", "evaluation_id", "range_min", "range_max", "scale",
    "is_best", "is_edge", "objective_score", "objective_p10", "objective_p90",
    "depth_log_l1_median",
]
SCALE_FIELDS_SUFFIX = [
    "reference_count", "valid_reference_count", "converged", "achieved_accuracy",
]
REFERENCE_FIELDS_PREFIX = [
    "evaluation_id", "scale", "reference_sample", "camera_distance",
    "distance_weight", "used", "rejection_reason", "valid_depth_fraction",
    "depth_log_l1",
]
REFERENCE_FIELDS_SUFFIX = [
    "rgb_path", "depth_path",
    "alpha_path", "rgb_error_path", "depth_error_path", "depth_filter_path",
    "ground_truth_rgb_path", "ground_truth_depth_path",
]
RENDER_TYPES = (
    "rgb", "depth", "alpha", "rgb_error", "depth_error", "depth_filter",
)

INFERNO_STOPS = np.asarray(
    [
        (0.000, 0, 0, 4),
        (0.125, 31, 12, 72),
        (0.250, 85, 15, 109),
        (0.375, 136, 34, 106),
        (0.500, 187, 55, 84),
        (0.625, 227, 89, 51),
        (0.750, 249, 140, 10),
        (0.875, 249, 201, 50),
        (1.000, 252, 255, 164),
    ],
    dtype=np.float32,
)

DEPTH_FILTER_LABELS = (
    (0, "valid", (45, 180, 80)),
    (1, "GT invalid", (110, 110, 110)),
    (2, "GT too close", (30, 190, 210)),
    (3, "GT too far", (50, 100, 220)),
    (4, "render invalid", (210, 50, 190)),
    (5, "low alpha", (235, 135, 35)),
)


def _number(
    parameters: dict[str, Any],
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    value = parameters.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"job.parameters.{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"job.parameters.{name} must be a finite number")
    if minimum is not None and (
        result < minimum or (strict_minimum and result <= minimum)
    ):
        comparator = "greater than" if strict_minimum else "at least"
        raise ValueError(f"job.parameters.{name} must be {comparator} {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"job.parameters.{name} must be at most {maximum}")
    return result


def _integer(parameters: dict[str, Any], name: str, default: int, minimum: int) -> int:
    value = parameters.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"job.parameters.{name} must be an integer of at least {minimum}")
    return value


def _choice(parameters: dict[str, Any], name: str, default: str, choices: set[str]) -> str:
    value = str(parameters.get(name, default)).strip().lower()
    if value not in choices:
        raise ValueError(
            f"job.parameters.{name} must be one of {', '.join(sorted(choices))}"
        )
    return value


def _save_render_settings(
    parameters: dict[str, Any],
) -> tuple[str, set[str], bool]:
    raw_settings = parameters.get("save_renders") or {}
    if not isinstance(raw_settings, dict):
        raise ValueError("job.parameters.save_renders must be an object")

    scales = str(raw_settings.get("scales", "none")).strip().lower()
    if scales not in {"none", "best", "all"}:
        raise ValueError(
            "job.parameters.save_renders.scales must be one of all, best, none"
        )

    raw_types = raw_settings.get("types", "all")
    if isinstance(raw_types, str):
        if raw_types.strip().lower() != "all":
            raise ValueError(
                "job.parameters.save_renders.types must be all or a list"
            )
        raw_types = list(RENDER_TYPES)
    if not isinstance(raw_types, list):
        raise ValueError("job.parameters.save_renders.types must be all or a list")
    render_types: set[str] = set()
    for raw_type in raw_types:
        render_type = str(raw_type).strip().lower()
        if render_type not in RENDER_TYPES:
            raise ValueError(
                "job.parameters.save_renders.types entries must be one of "
                + ", ".join(RENDER_TYPES)
            )
        render_types.add(render_type)
    if scales != "none" and not render_types:
        raise ValueError(
            "job.parameters.save_renders.types must not be empty when scales are saved"
        )

    copy_ground_truth = raw_settings.get("copy_ground_truth", False)
    if not isinstance(copy_ground_truth, bool):
        raise ValueError(
            "job.parameters.save_renders.copy_ground_truth must be a boolean"
        )
    if copy_ground_truth and scales == "none":
        raise ValueError(
            "job.parameters.save_renders.copy_ground_truth requires scales best or all"
        )
    return scales, render_types, copy_ground_truth


def _render_types_for_mode(render_types: set[str], mode: str) -> set[str]:
    active = set(render_types)
    if mode not in {"image", "hybrid"}:
        active.difference_update({"rgb", "rgb_error"})
    if mode not in {"depth", "hybrid"}:
        active.difference_update({"depth", "depth_error", "depth_filter"})
    return active


def _image_metric(parameters: dict[str, Any]) -> str:
    metric = str(parameters.get("metric", "lpips")).strip().lower()
    metric = METRIC_ALIASES.get(metric, metric)
    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"job.parameters.metric must be one of {', '.join(SUPPORTED_METRICS)}"
        )
    return metric


def _image_metrics_for_mode(mode: str, metric: str) -> list[str]:
    return [metric] if mode in {"image", "hybrid"} else []


def _calibration_output_variant(mode: str, metric: str, variant: str) -> str:
    metric_label = metric if mode != "depth" else "logl1"
    variant_hash = variant.rsplit("-", 1)[-1]
    return f"{_safe_name(mode)}-{_safe_name(metric_label)}-{variant_hash}"


def _load_depth(path: Path, encoding: str) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        depth = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as source:
            depth = np.asarray(source)
    if encoding == "tartanair_float32_rgba" or (
        depth.dtype == np.uint8 and depth.ndim == 3 and depth.shape[-1] == 4
    ):
        depth = np.squeeze(np.ascontiguousarray(depth).view("<f4"), axis=-1)
    elif depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"ground-truth depth must be a single-channel image: {path}")
    return np.asarray(depth, dtype=np.float32)


def _safe_name(value: str) -> str:
    normalized = value.strip().replace("\\", "-").replace("/", "-")
    if not normalized or normalized in {".", ".."}:
        raise ValueError(f"invalid sample id: {value!r}")
    return normalized


def _save_gray(values: np.ndarray, path: Path) -> None:
    image = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8), mode="L").save(path)


def _inferno(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    positions = INFERNO_STOPS[:, 0]
    channels = [
        np.interp(clipped, positions, INFERNO_STOPS[:, channel])
        for channel in range(1, 4)
    ]
    return np.stack(channels, axis=-1).astype(np.uint8)


def _save_error_heatmap(
    values: np.ndarray,
    path: Path,
    label: str,
    valid: np.ndarray | None = None,
) -> None:
    values = np.asarray(values, dtype=np.float32)
    usable = np.isfinite(values)
    if valid is not None:
        usable &= valid
    high = float(np.quantile(values[usable], 0.95)) if usable.any() else 1.0
    high = max(high, 1e-12)
    colors = _inferno(np.nan_to_num(values / high, nan=0.0, posinf=1.0, neginf=0.0))
    colors[~usable] = (55, 55, 55)

    height, width = values.shape
    footer_height = 38
    canvas = Image.new("RGB", (width, height + footer_height), "white")
    canvas.paste(Image.fromarray(colors, mode="RGB"), (0, 0))
    draw = ImageDraw.Draw(canvas)
    bar_left = min(8, max(0, width - 1))
    bar_right = max(bar_left + 1, width - 8)
    bar_top = height + 6
    bar_height = 10
    gradient = np.linspace(0.0, 1.0, max(1, bar_right - bar_left), dtype=np.float32)
    gradient_rgb = _inferno(np.repeat(gradient[None, :], bar_height, axis=0))
    canvas.paste(Image.fromarray(gradient_rgb, mode="RGB"), (bar_left, bar_top))
    draw.text((bar_left, height + 19), f"{label}: 0", fill="black")
    high_label = f"p95={high:.4g}"
    text_width = draw.textlength(high_label)
    draw.text((max(bar_left, bar_right - text_width), height + 19), high_label, fill="black")
    canvas.save(path)


def _depth_diagnostics(
    ground_truth: np.ndarray,
    predicted: np.ndarray,
    rendered_alpha: np.ndarray,
    min_gt_depth: float,
    max_gt_depth: float,
    min_alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reason = np.ones(ground_truth.shape, dtype=np.uint8)
    finite_positive_gt = np.isfinite(ground_truth) & (ground_truth > 0)
    reason[finite_positive_gt & (ground_truth < min_gt_depth)] = 2
    reason[finite_positive_gt & (ground_truth > max_gt_depth)] = 3
    in_depth_range = finite_positive_gt & (ground_truth >= min_gt_depth) & (
        ground_truth <= max_gt_depth
    )
    valid_prediction = np.isfinite(predicted) & (predicted > 0)
    reason[in_depth_range & ~valid_prediction] = 4
    valid_alpha = np.isfinite(rendered_alpha) & (rendered_alpha >= min_alpha)
    reason[in_depth_range & valid_prediction & ~valid_alpha] = 5
    valid = in_depth_range & valid_prediction & valid_alpha
    reason[valid] = 0
    error = np.full(ground_truth.shape, np.nan, dtype=np.float32)
    error[valid] = np.abs(np.log(predicted[valid] / ground_truth[valid]))
    return valid, error, reason


def _save_depth_filter(
    ground_truth: np.ndarray,
    reasons: np.ndarray,
    path: Path,
    min_gt_depth: float,
    max_gt_depth: float,
) -> None:
    positive = np.isfinite(ground_truth) & (ground_truth > 0)
    clipped = np.clip(ground_truth, max(min_gt_depth, 1e-12), max_gt_depth)
    log_min = math.log(max(min_gt_depth, 1e-12))
    log_max = math.log(max_gt_depth)
    if math.isclose(log_min, log_max):
        brightness = np.ones(ground_truth.shape, dtype=np.float32)
    else:
        normalized = (np.log(clipped) - log_min) / (log_max - log_min)
        brightness = 0.3 + 0.7 * (1.0 - np.clip(normalized, 0.0, 1.0))
    brightness[~positive] = 0.45

    colors = np.zeros((*ground_truth.shape, 3), dtype=np.uint8)
    for code, _, base_color in DEPTH_FILTER_LABELS:
        selected = reasons == code
        colors[selected] = np.clip(
            np.asarray(base_color, dtype=np.float32) * brightness[selected, None],
            0,
            255,
        ).astype(np.uint8)

    height, width = ground_truth.shape
    footer_height = 40
    canvas = Image.new("RGB", (width, height + footer_height), "white")
    canvas.paste(Image.fromarray(colors, mode="RGB"), (0, 0))
    draw = ImageDraw.Draw(canvas)
    x = 6
    y = height + 7
    for _, label, color in DEPTH_FILTER_LABELS:
        draw.rectangle((x, y, x + 9, y + 9), fill=color)
        draw.text((x + 13, y - 2), label, fill="black")
        x += int(draw.textlength(label)) + 31
        if x + 90 > width:
            x = 6
            y += 17
    canvas.save(path)


def _metric_value(metrics: list[dict[str, Any]], name: str) -> float | None:
    for metric in metrics:
        if metric["name"] != name:
            continue
        if isinstance(metric["value"], (int, float)):
            value = float(metric["value"])
            return value if math.isfinite(value) else None
        if metric["value"] == "infinity" and name in {"psnr", "ws_psnr"}:
            return math.inf
    return None


def _image_loss(metric_name: str, value: float) -> float:
    if metric_name in {"psnr", "ws_psnr"}:
        return 10.0 ** (-value / 10.0)
    if metric_name == "ssim":
        return max(0.0, 1.0 - value)
    return max(0.0, value)


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _weighted_quantile(
    weighted_values: list[tuple[float, float]], quantile: float
) -> float | None:
    if not weighted_values:
        return None
    values = np.asarray([item[0] for item in weighted_values], dtype=np.float64)
    weights = np.asarray([item[1] for item in weighted_values], dtype=np.float64)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    positions = (np.cumsum(weights) - 0.5 * weights) / weights.sum()
    return float(np.interp(quantile, positions, values, left=values[0], right=values[-1]))


def _calibration_metric(
    name: str,
    value: float | int | bool,
    metric_type: str,
    unit: str | None = None,
) -> dict[str, Any]:
    metric: dict[str, Any] = {
        "namespace": "scale_calibration",
        "name": name,
        "type": metric_type,
        "value": value,
        "source": "evaluator",
    }
    if unit:
        metric["unit"] = unit
    return metric


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_scale_chart(rows: list[dict[str, Any]], path: Path) -> bool:
    points_by_scale: dict[float, float] = {}
    for row in rows:
        if row.get("objective_score") in (None, ""):
            continue
        scale = float(row["scale"])
        score = float(row["objective_score"])
        if scale != 0 and math.isfinite(scale) and math.isfinite(score):
            points_by_scale[scale] = score
    points = sorted(points_by_scale.items())
    if len(points) < 2:
        return False
    sign = -1.0 if points[0][0] < 0 else 1.0
    if any((scale < 0) != (sign < 0) for scale, _ in points):
        return False
    width, height = 1000, 420
    left, right, top, bottom = 90, width - 30, 35, height - 65
    chart = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(chart)
    draw.text((left, 10), "Scale calibration objective", fill="black")
    log_scales = [sign * math.log(abs(point[0])) for point in points]
    scores = [point[1] for point in points]
    x_min, x_max = min(log_scales), max(log_scales)
    y_min, y_max = min(scores), max(scores)
    if math.isclose(y_min, y_max):
        padding = max(abs(y_min) * 0.05, 1e-6)
        y_min -= padding
        y_max += padding
    x_span, y_span = x_max - x_min, y_max - y_min
    tick_intervals = 5

    def centered_text(x: float, y: float, text: str) -> None:
        bounds = draw.textbbox((0, 0), text)
        draw.text((x - (bounds[2] - bounds[0]) / 2, y), text, fill="#374151")

    def right_aligned_text(x: float, y: float, text: str) -> None:
        bounds = draw.textbbox((0, 0), text)
        draw.text((x - (bounds[2] - bounds[0]), y), text, fill="#374151")

    for tick_index in range(tick_intervals + 1):
        fraction = tick_index / tick_intervals
        x = left + fraction * (right - left)
        y = bottom - fraction * (bottom - top)
        if tick_index not in {0, tick_intervals}:
            draw.line((x, top, x, bottom), fill="#e5e7eb", width=1)
            draw.line((left, y, right, y), fill="#e5e7eb", width=1)
        draw.line((x, bottom, x, bottom + 4), fill="#374151", width=1)
        draw.line((left - 4, y, left, y), fill="#374151", width=1)
        coordinate = x_min + fraction * x_span
        centered_text(x, bottom + 8, f"{sign * math.exp(sign * coordinate):.5g}")
        right_aligned_text(
            left - 8,
            y - 6,
            f"{y_min + fraction * y_span:.5g}",
        )
    draw.line((left, top, left, bottom), fill="#374151", width=2)
    draw.line((left, bottom, right, bottom), fill="#374151", width=2)
    plotted = [
        (
            left + (sign * math.log(abs(scale)) - x_min) / x_span * (right - left),
            bottom - (score - y_min) / y_span * (bottom - top),
        )
        for scale, score in points
    ]
    draw.line(plotted, fill="#2563eb", width=2)
    for x, y in plotted:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#2563eb")
    draw.text(((left + right) // 2 - 30, bottom + 31), "scale (log)", fill="#374151")
    chart.save(path)
    return True


def run_job(job_request: dict[str, Any]) -> dict[str, Any]:
    started_at = time.time()
    workspace_root = Path(job_request["runtime"]["workspace_dir"])
    workspace_root.mkdir(parents=True, exist_ok=True)
    variant = _variant_key(job_request, "scale")
    log_name = f"runner-{variant}.log"
    with tee_job_output(workspace_root / log_name):
        return _run_job_logged(job_request, started_at, workspace_root, variant, log_name)


def _run_job_logged(
    job_request: dict[str, Any],
    started_at: float,
    workspace_root: Path,
    variant: str,
    log_name: str,
) -> dict[str, Any]:
    monitor: ResourceMonitor | None = None
    metrics_name = f"metrics-{variant}.json"
    metrics_path = workspace_root / metrics_name
    output_variant: str | None = None
    inputs: dict[str, dict[str, dict[str, Any]]] = {}
    parameters: dict[str, Any] = {}
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("3DGS scale calibration requires an NVIDIA CUDA GPU")
        job = job_request["job"]
        primary_sample = str(job["primary_sample"])
        inputs = _normalize_inputs(job_request.get("inputs"))
        data = inputs.get("data", {}).get(primary_sample, {})
        candidate = inputs.get("candidate", {}).get(primary_sample)
        splat_path = _required_file(
            candidate,
            "3dgs",
            f"inputs.candidate.{primary_sample}.3dgs",
        )
        raw_parameters = job.get("parameters") or {}
        if not isinstance(raw_parameters, dict):
            raise ValueError("job.parameters must be an object")
        parameters = dict(raw_parameters)
        requested_mode = _choice(
            parameters, "comparison_mode", "auto", {"auto", "image", "depth", "hybrid"}
        )
        depth_weight = _number(
            parameters, "hybrid_depth_weight", 0.7, minimum=0.0, maximum=1.0
        )
        range_factor = _number(
            parameters, "scale_range_factor", 1.5, minimum=1.0, strict_minimum=True
        )
        target_accuracy = _number(
            parameters, "target_accuracy", 0.001, minimum=0.0, strict_minimum=True
        )
        points_per_round = _integer(parameters, "search_points_per_round", 11, 3)
        max_rounds = _integer(parameters, "max_search_rounds", 10, 1)
        max_distance = _max_distance(parameters, default=10.0)
        max_references = _max_references(parameters, default=100)
        min_gt_depth = _number(
            parameters, "min_ground_truth_depth", 0.1, minimum=0.0
        )
        max_gt_depth = _number(
            parameters, "max_ground_truth_depth", 50.0, minimum=0.0, strict_minimum=True
        )
        if max_gt_depth <= min_gt_depth:
            raise ValueError(
                "job.parameters.max_ground_truth_depth must exceed min_ground_truth_depth"
            )
        min_alpha = _number(
            parameters, "depth_min_render_alpha", 0.3, minimum=0.0, maximum=1.0
        )
        min_valid_fraction = _number(
            parameters, "depth_min_valid_fraction", 0.3, minimum=0.0, maximum=1.0
        )
        distance_weight_half_life = _number(
            parameters,
            "view_distance_weight_half_life",
            2.5,
            minimum=0.0,
            strict_minimum=True,
        )
        image_metric = _image_metric(parameters)
        saved_scales, render_types, copy_ground_truth = _save_render_settings(
            parameters
        )
        metadata_scale = scene_scale(job.get("primary_output_metadata"))
        starting_scene_scale = resolve_scene_scale(
            metadata_scale,
            parameters.get("initial_scene_scale_overwrite"),
        )
        sample_metadata = job.get("primary_sample_metadata") or {}
        if not isinstance(sample_metadata, dict):
            raise ValueError("job.primary_sample_metadata must be an object")
        pose_convention = str(
            sample_metadata.get("pose_convention") or "camera_to_world"
        ).strip()
        pose_coordinate_system = str(
            sample_metadata.get("pose_coordinate_system") or ""
        ).strip()
        depth_metadata = sample_metadata.get("depth") or {}
        depth_encoding = (
            str(depth_metadata.get("encoding") or "").strip()
            if isinstance(depth_metadata, dict)
            else ""
        )
        primary_pose = data.get("camera_pose") or {}

        views: list[dict[str, Any]] = []
        if data.get("depth"):
            views.append(
                {
                    "sample": primary_sample,
                    "role": "data",
                    "pose": primary_pose,
                    "distance": 0.0,
                    "depth_path": _required_file(
                        data, "depth", f"inputs.data.{primary_sample}.depth"
                    ),
                    "image_path": None,
                }
            )
        eligible: list[dict[str, Any]] = []
        skipped_references: list[dict[str, Any]] = []
        for sample_id, sample_data in inputs.get("references", {}).items():
            pose = sample_data.get("camera_pose")
            if pose is None:
                raise ValueError(f"inputs.references.{sample_id}.camera_pose is required")
            _, distance = _normalized_world_to_camera(
                primary_pose,
                pose,
                convention=pose_convention,
                primary_field=f"inputs.data.{primary_sample}.camera_pose",
                target_field=f"inputs.references.{sample_id}.camera_pose",
                coordinate_system=pose_coordinate_system,
            )
            if distance > max_distance:
                skipped_references.append(
                    {"sample": sample_id, "distance": distance, "reason": "distance_exceeds_maximum"}
                )
                continue
            if not sample_data.get("image") and not sample_data.get("depth"):
                skipped_references.append(
                    {"sample": sample_id, "distance": distance, "reason": "no_comparison_data"}
                )
                continue
            eligible.append(
                {
                    "sample": sample_id,
                    "role": "references",
                    "pose": pose,
                    "distance": distance,
                    "image_path": _required_file(
                        sample_data, "image", f"inputs.references.{sample_id}.image"
                    ) if sample_data.get("image") else None,
                    "depth_path": _required_file(
                        sample_data, "depth", f"inputs.references.{sample_id}.depth"
                    ) if sample_data.get("depth") else None,
                }
            )
        eligible.sort(key=lambda view: (view["distance"], view["sample"]))
        for skipped in eligible[max_references:]:
            skipped_references.append(
                {"sample": skipped["sample"], "distance": skipped["distance"], "reason": "reference_limit_exceeded"}
            )
        views.extend(eligible[:max_references])

        has_images = any(view["image_path"] for view in views)
        has_depth = any(view["depth_path"] for view in views)
        mode = requested_mode
        if mode == "auto":
            mode = "hybrid" if has_images and has_depth else "depth" if has_depth else "image"
        if mode in {"image", "hybrid"} and not has_images:
            raise ValueError(f"{mode} comparison requires at least one reference image")
        if mode in {"depth", "hybrid"} and not has_depth:
            raise ValueError(f"{mode} comparison requires at least one ground-truth depth map")
        uses_image = mode in {"image", "hybrid"}
        uses_depth = mode in {"depth", "hybrid"}
        selected_metrics = _image_metrics_for_mode(mode, image_metric)
        active_render_types = _render_types_for_mode(render_types, mode)
        output_variant = _calibration_output_variant(mode, image_metric, variant)
        metrics_name = f"metrics-{output_variant}.json"
        metrics_path = workspace_root / metrics_name
        renders_name = f"renders-{output_variant}"
        renders_directory = workspace_root / renders_name
        required_path = "image_path" if mode == "image" else "depth_path" if mode == "depth" else None
        if required_path is not None:
            filtered_views: list[dict[str, Any]] = []
            for view in views:
                if view[required_path]:
                    filtered_views.append(view)
                elif view["role"] == "references":
                    skipped_references.append(
                        {
                            "sample": view["sample"],
                            "distance": view["distance"],
                            "reason": f"missing_{required_path.removesuffix('_path')}_for_{mode}_mode",
                        }
                    )
            views = filtered_views

        for view in views:
            view["reference_image"] = None
            view["ground_truth_depth"] = None
            width = height = None
            if uses_image and view["image_path"]:
                reference_image, info = _load_rgb_image(view["image_path"])
                view["reference_image"] = reference_image
                width, height = int(info["width"]), int(info["height"])
            if uses_depth and view["depth_path"]:
                ground_truth_depth = _load_depth(view["depth_path"], depth_encoding)
                view["ground_truth_depth"] = ground_truth_depth
                depth_height, depth_width = ground_truth_depth.shape
                if width is not None and (width, height) != (depth_width, depth_height):
                    raise ValueError(
                        f"image and depth dimensions differ for {view['sample']}: "
                        f"image={width}x{height}, depth={depth_width}x{depth_height}"
                    )
                width, height = depth_width, depth_height
            view["width"], view["height"] = width, height

        monitor_data = {
            f"{role}.{sample_id}.{data_type}": value
            for role, samples in inputs.items()
            for sample_id, sample_data in samples.items()
            for data_type, value in sample_data.items()
        }
        monitor = ResourceMonitor(sample_data=monitor_data, output_dir=workspace_root)
        monitor.start()
        logger.info(
            event_message(
                "scale_calibration_started",
                job_id=job["job_id"],
                mode=mode,
                metadata_scene_scale=metadata_scale,
                starting_scene_scale=starting_scene_scale,
                view_count=len(views),
                skipped_reference_count=len(skipped_references),
                max_references=max_references,
                metric=image_metric if uses_image else None,
                metrics=selected_metrics,
                search_points_per_round=points_per_round,
                max_search_rounds=max_rounds,
                target_accuracy=target_accuracy,
                view_distance_weight_half_life=distance_weight_half_life,
            )
        )
        device = torch.device("cuda")
        load_started_at = time.time()
        splats = load_graphdeco_ply(splat_path, device)
        evaluator = (
            create_image_quality_evaluator(
                selected_metrics,
                device,
                workspace_root,
            )
            if uses_image
            else None
        )
        logger.info(
            event_message(
                "scale_calibration_resources_loaded",
                job_id=job["job_id"],
                gaussian_count=int(splats["means"].shape[0]),
                device=str(device),
                elapsed_seconds=round(time.time() - load_started_at, 3),
            )
        )

        def render_view(view: dict[str, Any], scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            world_to_camera, _ = _normalized_world_to_camera(
                primary_pose,
                view["pose"],
                convention=pose_convention,
                primary_field=f"inputs.data.{primary_sample}.camera_pose",
                target_field=f"inputs.{view['role']}.{view['sample']}.camera_pose",
                scene_scale=scale,
                coordinate_system=pose_coordinate_system,
            )
            image, depth, alpha = render_panorama_outputs(
                splats,
                width=int(view["width"]),
                height=int(view["height"]),
                device=device,
                world_to_camera=world_to_camera,
                include_depth=True,
            )
            assert depth is not None and alpha is not None
            return image, depth, alpha

        def depth_values(
            view: dict[str, Any], rendered_depth: torch.Tensor, alpha: torch.Tensor, scale: float
        ) -> tuple[float, float | None, np.ndarray]:
            ground_truth = view["ground_truth_depth"]
            if not uses_depth or ground_truth is None:
                return 0.0, None, np.empty((0,), dtype=np.float32)
            predicted = rendered_depth.detach().cpu().numpy() / abs(scale)
            rendered_alpha = alpha.detach().cpu().numpy()
            valid, error, _ = _depth_diagnostics(
                ground_truth,
                predicted,
                rendered_alpha,
                min_gt_depth,
                max_gt_depth,
                min_alpha,
            )
            valid_fraction = float(valid.mean())
            if not valid.any() or valid_fraction < min_valid_fraction:
                return valid_fraction, None, np.empty((0,), dtype=np.float32)
            errors = error[valid]
            return valid_fraction, float(np.median(errors)), errors

        depth_matches: list[float] = []
        if mode in {"depth", "hybrid"}:
            input_depth_view = next(
                (
                    view
                    for view in views
                    if view["role"] == "data"
                    and view["ground_truth_depth"] is not None
                ),
                None,
            )
            logger.info(
                event_message(
                    "depth_scale_matching_started",
                    job_id=job["job_id"],
                    metadata_scene_scale=metadata_scale,
                    starting_scene_scale=starting_scene_scale,
                    depth_view_count=1 if input_depth_view is not None else 0,
                    depth_match_source="data",
                    input_depth_available=input_depth_view is not None,
                )
            )
            if input_depth_view is not None:
                ground_truth = input_depth_view["ground_truth_depth"]
                _, rendered_depth, alpha = render_view(
                    input_depth_view, starting_scene_scale
                )
                raw_depth = rendered_depth.detach().cpu().numpy()
                rendered_alpha = alpha.detach().cpu().numpy()
                valid = (
                    np.isfinite(ground_truth) & (ground_truth >= min_gt_depth)
                    & (ground_truth <= max_gt_depth) & np.isfinite(raw_depth)
                    & (raw_depth > 0) & (rendered_alpha >= min_alpha)
                )
                if valid.any() and float(valid.mean()) >= min_valid_fraction:
                    matched_magnitude = float(
                        np.median(raw_depth[valid] / ground_truth[valid])
                    )
                    depth_matches.append(
                        math.copysign(matched_magnitude, starting_scene_scale)
                    )
            if not depth_matches:
                logger.info(
                    event_message(
                        "depth_scale_matching_fallback",
                        job_id=job["job_id"],
                        initial_scale=starting_scene_scale,
                        reason=(
                            "missing_input_depth"
                            if input_depth_view is None
                            else "insufficient_valid_input_depth"
                        ),
                    )
                )
        matched_scale = _median(depth_matches) or starting_scene_scale
        search_center = initial_scale(
            starting_scene_scale,
            matched_scale,
            mode,
            depth_weight,
        )
        lower, upper = sorted(
            (search_center / range_factor, search_center * range_factor)
        )
        logger.info(
            event_message(
                "scale_search_initialized",
                job_id=job["job_id"],
                depth_matched_scale=matched_scale,
                initial_scale=search_center,
                range_min=lower,
                range_max=upper,
                depth_match_count=len(depth_matches),
            )
        )
        scale_rows: list[dict[str, Any]] = []
        reference_rows: list[dict[str, Any]] = []
        evaluation_details: dict[str, list[dict[str, Any]]] = {}

        def evaluate_scale(
            scale: float,
            evaluation_id: str,
            saved_types: set[str],
        ) -> dict[str, Any]:
            detail_rows: list[dict[str, Any]] = []
            image_losses: list[tuple[float, float]] = []
            depth_scores: list[tuple[float, float]] = []
            metric_scores: dict[str, list[tuple[float, float]]] = {
                name: [] for name in selected_metrics
            }
            for view in views:
                distance_weight = 0.5 ** (
                    float(view["distance"]) / distance_weight_half_life
                )
                image, rendered_depth, alpha = render_view(view, scale)
                image_metrics: list[dict[str, Any]] = []
                if uses_image and view["reference_image"] is not None:
                    image_metrics = _evaluate(
                        view["reference_image"], image.detach().cpu().unsqueeze(0),
                        selected_metrics, evaluator,
                    )
                    for name in selected_metrics:
                        value = _metric_value(image_metrics, name)
                        if value is not None:
                            metric_scores[name].append((value, distance_weight))
                valid_fraction, depth_score, _ = depth_values(
                    view, rendered_depth, alpha, scale
                )
                if depth_score is not None:
                    depth_scores.append((depth_score, distance_weight))
                image_objective = _metric_value(image_metrics, image_metric)
                if image_objective is not None:
                    image_losses.append(
                        (_image_loss(image_metric, image_objective), distance_weight)
                    )
                used = (
                    (mode == "image" and image_objective is not None)
                    or (mode == "depth" and depth_score is not None)
                    or (mode == "hybrid" and (image_objective is not None or depth_score is not None))
                )
                reasons: list[str] = []
                if mode in {"image", "hybrid"} and image_objective is None:
                    reasons.append("missing_image_score")
                if mode in {"depth", "hybrid"} and depth_score is None:
                    reasons.append("insufficient_valid_depth")
                row: dict[str, Any] = {
                    "evaluation_id": evaluation_id,
                    "scale": scale,
                    "reference_sample": view["sample"],
                    "camera_distance": view["distance"],
                    "distance_weight": distance_weight,
                    "used": used,
                    "rejection_reason": ";".join(reasons),
                    "valid_depth_fraction": (
                        valid_fraction
                        if uses_depth and view["ground_truth_depth"] is not None
                        else ""
                    ),
                    "depth_log_l1": depth_score if depth_score is not None else "",
                    "rgb_path": "", "depth_path": "", "alpha_path": "",
                    "rgb_error_path": "", "depth_error_path": "",
                    "depth_filter_path": "", "ground_truth_rgb_path": "",
                    "ground_truth_depth_path": "",
                }
                for name in selected_metrics:
                    value = _metric_value(image_metrics, name)
                    row[name] = value if value is not None else ""
                if saved_types:
                    directory = renders_directory / evaluation_id
                    directory.mkdir(parents=True, exist_ok=True)
                    sample_name = _safe_name(str(view["sample"]))
                    if "rgb" in saved_types:
                        rgb_path = directory / f"{sample_name}-rgb.png"
                        _save_image(image, rgb_path)
                        row["rgb_path"] = rgb_path.relative_to(workspace_root).as_posix()
                    if "depth" in saved_types:
                        depth_path = directory / f"{sample_name}-depth.npy"
                        # Store the same dataset-unit depth that was compared with GT.
                        np.save(depth_path, rendered_depth.detach().cpu().numpy() / abs(scale))
                        row["depth_path"] = depth_path.relative_to(workspace_root).as_posix()
                    if "alpha" in saved_types:
                        alpha_path = directory / f"{sample_name}-alpha.png"
                        _save_gray(alpha.detach().cpu().numpy(), alpha_path)
                        row["alpha_path"] = alpha_path.relative_to(workspace_root).as_posix()
                    if "rgb_error" in saved_types and view["reference_image"] is not None:
                        reference_array = view["reference_image"][0].permute(1, 2, 0).numpy()
                        candidate_array = image.detach().cpu().permute(1, 2, 0).numpy()
                        rgb_error_path = directory / f"{sample_name}-rgb_error.png"
                        _save_error_heatmap(
                            np.abs(reference_array - candidate_array).mean(axis=-1),
                            rgb_error_path,
                            "RGB mean absolute error",
                        )
                        row["rgb_error_path"] = rgb_error_path.relative_to(workspace_root).as_posix()
                    wants_depth_diagnostics = bool(
                        {"depth_error", "depth_filter"} & saved_types
                    )
                    if wants_depth_diagnostics and view["ground_truth_depth"] is not None:
                        ground_truth = view["ground_truth_depth"]
                        predicted = rendered_depth.detach().cpu().numpy() / abs(scale)
                        rendered_alpha = alpha.detach().cpu().numpy()
                        valid, depth_error, depth_reasons = _depth_diagnostics(
                            ground_truth,
                            predicted,
                            rendered_alpha,
                            min_gt_depth,
                            max_gt_depth,
                            min_alpha,
                        )
                        if "depth_error" in saved_types:
                            depth_error_path = directory / f"{sample_name}-depth_error.png"
                            _save_error_heatmap(
                                depth_error,
                                depth_error_path,
                                "Absolute log-depth error",
                                valid,
                            )
                            row["depth_error_path"] = depth_error_path.relative_to(workspace_root).as_posix()
                        if "depth_filter" in saved_types:
                            depth_filter_path = directory / f"{sample_name}-depth_filter.png"
                            _save_depth_filter(
                                ground_truth,
                                depth_reasons,
                                depth_filter_path,
                                min_gt_depth,
                                max_gt_depth,
                            )
                            row["depth_filter_path"] = depth_filter_path.relative_to(workspace_root).as_posix()
                detail_rows.append(row)
                del image, rendered_depth, alpha
            if mode == "image":
                if not image_losses:
                    raise ValueError(f"scale {scale:.8g} has no valid image comparisons")
                objectives = image_losses
                objective = _weighted_quantile(objectives, 0.5)
                objective_p10 = _weighted_quantile(objectives, 0.1)
                objective_p90 = _weighted_quantile(objectives, 0.9)
            elif mode == "depth":
                if not depth_scores:
                    raise ValueError(f"scale {scale:.8g} has no valid depth comparisons")
                objectives = depth_scores
                objective = _weighted_quantile(objectives, 0.5)
                objective_p10 = _weighted_quantile(objectives, 0.1)
                objective_p90 = _weighted_quantile(objectives, 0.9)
            else:
                if not image_losses or not depth_scores:
                    raise ValueError(f"scale {scale:.8g} requires valid image and depth comparisons")
                objective = float(
                    (1.0 - depth_weight) * _weighted_quantile(image_losses, 0.5)
                    + depth_weight * _weighted_quantile(depth_scores, 0.5)
                )
                objective_p10 = float(
                    (1.0 - depth_weight) * _weighted_quantile(image_losses, 0.1)
                    + depth_weight * _weighted_quantile(depth_scores, 0.1)
                )
                objective_p90 = float(
                    (1.0 - depth_weight) * _weighted_quantile(image_losses, 0.9)
                    + depth_weight * _weighted_quantile(depth_scores, 0.9)
                )
            return {
                "details": detail_rows,
                "objective": objective,
                "objective_p10": objective_p10,
                "objective_p90": objective_p90,
                "depth": _weighted_quantile(depth_scores, 0.5),
                "metrics": {
                    name: _weighted_quantile(values, 0.5)
                    for name, values in metric_scores.items()
                },
                "valid_count": sum(1 for row in detail_rows if row["used"]),
            }

        converged = False
        achieved_accuracy = relative_accuracy(lower, upper, search_center)
        for round_index in range(1, max_rounds + 1):
            points = logarithmic_points(lower, upper, points_per_round)
            round_results: list[dict[str, Any]] = []
            for point_index, scale in enumerate(points, start=1):
                evaluation_id = f"r{round_index:02d}-p{point_index:02d}"
                evaluation_started_at = time.time()
                evaluated = evaluate_scale(
                    scale,
                    evaluation_id,
                    active_render_types if saved_scales == "all" else set(),
                )
                evaluation_details[evaluation_id] = evaluated["details"]
                reference_rows.extend(evaluated["details"])
                row = {
                    "sample": primary_sample, "mode": mode, "initial_scale": search_center,
                    "scale_range_factor": range_factor, "target_accuracy": target_accuracy,
                    "view_distance_weight_half_life": distance_weight_half_life,
                    "round": round_index, "point": point_index,
                    "evaluation_id": evaluation_id, "range_min": lower, "range_max": upper,
                    "scale": scale, "is_best": False, "is_edge": False,
                    "objective_score": evaluated["objective"],
                    "objective_p10": evaluated["objective_p10"],
                    "objective_p90": evaluated["objective_p90"],
                    "depth_log_l1_median": evaluated["depth"] if evaluated["depth"] is not None else "",
                    "reference_count": len(views), "valid_reference_count": evaluated["valid_count"],
                    "converged": False, "achieved_accuracy": "",
                }
                for name in selected_metrics:
                    value = evaluated["metrics"][name]
                    row[f"{name}_median"] = value if value is not None else ""
                round_results.append(row)
                scale_rows.append(row)
                logger.info(
                    event_message(
                        "scale_evaluation_completed",
                        job_id=job["job_id"],
                        round=round_index,
                        point=point_index,
                        points_in_round=len(points),
                        evaluation=len(scale_rows),
                        max_evaluations=points_per_round * max_rounds,
                        scale=scale,
                        objective_score=evaluated["objective"],
                        valid_reference_count=evaluated["valid_count"],
                        elapsed_seconds=round(
                            time.time() - evaluation_started_at, 3
                        ),
                    )
                )
            best_index = min(
                range(len(round_results)), key=lambda index: round_results[index]["objective_score"]
            )
            next_lower, next_upper, at_edge = next_range(points, best_index, range_factor)
            achieved_accuracy = relative_accuracy(
                next_lower, next_upper, float(round_results[best_index]["scale"])
            )
            round_results[best_index]["is_best"] = True
            round_results[best_index]["is_edge"] = at_edge
            round_results[best_index]["achieved_accuracy"] = achieved_accuracy
            logger.info(
                event_message(
                    "scale_search_round_completed",
                    job_id=job["job_id"],
                    round=round_index,
                    best_scale=round_results[best_index]["scale"],
                    best_objective_score=round_results[best_index]["objective_score"],
                    best_on_range_edge=at_edge,
                    achieved_accuracy=achieved_accuracy,
                    next_range_min=next_lower,
                    next_range_max=next_upper,
                )
            )
            if not at_edge and achieved_accuracy <= target_accuracy:
                converged = True
                round_results[best_index]["converged"] = True
                break
            lower, upper = next_lower, next_upper

        best_row = min(scale_rows, key=lambda row: float(row["objective_score"]))
        best_scale = float(best_row["scale"])
        best_evaluation_id = str(best_row["evaluation_id"])
        if saved_scales == "best":
            saved = evaluate_scale(best_scale, "best", active_render_types)["details"]
            saved_by_sample = {str(row["reference_sample"]): row for row in saved}
            for row in evaluation_details[best_evaluation_id]:
                saved_row = saved_by_sample[str(row["reference_sample"])]
                for field in (
                    "rgb_path", "depth_path", "alpha_path", "rgb_error_path",
                    "depth_error_path", "depth_filter_path",
                ):
                    if saved_row[field]:
                        row[field] = saved_row[field]
        elif saved_scales == "all":
            original_prefix = f"{renders_name}/{best_evaluation_id}/"
            best_prefix = f"{renders_name}/best/"
            best_source = renders_directory / best_evaluation_id
            if best_source.is_dir():
                best_source.replace(renders_directory / "best")
                for row in evaluation_details[best_evaluation_id]:
                    for field in (
                        "rgb_path", "depth_path", "alpha_path", "rgb_error_path",
                        "depth_error_path", "depth_filter_path",
                    ):
                        if row[field].startswith(original_prefix):
                            row[field] = best_prefix + row[field][len(original_prefix):]

        if copy_ground_truth and active_render_types:
            best_directory = renders_directory / "best"
            best_directory.mkdir(parents=True, exist_ok=True)
            for view in views:
                sample_name = _safe_name(str(view["sample"]))
                rgb_relative = depth_relative = ""
                if "rgb" in active_render_types and view["image_path"]:
                    suffix = Path(view["image_path"]).suffix or ".png"
                    path = best_directory / f"{sample_name}-rgb-GT{suffix}"
                    shutil.copy2(view["image_path"], path)
                    rgb_relative = path.relative_to(workspace_root).as_posix()
                if "depth" in active_render_types and view["depth_path"]:
                    suffix = Path(view["depth_path"]).suffix or ".npy"
                    path = best_directory / f"{sample_name}-depth-GT{suffix}"
                    shutil.copy2(view["depth_path"], path)
                    depth_relative = path.relative_to(workspace_root).as_posix()
                for row in evaluation_details[best_evaluation_id]:
                    if str(row["reference_sample"]) == str(view["sample"]):
                        row["ground_truth_rgb_path"] = rgb_relative
                        row["ground_truth_depth_path"] = depth_relative

        scale_csv = workspace_root / f"scale-evaluations-{output_variant}.csv"
        reference_csv = (
            workspace_root / f"scale-reference-evaluations-{output_variant}.csv"
        )
        _write_csv(
            scale_csv,
            SCALE_FIELDS_PREFIX
            + [f"{name}_median" for name in selected_metrics]
            + SCALE_FIELDS_SUFFIX,
            scale_rows,
        )
        _write_csv(
            reference_csv,
            REFERENCE_FIELDS_PREFIX + selected_metrics + REFERENCE_FIELDS_SUFFIX,
            reference_rows,
        )
        del splats
        torch.cuda.empty_cache()
        resource_metrics = monitor.stop()
        monitor = None
        summary_metrics = [
            _calibration_metric("best_scene_scale", best_scale, "float", "ratio"),
            _calibration_metric("achieved_accuracy", achieved_accuracy, "float", "ratio"),
            _calibration_metric("converged", converged, "boolean"),
            _calibration_metric("objective_score", float(best_row["objective_score"]), "float", "score"),
            _calibration_metric("search_rounds", int(scale_rows[-1]["round"]), "integer", "rounds"),
            _calibration_metric("scale_evaluations", len(scale_rows), "integer", "evaluations"),
            _calibration_metric("valid_reference_count", int(best_row["valid_reference_count"]), "integer", "views"),
        ]
        metrics = summary_metrics + resource_metrics
        report = {
            "mode": mode,
            "metadata_scene_scale": metadata_scale,
            "starting_scene_scale": starting_scene_scale,
            "depth_matched_scale": matched_scale,
            "initial_scale": search_center,
            "best_scene_scale": best_scale,
            "best_evaluation_id": best_evaluation_id,
            "converged": converged,
            "achieved_accuracy": achieved_accuracy,
            "parameters": parameters,
            "skipped_references": skipped_references,
            "metrics": summary_metrics,
            "resource_metrics": resource_metrics,
        }
        _write_metrics_file(metrics_path, report)
        chart_path = workspace_root / f"scale-curves-{output_variant}.png"
        has_chart = _save_scale_chart(scale_rows, chart_path)
        completed_at = time.time()
        final_log_name = f"runner-{output_variant}.log"
        if final_log_name != log_name:
            (workspace_root / log_name).replace(workspace_root / final_log_name)
            log_name = final_log_name
        artifacts = [
            {"artifact_type": "job_log", "path": log_name},
            {"artifact_type": "metric_summary", "path": metrics_name},
        ]
        if has_chart:
            artifacts.append({"artifact_type": "scale_curve", "path": chart_path.name})
        if renders_directory.is_dir():
            artifacts.append(
                {"artifact_type": "rendered_outputs", "path": renders_name}
            )
        logger.info(
            event_message(
                "scale_calibration_completed", job_id=job["job_id"],
                best_scene_scale=best_scale, converged=converged,
                objective_score=float(best_row["objective_score"]),
                search_rounds=int(scale_rows[-1]["round"]),
                scale_evaluations=len(scale_rows),
                elapsed_seconds=round(completed_at - started_at, 3),
            )
        )
        return {
            "status": "completed",
            "started_at": _timestamp(started_at),
            "completed_at": _timestamp(completed_at),
            "output_files": {
                primary_sample: {
                    "scale_calibration": scale_csv.name,
                    "scale_reference_evaluations": reference_csv.name,
                }
            },
            "metrics": metrics,
            "artifacts": artifacts,
            "failure": None,
        }
    except Exception as exc:
        resource_metrics = monitor.stop() if monitor is not None else []
        completed_at = time.time()
        print(f"3DGS scale calibration failed: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        _write_metrics_file(
            metrics_path,
            {"inputs": inputs, "parameters": parameters, "status": "failed", "error": str(exc), "resource_metrics": resource_metrics},
        )
        if output_variant is not None:
            final_log_name = f"runner-{output_variant}.log"
            if final_log_name != log_name:
                (workspace_root / log_name).replace(workspace_root / final_log_name)
                log_name = final_log_name
        return {
            "status": "failed",
            "started_at": _timestamp(started_at),
            "completed_at": _timestamp(completed_at),
            "metrics": resource_metrics,
            "artifacts": [
                {"artifact_type": "job_log", "path": log_name},
                {"artifact_type": "metric_summary", "path": metrics_name},
            ],
            "failure": {
                "code": "3DGS_SCALE_CALIBRATION_FAILED",
                "message": str(exc),
                "retryable": False,
                "stage": "adapter",
            },
        }
