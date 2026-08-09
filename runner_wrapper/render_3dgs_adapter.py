from __future__ import annotations

"""Render 3DGS panoramas at primary and reference camera poses."""

import logging
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

from runner_wrapper.fr_iqa_adapter import (
    create_image_quality_evaluator,
    _evaluate,
    _load_rgb_image,
    _normalize_inputs,
    _selected_metrics,
    _timestamp,
    _variant_key,
    _write_metrics_file,
    event_message,
)
from runner_wrapper.gsplat_renderer import load_graphdeco_ply, render_panorama
from runner_wrapper.job_logging import tee_job_output
from runner_wrapper.measurements import ResourceMonitor
from runner_wrapper.scale_search import scene_scale

logger = logging.getLogger("runner_wrapper.render_3dgs_adapter")


def _required_file(mapping: Any, data_type: str, field_name: str) -> Path:
    if not isinstance(mapping, dict):
        raise ValueError(f"{field_name.rsplit('.', 1)[0]} must be an object")
    raw_path = mapping.get(data_type)
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"{field_name} must be a non-empty path")
    path = Path(raw_path.strip())
    if not path.is_file():
        raise FileNotFoundError(f"{field_name} not found: {path}")
    return path


def _output_images(parameters: dict[str, Any]) -> bool:
    value = parameters.get("output_images", True)
    if not isinstance(value, bool):
        raise ValueError("job.parameters.output_images must be a boolean")
    return value


def _max_distance(
    parameters: dict[str, Any],
    default: float = 50.0,
) -> float:
    value = parameters.get("max_distance", default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("job.parameters.max_distance must be a non-negative number")
    distance = float(value)
    if not math.isfinite(distance) or distance < 0:
        raise ValueError("job.parameters.max_distance must be a non-negative number")
    return distance


def _max_references(
    parameters: dict[str, Any],
    default: int = 500,
) -> int:
    value = parameters.get("max_references", default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("job.parameters.max_references must be a non-negative integer")
    return value


def _numeric_vector(value: Any, *, length: int, field_name: str) -> torch.Tensor:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{field_name} must contain {length} numbers")
    numbers = [float(item) for item in value]
    if not all(math.isfinite(number) for number in numbers):
        raise ValueError(f"{field_name} must contain only finite numbers")
    return torch.tensor(numbers, dtype=torch.float64)


def _quaternion_rotation(quaternion: torch.Tensor, field_name: str) -> torch.Tensor:
    norm = torch.linalg.vector_norm(quaternion)
    if float(norm) == 0:
        raise ValueError(f"{field_name} must not be zero length")
    x, y, z, w = quaternion / norm
    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=torch.float64,
    )


def _camera_to_world(
    camera_pose: Any,
    *,
    field_name: str,
    convention: str,
) -> torch.Tensor:
    if not isinstance(camera_pose, dict):
        raise ValueError(f"{field_name} must be an object")
    position = _numeric_vector(
        camera_pose.get("position", [0.0, 0.0, 0.0]),
        length=3,
        field_name=f"{field_name}.position",
    )
    quaternion = _numeric_vector(
        camera_pose.get("rotation_quaternion_xyzw", [0.0, 0.0, 0.0, 1.0]),
        length=4,
        field_name=f"{field_name}.rotation_quaternion_xyzw",
    )
    transform = torch.eye(4, dtype=torch.float64)
    transform[:3, :3] = _quaternion_rotation(
        quaternion,
        f"{field_name}.rotation_quaternion_xyzw",
    )
    transform[:3, 3] = position
    if convention == "camera_to_world":
        return transform
    if convention == "world_to_camera":
        return torch.linalg.inv(transform)
    raise ValueError(f"unsupported pose convention: {convention}")


def _normalized_world_to_camera(
    primary_pose: Any,
    target_pose: Any,
    *,
    convention: str,
    primary_field: str,
    target_field: str,
    scene_scale: float = 1.0,
) -> tuple[torch.Tensor, float]:
    primary_to_world = _camera_to_world(
        primary_pose,
        field_name=primary_field,
        convention=convention,
    )
    target_to_world = _camera_to_world(
        target_pose,
        field_name=target_field,
        convention=convention,
    )
    target_to_primary = torch.linalg.inv(primary_to_world) @ target_to_world
    distance = float(torch.linalg.vector_norm(target_to_primary[:3, 3]))
    target_to_primary[:3, 3] *= scene_scale
    return torch.linalg.inv(target_to_primary).to(dtype=torch.float32), distance


def _sample_image_name(sample_id: str, variant: str) -> str:
    normalized = sample_id.strip().replace("\\", "-").replace("/", "-")
    if not normalized or normalized in {".", ".."}:
        raise ValueError(f"invalid sample id for image output: {sample_id!r}")
    return f"{normalized}-{variant}.png"


def _save_image(tensor: torch.Tensor, path: Path) -> None:
    array = (
        tensor.detach().cpu().permute(1, 2, 0).mul(255).round().byte().numpy()
    )
    Image.fromarray(array, mode="RGB").save(path)


def _save_distance_chart(
    views: list[dict[str, Any]],
    metric_names: list[str],
    path: Path,
) -> bool:
    if len(views) < 2:
        return False

    panel_width = 1000
    panel_height = 320
    top_margin = 30
    chart = Image.new(
        "RGB",
        (panel_width, top_margin + panel_height * len(metric_names)),
        "white",
    )
    draw = ImageDraw.Draw(chart)
    colors = ["#2563eb", "#dc2626", "#059669", "#9333ea"]
    distance_unit = str(views[0]["distance_unit"])
    tick_intervals = 5

    def centered_text(x: float, y: float, text: str) -> None:
        bounds = draw.textbbox((0, 0), text)
        draw.text((x - (bounds[2] - bounds[0]) / 2, y), text, fill="#374151")

    def right_aligned_text(x: float, y: float, text: str) -> None:
        bounds = draw.textbbox((0, 0), text)
        draw.text((x - (bounds[2] - bounds[0]), y), text, fill="#374151")

    for metric_index, metric_name in enumerate(metric_names):
        points = [
            (float(view["distance"]), float(view["metrics"][metric_name]))
            for view in views
            if isinstance(view["metrics"].get(metric_name), (int, float))
            and math.isfinite(float(view["metrics"][metric_name]))
        ]
        if not points:
            continue
        points.sort()
        panel_top = top_margin + metric_index * panel_height
        left, right = 80, panel_width - 30
        top, bottom = panel_top + 35, panel_top + panel_height - 50
        max_distance = max(distance for distance, _ in points)
        min_value = min(value for _, value in points)
        max_value = max(value for _, value in points)
        if math.isclose(min_value, max_value):
            padding = max(abs(min_value) * 0.05, 1e-6)
            min_value -= padding
            max_value += padding
        distance_span = max(max_distance, 1e-9)
        value_span = max_value - min_value

        draw.text((left, panel_top + 5), f"{metric_name.upper()} by distance", fill="black")
        for tick_index in range(tick_intervals + 1):
            fraction = tick_index / tick_intervals
            x = left + fraction * (right - left)
            y = bottom - fraction * (bottom - top)
            if tick_index not in {0, tick_intervals}:
                draw.line((x, top, x, bottom), fill="#e5e7eb", width=1)
                draw.line((left, y, right, y), fill="#e5e7eb", width=1)
            draw.line((x, bottom, x, bottom + 4), fill="#374151", width=1)
            draw.line((left - 4, y, left, y), fill="#374151", width=1)
            centered_text(x, bottom + 8, f"{fraction * max_distance:.4g}")
            right_aligned_text(
                left - 8,
                y - 6,
                f"{min_value + fraction * value_span:.4g}",
            )
        draw.line((left, top, left, bottom), fill="#374151", width=2)
        draw.line((left, bottom, right, bottom), fill="#374151", width=2)
        draw.text(
            ((left + right) // 2 - 45, bottom + 29),
            f"distance ({distance_unit})",
            fill="#374151",
        )

        plotted: list[tuple[float, float]] = []
        for distance, value in points:
            x = left + (distance / distance_span) * (right - left)
            y = bottom - ((value - min_value) / value_span) * (bottom - top)
            plotted.append((x, y))
        if len(plotted) > 1:
            draw.line(plotted, fill=colors[metric_index % len(colors)], width=2)
        for x, y in plotted:
            draw.ellipse(
                (x - 3, y - 3, x + 3, y + 3),
                fill=colors[metric_index % len(colors)],
            )

    chart.save(path)
    return True


def _annotate_metrics(
    metrics: list[dict[str, Any]],
    *,
    sample_id: str,
    role: str,
    distance: float,
    distance_unit: str,
) -> list[dict[str, Any]]:
    for metric in metrics:
        metric["metadata"] = {
            "sample": sample_id,
            "role": role,
            "distance": distance,
            "distance_unit": distance_unit,
        }
    return metrics


def run_job(job_request: dict[str, Any]) -> dict[str, Any]:
    started_at = time.time()
    workspace_root = Path(job_request["runtime"]["workspace_dir"])
    workspace_root.mkdir(parents=True, exist_ok=True)
    variant = _variant_key(job_request, "render")
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
    inputs: dict[str, dict[str, dict[str, Any]]] = {}
    parameters: dict[str, Any] = {}

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("3DGS rendering requires an NVIDIA CUDA GPU")

        job = job_request["job"]
        inputs = _normalize_inputs(job_request.get("inputs"))
        primary_sample = str(job["primary_sample"])
        data_samples = inputs.get("data", {})
        candidate_samples = inputs.get("candidate", {})
        reference_samples = inputs.get("references", {})
        if primary_sample not in data_samples:
            raise ValueError(f"primary sample not found in inputs.data: {primary_sample}")
        if primary_sample not in candidate_samples:
            raise ValueError(
                f"primary sample not found in inputs.candidate: {primary_sample}"
            )

        primary_data = data_samples[primary_sample]
        primary_image = _required_file(
            primary_data,
            "image",
            f"inputs.data.{primary_sample}.image",
        )
        splat_path = _required_file(
            candidate_samples[primary_sample],
            "3dgs",
            f"inputs.candidate.{primary_sample}.3dgs",
        )
        raw_parameters = job.get("parameters") or {}
        if not isinstance(raw_parameters, dict):
            raise ValueError("job.parameters must be an object")
        parameters = dict(raw_parameters)
        selected_metrics = _selected_metrics(parameters)
        keep_images = _output_images(parameters)
        max_distance = _max_distance(parameters)
        max_references = _max_references(parameters)
        model_scene_scale = scene_scale(job.get("primary_output_metadata"))

        metadata = job.get("primary_sample_metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("job.primary_sample_metadata must be an object")
        pose_convention = str(
            metadata.get("pose_convention") or "camera_to_world"
        ).strip()
        distance_unit = str(metadata.get("pose_units") or "unspecified").strip()
        primary_pose = primary_data.get("camera_pose") or {}

        views: list[dict[str, Any]] = [
            {
                "sample": primary_sample,
                "role": "data",
                "image": primary_image,
                "world_to_camera": torch.eye(4, dtype=torch.float32),
                "distance": 0.0,
            }
        ]
        skipped_views: list[dict[str, Any]] = []
        eligible_references: list[dict[str, Any]] = []
        for sample_id, sample_data in reference_samples.items():
            reference_pose = sample_data.get("camera_pose")
            if reference_pose is None:
                raise ValueError(
                    f"inputs.references.{sample_id}.camera_pose is required"
                )
            world_to_camera, distance = _normalized_world_to_camera(
                primary_pose,
                reference_pose,
                convention=pose_convention,
                primary_field=f"inputs.data.{primary_sample}.camera_pose",
                target_field=f"inputs.references.{sample_id}.camera_pose",
                scene_scale=model_scene_scale,
            )
            if distance > max_distance:
                skipped_views.append(
                    {
                        "sample": sample_id,
                        "role": "references",
                        "distance": distance,
                        "distance_unit": distance_unit,
                        "reason": "distance_exceeds_maximum",
                    }
                )
                continue
            eligible_references.append(
                {
                    "sample": sample_id,
                    "role": "references",
                    "sample_data": sample_data,
                    "world_to_camera": world_to_camera,
                    "distance": distance,
                }
            )
        eligible_references.sort(key=lambda view: (view["distance"], view["sample"]))
        selected_references = eligible_references[:max_references]
        for skipped in eligible_references[max_references:]:
            skipped_views.append(
                {
                    "sample": skipped["sample"],
                    "role": "references",
                    "distance": skipped["distance"],
                    "distance_unit": distance_unit,
                    "reason": "reference_limit_exceeded",
                }
            )
        for selected in selected_references:
            sample_id = str(selected["sample"])
            selected["image"] = _required_file(
                selected.pop("sample_data"),
                "image",
                f"inputs.references.{sample_id}.image",
            )
            views.append(selected)

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
                "render_evaluation_started",
                job_id=job["job_id"],
                primary_sample=primary_sample,
                scene_3dgs=str(splat_path),
                reference_count=len(reference_samples),
                selected_reference_count=len(views) - 1,
                skipped_reference_count=len(skipped_views),
                skipped_reasons={
                    reason: sum(
                        1 for view in skipped_views if view["reason"] == reason
                    )
                    for reason in sorted({view["reason"] for view in skipped_views})
                },
                max_distance=max_distance,
                max_references=max_references,
                scene_scale=model_scene_scale,
                metrics=selected_metrics,
            )
        )

        device = torch.device("cuda")
        load_started_at = time.time()
        splats = load_graphdeco_ply(splat_path, device)
        evaluator = create_image_quality_evaluator(
            selected_metrics,
            device,
            workspace_root,
        )
        logger.info(
            event_message(
                "render_resources_loaded",
                job_id=job["job_id"],
                gaussian_count=int(splats["means"].shape[0]),
                device=str(device),
                elapsed_seconds=round(time.time() - load_started_at, 3),
            )
        )
        quality_metrics: list[dict[str, Any]] = []
        view_results: list[dict[str, Any]] = []
        output_files: dict[str, dict[str, str]] = {}

        for view_index, view in enumerate(views, start=1):
            view_started_at = time.time()
            sample_id = str(view["sample"])
            reference, reference_info = _load_rgb_image(Path(view["image"]))
            image = render_panorama(
                splats,
                width=int(reference_info["width"]),
                height=int(reference_info["height"]),
                device=device,
                world_to_camera=view["world_to_camera"],
            )
            candidate = image.detach().cpu().unsqueeze(0)
            view_metrics = _annotate_metrics(
                _evaluate(reference, candidate, selected_metrics, evaluator),
                sample_id=sample_id,
                role=str(view["role"]),
                distance=float(view["distance"]),
                distance_unit=distance_unit,
            )
            quality_metrics.extend(view_metrics)
            view_results.append(
                {
                    "sample": sample_id,
                    "role": view["role"],
                    "distance": view["distance"],
                    "distance_unit": distance_unit,
                    "metrics": {
                        metric["name"]: metric["value"] for metric in view_metrics
                    },
                }
            )
            if keep_images:
                image_path = workspace_root / _sample_image_name(sample_id, variant)
                _save_image(image, image_path)
                output_files[sample_id] = {"image": image_path.name}
            logger.info(
                event_message(
                    "view_evaluation_completed",
                    job_id=job["job_id"],
                    sample=sample_id,
                    role=view["role"],
                    view=view_index,
                    view_count=len(views),
                    distance=view["distance"],
                    distance_unit=distance_unit,
                    metrics={
                        metric["name"]: metric["value"] for metric in view_metrics
                    },
                    elapsed_seconds=round(time.time() - view_started_at, 3),
                )
            )
            del reference, image, candidate

        del splats
        torch.cuda.empty_cache()
        resource_metrics = monitor.stop()
        monitor = None
        metrics = quality_metrics + resource_metrics
        completed_at = time.time()

        report: dict[str, Any] = {
            "inputs": inputs,
            "views": view_results,
        }
        if skipped_views:
            report["skipped_views"] = skipped_views
        if output_files:
            report["output_files"] = output_files
        if parameters:
            report["parameters"] = parameters
        if quality_metrics:
            report["metrics"] = quality_metrics
        if resource_metrics:
            report["resource_metrics"] = resource_metrics
        _write_metrics_file(metrics_path, report)
        chart_path = workspace_root / f"distance-{variant}.png"
        has_chart = _save_distance_chart(
            view_results,
            selected_metrics,
            chart_path,
        )
        logger.info(
            event_message(
                "render_evaluation_completed",
                job_id=job["job_id"],
                view_count=len(views),
                skipped_reference_count=len(skipped_views),
                output_images=keep_images,
                elapsed_seconds=round(completed_at - started_at, 3),
            )
        )
        artifacts = [
            {"artifact_type": "job_log", "path": log_name},
            {"artifact_type": "metric_summary", "path": metrics_name},
        ]
        if has_chart:
            artifacts.append(
                {
                    "artifact_type": "evaluation_chart",
                    "path": chart_path.name,
                }
            )
        result: dict[str, Any] = {
            "status": "completed",
            "started_at": _timestamp(started_at),
            "completed_at": _timestamp(completed_at),
            "metrics": metrics,
            "artifacts": artifacts,
            "failure": None,
        }
        if output_files:
            result["output_files"] = output_files
        return result
    except Exception as exc:
        resource_metrics = monitor.stop() if monitor is not None else []
        completed_at = time.time()
        print(f"3DGS render evaluation failed: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        _write_metrics_file(
            metrics_path,
            {
                "inputs": inputs,
                **({"parameters": parameters} if parameters else {}),
                "status": "failed",
                "error": str(exc),
                "resource_metrics": resource_metrics,
            },
        )
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
                "code": "3DGS_RENDER_IQA_FAILED",
                "message": str(exc),
                "retryable": False,
                "stage": "adapter",
            },
        }
