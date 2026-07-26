from __future__ import annotations

"""Full-reference image-quality evaluator adapter."""

import hashlib
import json
import logging
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)
from torchmetrics.functional.image.dists import DISTSNetwork
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from runner_wrapper.job_logging import tee_job_output
from runner_wrapper.measurements import ResourceMonitor

logger = logging.getLogger("runner_wrapper.fr_iqa_adapter")

SUPPORTED_METRICS = ("psnr", "ssim", "lpips", "ws_psnr", "dists")
METRIC_ALIASES = {"ws-psnr": "ws_psnr"}


def event_message(event: str, **fields: object) -> str:
    return json.dumps({"event": event, **fields}, sort_keys=True)


def _timestamp(value: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(value))


def _variant_key(
    job_request: dict[str, Any],
    label: str,
) -> str:
    payload = {
        "inputs": job_request.get("inputs") or {},
        "parameters": job_request.get("job", {}).get("parameters") or {},
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()[:10]
    return f"{label}-{digest}"


def _normalize_inputs(raw_inputs: Any) -> dict[str, dict[str, dict[str, Any]]]:
    if raw_inputs is None:
        return {}
    if not isinstance(raw_inputs, dict):
        raise ValueError("inputs must be an object")

    normalized: dict[str, dict[str, dict[str, Any]]] = {}
    for raw_role, raw_samples in raw_inputs.items():
        role = str(raw_role).strip()
        if not role or not isinstance(raw_samples, dict):
            raise ValueError("each input role must contain a sample mapping")
        samples: dict[str, dict[str, Any]] = {}
        for raw_sample_id, raw_sample_data in raw_samples.items():
            sample_id = str(raw_sample_id).strip()
            if not sample_id or not isinstance(raw_sample_data, dict):
                raise ValueError(f"inputs.{role} must map sample ids to data mappings")
            sample_data: dict[str, Any] = {}
            for raw_data_type, value in raw_sample_data.items():
                data_type = str(raw_data_type).strip()
                if not data_type:
                    raise ValueError(f"inputs.{role}.{sample_id} contains an empty data type")
                sample_data[data_type] = value.strip() if isinstance(value, str) else value
            if sample_data:
                samples[sample_id] = sample_data
        if samples:
            normalized[role] = samples
    return normalized


def _required_path(mapping: Any, field_name: str) -> Path:
    if not isinstance(mapping, dict):
        raise ValueError(f"{field_name.rsplit('.', 1)[0]} must be an object")
    raw_path = mapping.get("image")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"{field_name} must be a non-empty path")
    path = Path(raw_path.strip())
    if not path.is_file():
        raise FileNotFoundError(f"{field_name} not found: {path}")
    return path


def _selected_metrics(parameters: Any) -> list[str]:
    if not isinstance(parameters, dict):
        raise ValueError("job.parameters must be an object")
    raw_metrics = parameters.get("metrics", list(SUPPORTED_METRICS))
    if isinstance(raw_metrics, str):
        values = raw_metrics.split(",")
    elif isinstance(raw_metrics, list):
        values = raw_metrics
    else:
        raise ValueError("job.parameters.metrics must be a list or comma-separated string")

    selected: list[str] = []
    for raw_value in values:
        metric = str(raw_value).strip().lower()
        metric = METRIC_ALIASES.get(metric, metric)
        if not metric or metric in selected:
            continue
        if metric not in SUPPORTED_METRICS:
            raise ValueError(
                f"unsupported metric {metric!r}; choose from {', '.join(SUPPORTED_METRICS)}"
            )
        selected.append(metric)
    if not selected:
        raise ValueError("job.parameters.metrics must select at least one metric")
    return selected


def _load_rgb_image(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        array = np.asarray(image, dtype=np.float32).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0) / 255.0
    return tensor, {
        "path": str(path),
        "format": path.suffix.lstrip(".").lower() or "unknown",
        "width": width,
        "height": height,
        "channels": 3,
        "color_mode": "RGB",
        "size_bytes": path.stat().st_size,
    }


def _quality_metric(name: str, value: float) -> dict[str, Any]:
    if math.isfinite(value):
        metric_type = "float"
        metric_value: float | str = value
    else:
        # JSON and PostgreSQL JSONB cannot safely carry a non-finite number.
        metric_type = "string"
        metric_value = "infinity" if value > 0 else "undefined"
    return {
        "namespace": "image_quality",
        "name": name,
        "type": metric_type,
        "value": metric_value,
        "unit": "dB" if name in {"psnr", "ws_psnr"} else "score",
        "source": "evaluator",
    }


def _weighted_spherical_psnr(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> torch.Tensor:
    height = reference.shape[-2]
    row_weights = torch.sin(
        math.pi
        * (
            torch.arange(height, device=reference.device, dtype=reference.dtype)
            + 0.5
        )
        / height
    ).view(1, 1, height, 1)
    weighted_error = ((candidate - reference).square() * row_weights).sum()
    sample_count, channel_count, _, width = reference.shape
    weighted_pixel_count = (
        row_weights.sum() * sample_count * channel_count * width
    )
    return -10.0 * torch.log10(weighted_error / weighted_pixel_count)


class ImageQualityEvaluator:
    """Load learned metrics once and evaluate one or more aligned image pairs."""

    def __init__(self, selected_metrics: list[str], device: torch.device) -> None:
        self.selected_metrics = selected_metrics
        self.device = device
        self.lpips: LearnedPerceptualImagePatchSimilarity | None = None
        self.dists: DISTSNetwork | None = None
        if "lpips" in selected_metrics:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex",
                reduction="mean",
                normalize=True,
            ).to(device)
            self.lpips.eval()
        if "dists" in selected_metrics:
            self.dists = DISTSNetwork().to(device)
            self.dists.eval()

    def evaluate(
        self,
        reference: torch.Tensor,
        candidate: torch.Tensor,
    ) -> list[dict[str, Any]]:
        reference = reference.to(self.device)
        candidate = candidate.to(self.device)
        metrics: list[dict[str, Any]] = []
        with torch.inference_mode():
            if "psnr" in self.selected_metrics:
                value = peak_signal_noise_ratio(candidate, reference, data_range=1.0)
                metrics.append(_quality_metric("psnr", float(value.item())))
            if "ssim" in self.selected_metrics:
                value = structural_similarity_index_measure(
                    candidate,
                    reference,
                    data_range=1.0,
                )
                metrics.append(_quality_metric("ssim", float(value.item())))
            if self.lpips is not None:
                self.lpips.reset()
                self.lpips.update(candidate, reference)
                value = self.lpips.compute()
                metrics.append(_quality_metric("lpips", float(value.item())))
            if "ws_psnr" in self.selected_metrics:
                value = _weighted_spherical_psnr(reference, candidate)
                metrics.append(_quality_metric("ws_psnr", float(value.item())))
            if self.dists is not None:
                value = self.dists(candidate, reference).mean()
                metrics.append(_quality_metric("dists", float(value.item())))
        return metrics


def _evaluate(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    selected_metrics: list[str],
    evaluator: ImageQualityEvaluator | None = None,
) -> list[dict[str, Any]]:
    if evaluator is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        evaluator = ImageQualityEvaluator(selected_metrics, device)
    return evaluator.evaluate(reference, candidate)


def _write_metrics_file(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def run_job(job_request: dict[str, Any]) -> dict[str, Any]:
    started_at = time.time()
    runtime = job_request["runtime"]
    output_root = Path(runtime["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    variant = _variant_key(job_request, "iqa")
    log_path = output_root / f"runner-{variant}.log"
    with tee_job_output(log_path):
        return _run_job_logged(job_request, started_at, output_root, variant, log_path.name)


def _run_job_logged(
    job_request: dict[str, Any],
    started_at: float,
    output_root: Path,
    variant: str,
    log_name: str,
) -> dict[str, Any]:
    monitor: ResourceMonitor | None = None
    metrics_name = f"metrics-{variant}.json"
    metrics_path = output_root / metrics_name
    inputs: dict[str, dict[str, dict[str, Any]]] = {}
    parameters: dict[str, Any] = {}

    try:
        job = job_request["job"]
        runtime = job_request["runtime"]
        inputs = _normalize_inputs(job_request.get("inputs"))
        data_samples = inputs.get("data", {})
        candidate_samples = inputs.get("candidate", {})
        primary_sample = str(job["primary_sample"])
        if primary_sample not in data_samples:
            raise ValueError(f"primary sample not found in inputs.data: {primary_sample}")
        if primary_sample not in candidate_samples:
            raise ValueError(
                f"primary sample not found in inputs.candidate: {primary_sample}"
            )

        reference_path = _required_path(
            data_samples[primary_sample],
            f"inputs.data.{primary_sample}.image",
        )
        candidate_path = _required_path(
            candidate_samples[primary_sample],
            f"inputs.candidate.{primary_sample}.image",
        )
        raw_parameters = job.get("parameters") or {}
        if not isinstance(raw_parameters, dict):
            raise ValueError("job.parameters must be an object")
        parameters = dict(raw_parameters)
        selected_metrics = _selected_metrics(parameters)

        monitor_data = {
            f"{role}.{sample_id}.{data_type}": value
            for role, samples in inputs.items()
            for sample_id, sample_data in samples.items()
            for data_type, value in sample_data.items()
        }
        monitor = ResourceMonitor(sample_data=monitor_data, output_dir=output_root)
        monitor.start()
        logger.info(
            event_message(
                "adapter_run_started",
                job_id=job["job_id"],
                batch_id=job.get("batch_id"),
                output_dir=runtime["output_dir"],
                input_roles=sorted(inputs),
            )
        )
        logger.info(
            event_message(
                "evaluation_started",
                job_id=job["job_id"],
                reference_image=str(reference_path),
                candidate_image=str(candidate_path),
                metrics=selected_metrics,
            )
        )

        reference, reference_info = _load_rgb_image(reference_path)
        candidate, candidate_info = _load_rgb_image(candidate_path)
        if reference.shape != candidate.shape:
            raise ValueError(
                "aligned image dimensions must match exactly: "
                f"reference={reference_info['width']}x{reference_info['height']}, "
                f"candidate={candidate_info['width']}x{candidate_info['height']}"
            )

        quality_metrics = _evaluate(reference, candidate, selected_metrics)
        resource_metrics = monitor.stop()
        monitor = None
        metrics = quality_metrics + resource_metrics
        completed_at = time.time()
        report: dict[str, Any] = {"inputs": inputs}
        if parameters:
            report["parameters"] = parameters
        if quality_metrics:
            report["metrics"] = quality_metrics
        if resource_metrics:
            report["resource_metrics"] = resource_metrics
        _write_metrics_file(metrics_path, report)
        logger.info(
            event_message(
                "evaluation_completed",
                job_id=job["job_id"],
                metrics={item["name"]: item["value"] for item in quality_metrics},
            )
        )
        return {
            "status": "completed",
            "started_at": _timestamp(started_at),
            "completed_at": _timestamp(completed_at),
            "metrics": metrics,
            "artifacts": [
                {
                    "artifact_type": "job_log",
                    "path": log_name,
                },
                {
                    "artifact_type": "metric_summary",
                    "path": metrics_name,
                },
            ],
            "failure": None,
        }
    except Exception as exc:
        resource_metrics = monitor.stop() if monitor is not None else []
        completed_at = time.time()
        print(f"image evaluation failed: {exc}", file=sys.stderr, flush=True)
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
                {
                    "artifact_type": "job_log",
                    "path": log_name,
                }
            ],
            "failure": {
                "code": "FR_IQA_FAILED",
                "message": str(exc),
                "retryable": False,
                "stage": "adapter",
            },
        }
