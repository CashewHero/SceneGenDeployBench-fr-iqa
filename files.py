from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path

logger = logging.getLogger("runner_wrapper.files")


def publish_file(source: str | Path, destination: str | Path) -> Path:
    """Atomically copy a file, preserving metadata when the filesystem allows it."""

    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.is_file():
        raise FileNotFoundError(f"source file not found: {source_path}")

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination_path.with_name(
        f".{destination_path.name}.{uuid.uuid4().hex}.part"
    )
    try:
        shutil.copyfile(source_path, temp_path)
        try:
            shutil.copystat(source_path, temp_path)
        except OSError as exc:
            logger.warning(
                "Could not preserve file metadata for %s: %s; publishing copied contents",
                destination_path,
                exc,
            )
        os.replace(temp_path, destination_path)
    finally:
        temp_path.unlink(missing_ok=True)

    return destination_path
