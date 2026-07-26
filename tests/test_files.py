from __future__ import annotations

import errno
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from runner_wrapper.files import publish_file


class PublishFileTests(unittest.TestCase):
    def test_preserves_metadata_when_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.bin"
            destination = root / "output" / "destination.bin"
            source.write_bytes(b"scene data")
            source.chmod(0o640)
            timestamp_ns = 1_700_000_000_123_456_789
            os.utime(source, ns=(timestamp_ns, timestamp_ns))

            publish_file(source, destination)

            self.assertEqual(destination.read_bytes(), b"scene data")
            self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o640)
            self.assertEqual(destination.stat().st_mtime_ns, timestamp_ns)

    def test_publishes_contents_when_metadata_is_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.bin"
            destination = root / "output" / "destination.bin"
            source.write_bytes(b"scene data")

            error = PermissionError(errno.EPERM, "Operation not permitted")
            with (
                mock.patch("runner_wrapper.files.shutil.copystat", side_effect=error),
                self.assertLogs("runner_wrapper.files", level="WARNING"),
            ):
                publish_file(source, destination)

            self.assertEqual(destination.read_bytes(), b"scene data")
            self.assertFalse(any(destination.parent.glob("*.part")))
