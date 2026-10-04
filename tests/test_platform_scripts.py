"""Model preparation must run inside the selected service image."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class PlatformScriptsTest(unittest.TestCase):
    def test_prepare_models_runs_service_image_without_devices(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            docker = Path(directory) / "docker"
            docker.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            docker.chmod(0o755)
            env = dict(os.environ, PATH=f"{directory}:{os.environ['PATH']}")
            result = subprocess.check_output(
                ["bash", str(root / "scripts/prepare-models.sh")],
                env=env,
                text=True,
            ).splitlines()
        self.assertEqual(
            result,
            [
                "run",
                "--rm",
                "-e",
                "HF_ENDPOINT",
                "-v",
                f"{root}/models:/app/models",
                "quantatrisk/asrserve:gpu",
                "--download-models",
            ],
        )
