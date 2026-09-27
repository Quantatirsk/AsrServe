"""Model preparation must select the same dependencies as deployment."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class PlatformScriptsTest(unittest.TestCase):
    def test_prepare_models_selects_platform_extra(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            for name, body in {
                'uname': 'echo "$TEST_PLATFORM"',
                'uv': 'printf "%s\\n" "$@"',
            }.items():
                script = Path(directory) / name
                script.write_text('#!/bin/sh\n' + body + '\n')
                script.chmod(0o755)
            for platform, device, extra in [
                ('Linux', 'cpu', 'cpu'),
                ('Linux', 'cuda:0', 'cuda'),
                ('Darwin', 'cpu', None),
            ]:
                with self.subTest(platform=platform, device=device):
                    env = dict(os.environ, PATH=f'{directory}:{os.environ["PATH"]}',
                               TEST_PLATFORM=platform, DEVICE=device)
                    result = subprocess.check_output(
                        ['bash', str(root / 'scripts/prepare-models.sh'),
                         '--export-dir', '/tmp/model export'], env=env, text=True,
                    ).splitlines()
                    self.assertEqual(result, ['run', '--frozen'] +
                                     (['--extra', extra] if extra else []) +
                                     ['python', '-m', 'app.utils.download_models',
                                      '--export-dir', '/tmp/model export'])
