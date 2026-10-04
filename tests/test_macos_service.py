"""launchd must finish unloading before installation can restart a service."""

import runpy
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class MacOSServiceTest(unittest.TestCase):
    def test_uninstall_waits_until_job_is_removed(self) -> None:
        script = Path(__file__).resolve().parents[1] / "scripts/macos-service.py"
        main = runpy.run_path(str(script))["main"]
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            agent = home / "Library/LaunchAgents/com.asrserve.native.plist"
            agent.parent.mkdir(parents=True)
            agent.touch()
            with (
                patch("pathlib.Path.home", return_value=home),
                patch("sys.platform", "darwin"),
                patch("os.getuid", return_value=501),
                patch("sys.argv", [str(script), "uninstall"]),
                patch("time.sleep") as sleep,
                patch(
                    "subprocess.run",
                    side_effect=[subprocess.CompletedProcess([], code) for code in (0, 0, 0, 1)],
                ) as run,
            ):
                main()
            self.assertFalse(agent.exists())
            sleep.assert_called_once_with(0.2)
            self.assertEqual(
                [call.args[0][1] for call in run.call_args_list],
                ["print", "bootout", "print", "print"],
            )
