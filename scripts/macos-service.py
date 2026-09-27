"""Install or remove the native ASR login service for the current macOS user."""

import argparse
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys

LABEL = "com.asrserve.native"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "uninstall"))
    parser.add_argument(
        "--alignment-mode", choices=("uniform", "forced"), default="uniform"
    )
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if sys.platform != "darwin" or os.getuid() == 0:
        parser.error("Run as your normal macOS login user, without sudo")
    if args.threads < 1:
        parser.error("threads must be positive")
    source = Path(__file__).resolve().parents[1]
    root = Path.home() / "Library/Application Support/AsrServe"
    agent = Path.home() / "Library/LaunchAgents" / f"{LABEL}.plist"
    domain = f"gui/{os.getuid()}"
    target = f"{domain}/{LABEL}"
    if args.action == "install":
        if source == root:
            parser.error("Install from the source Git checkout, not the runtime copy")
        library = Path("vendor/qwenasr/target/release/libqwen_asr.dylib")
        if not (source / library).is_file():
            parser.error("Run scripts/build-rust.sh before installing")
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg or not shutil.which("uv"):
            parser.error("FFmpeg and uv are required")
        root.mkdir(parents=True, exist_ok=True)
        logs = root / "logs"
        logs.mkdir(exist_ok=True)
        config = {
            "Label": LABEL,
            "ProgramArguments": ["/bin/bash", str(root / "scripts/start-native.sh")],
            "WorkingDirectory": str(root),
            "EnvironmentVariables": {
                "PATH": f"{Path(ffmpeg).parent}:/usr/bin:/bin:/usr/sbin:/sbin",
                "HF_HOME": str(root / "models/huggingface"),
                "ALIGNMENT_MODE": args.alignment_mode,
                "R2T2_CPU_THREADS": str(args.threads),
                "R2T2_MAX_SESSIONS": "1",
            },
            "RunAtLoad": True,
            "KeepAlive": True,
            "ThrottleInterval": 30,
            "ExitTimeOut": 60,
            "StandardOutPath": str(logs / "launchd.stdout.log"),
            "StandardErrorPath": str(logs / "launchd.stderr.log"),
        }
    loaded = (
        subprocess.run(["launchctl", "print", target], capture_output=True).returncode
        == 0
    )
    if loaded:
        subprocess.run(["launchctl", "bootout", target], check=True)
    if args.action == "uninstall":
        agent.unlink(missing_ok=True)
        print("Native ASR login service removed; models and logs retained")
        return
    names = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=source)
        .decode()
        .split("\0")
    )
    for name in filter(None, names):
        original = source / name
        if original.is_file():
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, destination)
    (root / library).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / library, root / library)
    if (source / ".env").is_file():
        shutil.copy2(source / ".env", root / ".env")
        (root / ".env").chmod(0o600)
    if (source / "models").is_dir():
        (root / "models").mkdir(exist_ok=True)
        subprocess.run(
            ["cp", "-cR", str(source / "models") + "/.", str(root / "models")],
            check=True,
        )
    install_env = dict(os.environ)
    install_env.pop("VIRTUAL_ENV", None)
    subprocess.run(["uv", "sync", "--frozen"], cwd=root, env=install_env, check=True)
    agent.parent.mkdir(parents=True, exist_ok=True)
    agent.write_bytes(plistlib.dumps(config))
    subprocess.run(["plutil", "-lint", str(agent)], check=True)
    subprocess.run(["launchctl", "bootstrap", domain, str(agent)], check=True)
    print(f"Installed {agent}; starts now and after login")
    print(f"Logs: {logs}; health: http://127.0.0.1:8000/health")


if __name__ == "__main__":
    main()
