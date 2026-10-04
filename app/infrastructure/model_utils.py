"""Shared model path resolution helpers."""

import os
from pathlib import Path


def _is_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _expand_path(value: str) -> Path:
    return Path(os.path.expandvars(value)).expanduser()


def is_huggingface_offline() -> bool:
    return _is_truthy(os.getenv("HF_HUB_OFFLINE"))


def get_huggingface_cache_root() -> Path:
    hf_hub_cache = (os.getenv("HF_HUB_CACHE") or "").strip()
    if hf_hub_cache:
        return _expand_path(hf_hub_cache)

    hf_home = (os.getenv("HF_HOME") or "").strip()
    if hf_home:
        return _expand_path(hf_home) / "hub"

    xdg_cache_home = (os.getenv("XDG_CACHE_HOME") or "").strip()
    if xdg_cache_home:
        return _expand_path(xdg_cache_home) / "huggingface" / "hub"

    return Path.home() / ".cache" / "huggingface" / "hub"


def get_huggingface_model_cache_dir(model_id: str) -> Path:
    org, model = model_id.split("/", 1)
    return get_huggingface_cache_root() / f"models--{org}--{model}"


def find_huggingface_snapshot_dir(model_ref_or_path: str) -> Path | None:
    raw_path = Path(model_ref_or_path).expanduser()
    if raw_path.exists():
        return raw_path.resolve()
    if "/" not in model_ref_or_path:
        return None

    base_dir = get_huggingface_model_cache_dir(model_ref_or_path)
    if not base_dir.exists():
        return None

    ref_main = base_dir / "refs" / "main"
    if ref_main.exists():
        snapshot_name = ref_main.read_text(encoding="utf-8").strip()
        snapshot_dir = base_dir / "snapshots" / snapshot_name
        if snapshot_dir.exists():
            return snapshot_dir.resolve()

    snapshots_dir = base_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
    snapshots.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return snapshots[0].resolve() if snapshots else None


def resolve_huggingface_snapshot_dir(model_ref_or_path: str) -> Path:
    snapshot_dir = find_huggingface_snapshot_dir(model_ref_or_path)
    if snapshot_dir is not None:
        return snapshot_dir

    raise FileNotFoundError(
        f"Hugging Face model path not found for '{model_ref_or_path}'. "
        f"Checked direct path and cache root: {get_huggingface_cache_root()}."
    )
