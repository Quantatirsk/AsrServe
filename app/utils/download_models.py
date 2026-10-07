"""Download and verify the single supported model set."""

from huggingface_hub import snapshot_download as hf_snapshot_download

from app.infrastructure import is_huggingface_offline
from app.services.asr.model_capabilities import ModelAsset, get_model_assets


def _download_modelscope_asset(asset: ModelAsset) -> None:
    from modelscope.hub.snapshot_download import snapshot_download

    snapshot_download(
        asset.model_id,
        revision=asset.revision,
        local_dir=asset.local_dir,
        allow_patterns=list(asset.required_patterns),
    )


def check_all_models() -> list[tuple[str, str, str | None]]:
    from app.utils.model_loader import (
        _build_required_model_integrity_specs,
        _check_model_integrity_spec,
    )

    assets = get_model_assets()
    specs = _build_required_model_integrity_specs()
    return [
        (asset.model_id, asset.description, asset.revision)
        for asset, spec in zip(assets, specs, strict=True)
        if not _check_model_integrity_spec(spec)["ok"]
    ]


def download_models() -> bool:
    missing = check_all_models()
    if missing and is_huggingface_offline():
        print(
            "Offline mode: required model files are missing:",
            [item[0] for item in missing],
        )
        return False
    assets = get_model_assets()
    missing_ids = {item[0] for item in missing}
    for asset in assets:
        if asset.model_id not in missing_ids:
            continue
        try:
            if asset.hub == "modelscope":
                _download_modelscope_asset(asset)
            else:
                hf_snapshot_download(
                    asset.model_id, revision=asset.revision, local_dir=asset.local_dir
                )
            print("Downloaded", asset.model_id)
        except Exception as error:
            print("Model download failed:", asset.model_id, str(error))
            return False
    if check_all_models():
        print("Model integrity check failed after download")
        return False
    print("All required models are ready")
    return True
