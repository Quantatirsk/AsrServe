import unittest
from unittest.mock import patch

from app.services.asr.model_capabilities import ModelAsset
from app.utils.download_models import download_models


class ModelDownloadTest(unittest.TestCase):
    def test_missing_nemotron_downloads_pinned_revision(self) -> None:
        asset = ModelAsset(
            "nvidia/Nemotron-3-Diarization",
            "Nemotron",
            revision="pinned-revision",
            local_dir="/models/nemotron",
        )
        with (
            patch(
                "app.utils.download_models.check_all_models",
                side_effect=[[(asset.model_id, asset.description, asset.revision)], []],
            ),
            patch(
                "app.utils.download_models.is_huggingface_offline",
                return_value=False,
            ),
            patch(
                "app.utils.download_models.get_huggingface_assets",
                return_value=[asset],
            ),
            patch("app.utils.download_models.hf_snapshot_download") as download,
        ):
            self.assertTrue(download_models())
        download.assert_called_once_with(
            asset.model_id, revision="pinned-revision", local_dir="/models/nemotron"
        )

    def test_offline_missing_assets_fail_without_network(self) -> None:
        with (
            patch(
                "app.utils.download_models.check_all_models",
                return_value=[
                    (
                        "nvidia/Nemotron-3-Diarization",
                        "Nemotron",
                        "revision",
                    )
                ],
            ),
            patch(
                "app.utils.download_models.is_huggingface_offline", return_value=True
            ),
            patch("app.utils.download_models.hf_snapshot_download") as download,
        ):
            self.assertFalse(download_models())
        download.assert_not_called()
