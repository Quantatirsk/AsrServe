import unittest
from unittest.mock import patch

from app.services.asr.model_capabilities import ModelAsset, get_model_assets
from app.utils.download_models import download_models


class ModelDownloadTest(unittest.TestCase):
    def test_punctuation_is_required_and_downloaded_to_its_local_directory(
        self,
    ) -> None:
        asset = get_model_assets()[-1]
        self.assertEqual(asset.hub, "modelscope")
        self.assertEqual(asset.revision, "v2.0.4")
        self.assertEqual(len(asset.file_hashes), 3)
        with (
            patch(
                "app.utils.download_models.check_all_models",
                side_effect=[[(asset.model_id, asset.description, asset.revision)], []],
            ),
            patch(
                "app.utils.download_models.is_huggingface_offline", return_value=False
            ),
            patch("app.utils.download_models.get_model_assets", return_value=[asset]),
            patch("app.utils.download_models.hf_snapshot_download") as hf,
            patch("modelscope.hub.snapshot_download.snapshot_download") as download,
        ):
            self.assertTrue(download_models())
        hf.assert_not_called()
        download.assert_called_once_with(
            asset.model_id,
            revision=asset.revision,
            local_dir=asset.local_dir,
            allow_patterns=["config.yaml", "model.pt", "tokens.json"],
        )

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
                "app.utils.download_models.get_model_assets",
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
