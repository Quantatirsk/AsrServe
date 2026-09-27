import unittest
from unittest.mock import patch

from app.core.device import detect_device
from app.core.exceptions import InvalidParameterException
from app.services.asr.manager import get_model_manager
from app.services.asr.model_capabilities import (
    get_huggingface_assets,
)
from app.services.asr.model_plan import get_runtime_model_ids
from app.services.realtime.protocol import MODEL_ID, MODEL_REPOSITORY, MODEL_REVISION


class OfflineContractTest(unittest.TestCase):
    def test_one_checkpoint_for_both_modes_with_separate_aligner(self) -> None:
        self.assertEqual(get_runtime_model_ids(), [MODEL_ID])
        assets = get_huggingface_assets()
        self.assertEqual(
            [asset.model_id for asset in assets],
            [
                "nvidia/Nemotron-3-Diarization",
                MODEL_REPOSITORY,
                "Qwen/Qwen3-ForcedAligner-0.6B",
            ],
        )
        self.assertEqual(assets[1].revision, MODEL_REVISION)
        self.assertEqual(assets[0].revision, "f667ed73aee57d40cc39428eb768b4fd87a0a29e")
        with self.assertRaises(InvalidParameterException):
            get_model_manager().get_declared_entry_config("unsupported-model")

    def test_explicit_cpu_and_cuda_without_fallback(self) -> None:
        with patch("app.core.device.torch.cuda.is_available", return_value=False):
            self.assertEqual(detect_device("cpu"), "cpu")
            with self.assertRaises(RuntimeError):
                detect_device("cuda:0")
        with patch("app.core.device.torch.cuda.is_available", return_value=True):
            for device in ("mps", "npu", "cuda", "cuda:1", "cpu:0"):
                with self.subTest(device=device), self.assertRaises(ValueError):
                    detect_device(device)
            self.assertEqual(detect_device("cuda:0"), "cuda:0")


if __name__ == "__main__":
    unittest.main()
