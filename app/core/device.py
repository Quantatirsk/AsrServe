"""Select an explicit CPU, CUDA or Ascend NPU device without runtime fallback."""

import torch


def detect_device(configured: str = "cuda:0") -> str:
    if configured == "cpu":
        return "cpu"
    if configured == "npu:0":
        try:
            import torch_npu  # noqa: F401 -- registers torch.npu
        except ImportError as exc:
            raise RuntimeError(
                "DEVICE=npu:0 requires the Ascend Python environment"
            ) from exc
        if not torch.npu.is_available():
            raise RuntimeError("DEVICE=npu:0 requires an available Ascend NPU")
        torch.npu.set_device(0)
        return configured
    if configured != "cuda:0":
        raise ValueError(
            "DEVICE must be cpu, cuda:0 or npu:0; select the accelerator with its visibility environment"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("DEVICE=cuda:0 requires an available NVIDIA CUDA GPU")
    return "cuda:0"
