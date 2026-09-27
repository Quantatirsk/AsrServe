# -*- coding: utf-8 -*-
"""
测试配置模块
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class TestConfig:
    """测试配置"""

    # 服务器配置
    host: str = "localhost"
    port: int = 8000
    timeout_seconds: float = 300.0  # 默认 5 分钟，长音频并发可能需要更长时间
    warmup_requests: int = 3

    # 并发配置
    concurrency_levels: List[int] = field(default_factory=lambda: [1, 2, 4])

    # ASR 配置
    asr_audio_file: Optional[Path] = None
    asr_sample_rate: int = 16000
    asr_chunk_size: int = 2560  # 160ms @ 16kHz
    asr_format: str = "int16_le"

    # 输出配置
    output_dir: Path = field(default_factory=lambda: Path("./benchmark_results"))
    report_name: str = "benchmark_report"

    @property
    def ws_base_url(self) -> str:
        """WebSocket 基础 URL"""
        return f"ws://{self.host}:{self.port}"

    @property
    def asr_ws_url(self) -> str:
        """ASR WebSocket URL"""
        return f"{self.ws_base_url}/v1/stream"

    def validate(self) -> None:
        """
        验证配置

        Raises:
            ValueError: 配置无效
        """
        if self.asr_audio_file is None:
            raise ValueError("ASR 测试需要提供音频文件路径 (--audio-file)")
        if not self.asr_audio_file.exists():
            raise ValueError(f"音频文件不存在: {self.asr_audio_file}")

        if not self.concurrency_levels:
            raise ValueError("至少需要一个并发级别")

        for level in self.concurrency_levels:
            if level < 1:
                raise ValueError(f"并发级别必须大于 0: {level}")

        if self.timeout_seconds <= 0:
            raise ValueError("超时时间必须大于 0")
