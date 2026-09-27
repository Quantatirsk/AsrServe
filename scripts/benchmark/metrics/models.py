# -*- coding: utf-8 -*-
"""
性能指标数据类
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ASRMetrics:
    """ASR 单次请求指标"""

    request_id: str
    concurrency_level: int
    start_time: float  # time.perf_counter()
    audio_duration_ms: float = 0.0

    # 时间戳
    first_result_time: Optional[float] = None  # 第一个 delta 事件
    complete_time: Optional[float] = None  # done 事件

    # 结果
    result_text: str = ""
    success: bool = False
    error_message: str = ""

    @property
    def first_result_latency_ms(self) -> Optional[float]:
        """首次响应延迟 (ms)"""
        if self.first_result_time is not None:
            return (self.first_result_time - self.start_time) * 1000
        return None

    @property
    def total_processing_time_ms(self) -> Optional[float]:
        """总处理时间 (ms)"""
        if self.complete_time is not None:
            return (self.complete_time - self.start_time) * 1000
        return None

    @property
    def rtf(self) -> Optional[float]:
        """RTF (Real-Time Factor) = 处理时间 / 音频时长"""
        total_time = self.total_processing_time_ms
        if total_time is not None and self.audio_duration_ms > 0:
            return total_time / self.audio_duration_ms
        return None


@dataclass
class AggregatedMetrics:
    """聚合后的指标 (针对一个并发级别)"""

    concurrency_level: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_test_time_seconds: float

    # 首次延迟统计 (ms)
    first_latency_avg: float = 0.0
    first_latency_p50: float = 0.0
    first_latency_p95: float = 0.0
    first_latency_p99: float = 0.0
    first_latency_max: float = 0.0

    # 总时间统计 (ms)
    total_time_avg: float = 0.0
    total_time_p50: float = 0.0
    total_time_p95: float = 0.0
    total_time_p99: float = 0.0
    total_time_max: float = 0.0

    # RTF 统计
    rtf_avg: float = 0.0
    rtf_p50: float = 0.0
    rtf_p95: float = 0.0
    rtf_p99: float = 0.0
    rtf_max: float = 0.0

    @property
    def success_rate(self) -> float:
        """成功率 (%)"""
        if self.total_requests > 0:
            return (self.successful_requests / self.total_requests) * 100
        return 0.0

    @property
    def throughput(self) -> float:
        """吞吐量 (成功请求数/秒)"""
        if self.total_test_time_seconds > 0:
            return self.successful_requests / self.total_test_time_seconds
        return 0.0
