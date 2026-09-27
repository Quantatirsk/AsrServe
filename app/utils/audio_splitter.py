# -*- coding: utf-8 -*-
"""
音频分割模块
基于 Nemotron 活动区间的音频分割，支持长音频分段识别
"""

import logging
import numpy as np
import librosa
import soundfile as sf
import tempfile
import os
from typing import List, Tuple, Optional
from dataclasses import dataclass

from ..core.config import settings
from ..core.exceptions import DefaultServerErrorException

logger = logging.getLogger(__name__)


@dataclass
class AudioSegment:
    """音频片段信息"""

    start_ms: int  # 开始时间（毫秒）
    end_ms: int  # 结束时间（毫秒）
    audio_data: Optional[np.ndarray] = None  # 音频数据
    temp_file: Optional[str] = None  # 临时文件路径

    @property
    def start_sec(self) -> float:
        """开始时间（秒）"""
        return self.start_ms / 1000.0

    @property
    def end_sec(self) -> float:
        """结束时间（秒）"""
        return self.end_ms / 1000.0

    @property
    def duration_ms(self) -> int:
        """时长（毫秒）"""
        return self.end_ms - self.start_ms

    @property
    def duration_sec(self) -> float:
        """时长（秒）"""
        return self.duration_ms / 1000.0


class AudioSplitter:
    """音频分割器

    使用已有语音活动边界分割长音频，不运行检测模型
    """

    DEFAULT_SAMPLE_RATE = 16000

    def __init__(self):
        self.split_trigger_ms = int(settings.MAX_SEGMENT_SEC * 1000)
        if self.split_trigger_ms < 1:
            raise ValueError(
                "Maximum segment duration must be at least one millisecond"
            )

    def merge_segments_greedy(
        self, speech_segments: List[Tuple[int, int]], total_duration_ms: int
    ) -> List[Tuple[int, int]]:
        """Pack the entire timeline into bounded chunks, preferring known pauses.

        Activity is a cut hint, never permission to discard unrecognized audio.
        These computation boundaries do not define transcript paragraphs.
        """
        spans = []
        for start, end in sorted(speech_segments):
            start, end = max(0, int(start)), min(total_duration_ms, int(end))
            if end <= start:
                continue
            if spans and start <= spans[-1][1]:
                spans[-1] = (spans[-1][0], max(end, spans[-1][1]))
            else:
                spans.append((start, end))
        pauses = [(left[1], right[0]) for left, right in zip(spans, spans[1:])]
        chunks = []
        current = pause_index = 0
        while current < total_duration_ms:
            limit = min(current + self.split_trigger_ms, total_duration_ms)
            cut = limit
            if limit < total_duration_ms:
                candidate = None
                while pause_index < len(pauses) and pauses[pause_index][0] <= limit:
                    start, end = pauses[pause_index]
                    point = limit if end >= limit else (start + end) // 2
                    if point > current:
                        candidate = point
                    if end > limit:
                        break
                    pause_index += 1
                if candidate is not None:
                    cut = candidate
            chunks.append((current, cut))
            current = cut
        return chunks

    def _split_by_fixed_duration(self, total_duration_ms: int) -> List[Tuple[int, int]]:
        return self.merge_segments_greedy([], total_duration_ms)

    def split_audio_file(
        self,
        audio_path: str,
        output_dir: Optional[str] = None,
        *,
        speech_segments: List[Tuple[int, int]],
    ) -> List[AudioSegment]:
        """分割音频文件

        Args:
            audio_path: 音频文件路径
            output_dir: 输出目录（可选，默认使用临时目录）
            speech_segments: 已知语音区间 [(start_ms, end_ms), ...]。给出时直接
                复用；空列表仍按固定时长分割以保留识别兜底

        Returns:
            音频片段列表
        """
        try:
            # 加载音频
            audio_data, sr = librosa.load(audio_path, sr=self.DEFAULT_SAMPLE_RATE)
            total_duration_ms = (len(audio_data) * 1000 + int(sr) - 1) // int(sr)

            logger.info(f"音频总时长: {total_duration_ms / 1000:.2f}秒")

            # 检查是否需要分割
            if len(audio_data) * 1000 <= self.split_trigger_ms * int(sr):
                logger.info("音频时长在限制内，无需分割")
                return [
                    AudioSegment(
                        start_ms=0,
                        end_ms=total_duration_ms,
                        audio_data=audio_data,
                        temp_file=audio_path,
                    )
                ]

            merged_segments = self.merge_segments_greedy(
                speech_segments, total_duration_ms
            )
            logger.info(
                "重分段完成: 来源=Nemotron, 原始语音区间=%d, 输出=%d",
                len(speech_segments),
                len(merged_segments),
            )

            # 切分音频并保存到临时文件
            logger.info("开始切分音频并保存临时文件...")
            output_dir = output_dir or settings.TEMP_DIR
            os.makedirs(output_dir, exist_ok=True)

            audio_segments = []
            for idx, (start_ms, end_ms) in enumerate(merged_segments):
                # 计算采样点范围
                start_sample = start_ms * int(sr) // 1000
                end_sample = end_ms * int(sr) // 1000

                # 提取音频片段
                segment_data = audio_data[start_sample:end_sample]

                # 保存到临时文件
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".wav",
                    dir=output_dir,
                    prefix=f"segment_{idx:03d}_",
                )
                temp_path = temp_file.name
                temp_file.close()

                sf.write(temp_path, segment_data, sr)

                segment = AudioSegment(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    audio_data=segment_data,
                    temp_file=temp_path,
                )
                audio_segments.append(segment)

                logger.debug(
                    f"分段 {idx + 1}/{len(merged_segments)}: "
                    f"{start_ms / 1000:.2f}s - {end_ms / 1000:.2f}s "
                    f"(时长: {segment.duration_sec:.2f}s)"
                )

            logger.info(f"音频切分完成，共 {len(audio_segments)} 个分段")
            return audio_segments

        except Exception as e:
            logger.error(f"音频分割失败: {e}")
            raise DefaultServerErrorException(f"音频分割失败: {str(e)}")
