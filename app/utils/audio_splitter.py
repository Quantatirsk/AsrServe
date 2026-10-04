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
from itertools import pairwise
from typing import List, Tuple, Optional
from dataclasses import dataclass

from ..core.config import settings
from ..core.exceptions import DefaultServerErrorException
from ..core.i18n import t

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

    # 默认配置
    DEFAULT_MIN_SEGMENT_SEC = 1.0  # 每段最小时长（秒）
    DEFAULT_SAMPLE_RATE = 16000  # 默认采样率

    def __init__(
        self,
        min_segment_sec: float = DEFAULT_MIN_SEGMENT_SEC,
    ):
        """初始化音频分割器

        Args:
            min_segment_sec: 每段最小时长（秒）
        """
        split_trigger_sec = settings.MAX_SEGMENT_SEC

        self.split_trigger_sec = split_trigger_sec
        self.min_segment_sec = min_segment_sec
        self.split_trigger_ms = int(split_trigger_sec * 1000)
        self.min_segment_ms = int(min_segment_sec * 1000)

    def merge_segments_greedy(
        self, speech_segments: List[Tuple[int, int]], total_duration_ms: int
    ) -> List[Tuple[int, int]]:
        """按语音活动区间边界切分整条时间轴

        策略：
        1. 活动区间边界只作为切点；区间之间的间隙同样作为独立片段送识别，
           因为未检测到说话人活动不等于静音（低音量语音常落在间隙里）
        2. 不把分离的语音区间打包进同一请求，段落拼接在识别之后进行
        3. 仅对短片段（< min_segment_ms）做邻段合并，超长片段按最大时长切分

        Args:
            speech_segments: 已检测到的语音段列表 [(start_ms, end_ms), ...]
            total_duration_ms: 音频总时长（毫秒）

        Returns:
            覆盖 [0, total_duration_ms] 的连续段列表 [(start_ms, end_ms), ...]
        """
        edges = {
            min(total_duration_ms, max(0, int(edge)))
            for span in speech_segments
            for edge in span
        }
        merged = list(pairwise(sorted(edges | {0, total_duration_ms})))

        # 只处理短片段：与相邻片段合并（不基于静音间隙）
        idx = 0
        while idx < len(merged):
            start_ms, end_ms = merged[idx]
            duration = end_ms - start_ms

            if duration >= self.min_segment_ms or len(merged) == 1:
                idx += 1
                continue

            if idx == 0:
                # 首段过短：并入后段
                next_end = merged[idx + 1][1]
                merged[idx + 1] = (start_ms, next_end)
                del merged[idx]
                continue

            if idx == len(merged) - 1:
                # 尾段过短：并入前段
                prev_start, _ = merged[idx - 1]
                merged[idx - 1] = (prev_start, end_ms)
                del merged[idx]
                idx = max(0, idx - 1)
                continue

            # 中间短段：优先并入时长更短的一侧，避免单段过长
            prev_start = merged[idx - 1][0]
            next_end = merged[idx + 1][1]
            merged_with_prev_duration = end_ms - prev_start
            merged_with_next_duration = next_end - start_ms

            if merged_with_prev_duration <= merged_with_next_duration:
                merged[idx - 1] = (prev_start, end_ms)
                del merged[idx]
                idx = max(0, idx - 1)
            else:
                merged[idx + 1] = (start_ms, next_end)
                del merged[idx]

        return [
            (start_ms + start, start_ms + end)
            for start_ms, end_ms in merged
            for start, end in self._split_by_fixed_duration(end_ms - start_ms)
        ]

    def _split_by_fixed_duration(self, total_duration_ms: int) -> List[Tuple[int, int]]:
        """按固定时长切分超长片段

        Args:
            total_duration_ms: 音频总时长（毫秒）

        Returns:
            切分后的段列表
        """
        if self.split_trigger_ms < 1:
            raise ValueError(
                "Maximum segment duration must be at least one millisecond"
            )
        segments = []
        current = 0
        min_tail_ms = min(self.min_segment_ms, self.split_trigger_ms)
        while current < total_duration_ms:
            end = min(current + self.split_trigger_ms, total_duration_ms)
            remaining = total_duration_ms - end
            if 0 < remaining < min_tail_ms:
                end = total_duration_ms - min_tail_ms
            segments.append((current, end))
            current = end
        return segments

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
            speech_segments: 已知语音区间 [(start_ms, end_ms), ...]，仅作为切点；
                输出片段始终覆盖完整音频

        Returns:
            音频片段列表
        """
        try:
            # 加载音频
            audio_data, sr = librosa.load(audio_path, sr=self.DEFAULT_SAMPLE_RATE)
            total_duration_ms = (len(audio_data) * 1000 + int(sr) - 1) // int(sr)

            logger.info(t("splitter.total_duration", duration=total_duration_ms / 1000))

            # 检查是否需要分割
            if len(audio_data) * 1000 <= self.split_trigger_ms * int(sr):
                logger.info(t("splitter.no_split_needed"))
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
            logger.info(t("splitter.splitting"))
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

            logger.info(t("splitter.split_complete", count=len(audio_segments)))
            return audio_segments

        except Exception as e:
            logger.error(t("splitter.split_failed", error=e))
            raise DefaultServerErrorException(t("splitter.split_failed", error=str(e)))
