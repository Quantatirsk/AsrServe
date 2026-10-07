"""
统一配置管理
ASR语音识别配置选项
"""

import math
import os
import sys
from pathlib import Path

OFFLINE_MAX_SECONDS = 60


class Settings:
    """统一应用配置类"""

    # 应用信息
    APP_NAME: str = "AsrServe"
    APP_VERSION: str = "1.0.5"
    APP_DESCRIPTION: str = "R2T2 offline and realtime speech recognition"

    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # 鉴权配置
    API_KEY: str | None = None  # 从环境变量API_KEY读取，如果为None则鉴权可选

    # 设备配置
    DEVICE: str = "cpu" if sys.platform == "darwin" else "cuda:0"
    R2T2_CPU_THREADS: int = 8
    ALIGNMENT_MODE: str = "forced"

    # 路径配置
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    TEMP_DIR: str = "temp"
    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str | None = str(BASE_DIR / "logs" / "asrserve.log")
    LOG_MAX_BYTES: int = 20 * 1024 * 1024  # 20MB
    LOG_BACKUP_COUNT: int = 50  # 保留50个备份文件

    NEMOTRON_MODEL_PATH: str = str(BASE_DIR / "models/nemotron-3-diarization")
    PUNCTUATION_MODEL_PATH: str = str(
        BASE_DIR / "models/modelscope/hub/models/iic"
        / "punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
    )
    R2T2_URL: str = ""
    R2T2_INTERNAL_TOKEN: str = ""
    # 音频处理配置
    MAX_AUDIO_SIZE: int = 2048 * 1024 * 1024  # 2GB

    # 音频分段配置
    MAX_SEGMENT_SEC: float = float(OFFLINE_MAX_SECONDS)

    def __init__(self) -> None:
        """从环境变量读取配置"""
        self._load_from_env()
        self._ensure_directories()

    def _load_from_env(self) -> None:
        """从环境变量加载配置"""
        # 服务器配置
        self.HOST = os.getenv("HOST", self.HOST)
        self.PORT = int(os.getenv("PORT", str(self.PORT)))
        self.DEBUG = os.getenv("DEBUG", "false").lower() == "true"

        # 日志配置
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", self.LOG_LEVEL)
        self.LOG_FILE = os.getenv("LOG_FILE", self.LOG_FILE)
        self.LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", str(self.LOG_MAX_BYTES)))
        self.LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", str(self.LOG_BACKUP_COUNT)))

        # 鉴权配置：空值/空白统一视为未配置
        self.API_KEY = (os.getenv("API_KEY") or "").strip() or None

        # 设备配置
        self.DEVICE = os.getenv("DEVICE", self.DEVICE)
        self.ALIGNMENT_MODE = os.getenv("ALIGNMENT_MODE", self.ALIGNMENT_MODE)
        if self.ALIGNMENT_MODE not in {"uniform", "forced"}:
            raise ValueError("ALIGNMENT_MODE must be uniform or forced")
        self.R2T2_CPU_THREADS = int(os.getenv("R2T2_CPU_THREADS", str(self.R2T2_CPU_THREADS)))
        if self.R2T2_CPU_THREADS < 1:
            raise ValueError("R2T2_CPU_THREADS must be greater than zero")

        self.NEMOTRON_MODEL_PATH = os.path.expanduser(
            os.getenv("NEMOTRON_MODEL_PATH", self.NEMOTRON_MODEL_PATH)
        )
        self.R2T2_URL = os.getenv("R2T2_URL", "").strip().rstrip("/")
        self.R2T2_INTERNAL_TOKEN = os.getenv("R2T2_INTERNAL_TOKEN", "").strip()

        # 音频处理配置
        # 支持简化格式：纯数字表示MB，或带单位（如 2048MB, 2GB）
        max_audio_size_str = os.getenv("MAX_AUDIO_SIZE")
        if max_audio_size_str:
            self.MAX_AUDIO_SIZE = self._parse_size(max_audio_size_str)

        self.MAX_SEGMENT_SEC = float(os.getenv("MAX_SEGMENT_SEC", str(self.MAX_SEGMENT_SEC)))
        if (
            not math.isfinite(self.MAX_SEGMENT_SEC)
            or not 0 < self.MAX_SEGMENT_SEC <= OFFLINE_MAX_SECONDS
        ):
            raise ValueError(
                f"MAX_SEGMENT_SEC must be greater than zero and at most {OFFLINE_MAX_SECONDS}"
            )

    def _parse_size(self, size_str: str) -> int:
        """解析带单位的大小字符串

        支持格式：
        - 纯数字：视为 MB（如 2048 = 2048MB = 2147483648 bytes）
        - 带单位：如 2GB, 2048MB, 1.5GB
        """
        size_str = size_str.strip().upper()

        # 如果纯数字，视为 MB
        if size_str.isdigit():
            return int(size_str) * 1024 * 1024

        # 带单位的处理
        if size_str.endswith("GB"):
            return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
        elif size_str.endswith("MB"):
            return int(float(size_str[:-2]) * 1024 * 1024)
        elif size_str.endswith("KB"):
            return int(float(size_str[:-2]) * 1024)
        else:
            # 默认视为字节
            return int(size_str)

    def _ensure_directories(self) -> None:
        """确保必需的目录存在"""
        os.makedirs(self.TEMP_DIR, exist_ok=True)

    @property
    def docs_url(self) -> str | None:
        """获取文档URL"""
        return "/docs"

    @property
    def redoc_url(self) -> str | None:
        """获取ReDoc URL"""
        return "/redoc"


# 全局配置实例
settings = Settings()
