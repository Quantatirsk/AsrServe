# -*- coding: utf-8 -*-
"""
WebSocket 客户端基类
"""

import json
import logging
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from websockets.legacy.client import WebSocketClientProtocol, connect  # type: ignore

logger = logging.getLogger(__name__)


class BaseWebSocketClient(ABC):
    """WebSocket 测试客户端基类"""

    def __init__(self, ws_url: str, timeout: float = 120.0):
        """
        初始化客户端

        Args:
            ws_url: WebSocket URL
            timeout: 超时时间 (秒)
        """
        self.ws_url = ws_url
        self.timeout = timeout
        self.websocket: Optional[WebSocketClientProtocol] = None
        self.task_id = self._generate_id()

    @staticmethod
    def _generate_id() -> str:
        """生成 32 位唯一 ID"""
        return str(uuid.uuid4()).replace("-", "")[:32]

    async def connect(self) -> None:
        """建立 WebSocket 连接"""
        self.websocket = await connect(
            self.ws_url,
            ping_interval=None,
            ping_timeout=None,
            max_size=10 * 1024 * 1024,  # 10MB
        )

    async def close(self) -> None:
        """关闭 WebSocket 连接"""
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception:
                pass
            self.websocket = None

    async def send_json(self, data: Dict[str, Any]) -> None:
        """发送 JSON 消息"""
        if self.websocket:
            await self.websocket.send(json.dumps(data, ensure_ascii=False))

    async def send_bytes(self, data: bytes) -> None:
        """发送二进制数据"""
        if self.websocket:
            await self.websocket.send(data)

    async def receive(self) -> Any:
        """接收消息 (JSON 或二进制)"""
        if self.websocket:
            return await self.websocket.recv()
        return None

    async def receive_json(self) -> Optional[Dict[str, Any]]:
        """接收 JSON 消息"""
        data = await self.receive()
        if isinstance(data, str):
            return json.loads(data)
        return None

    @abstractmethod
    async def run_test(self) -> Any:
        """
        执行测试

        Returns:
            测试指标
        """
        pass
