# -*- coding: utf-8 -*-
"""API v1版本路由"""

from fastapi import APIRouter, Request

from ...core.config import settings
from ...core.device import detect_device
from ...core.exceptions import AuthenticationException
from ...core.security import validate_token
from ...services.asr.runtime import get_runtime_router
from .realtime import router as realtime_router
from .openai_compatible import router as openai_router

api_router = APIRouter()

# R2T2 native WebSocket.
api_router.include_router(realtime_router)

# OpenAI 兼容 API
api_router.include_router(openai_router)


@api_router.get("/health", summary="服务健康检查")
async def health(request: Request):
    result, content = validate_token(request)
    if not result:
        raise AuthenticationException(content, "health")
    try:
        # Readiness must not wait for a busy offline inference slot.
        runtime_router = get_runtime_router()
        loaded_models = runtime_router.get_loaded_model_ids()
        model_loaded = runtime_router.resolve_model_id(None) in loaded_models
        return {
            "status": "healthy" if model_loaded else "unhealthy",
            "model_loaded": model_loaded,
            "device": detect_device(settings.DEVICE) if model_loaded else "unknown",
            "version": settings.APP_VERSION,
            "message": (
                "ASR service is running normally"
                if model_loaded
                else "ASR model not loaded"
            ),
            "loaded_models": loaded_models,
            "memory_usage": runtime_router.get_memory_usage().get("gpu_memory"),
        }
    except Exception as e:
        return {
            "status": "error",
            "model_loaded": False,
            "device": "unknown",
            "version": settings.APP_VERSION,
            "message": str(e),
        }

