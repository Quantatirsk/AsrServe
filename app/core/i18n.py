"""AsrServe 国际化 (i18n) 支持
Internationalization (i18n) support for AsrServe.

使用 ASR_LOCALE 环境变量在启动时选择语言
Uses ASR_LOCALE environment variable to select language at startup.
支持的语言: en (英文, 默认), zh (中文)
Supported locales: en (English, default), zh (Chinese).
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 默认语言 / Default locale
DEFAULT_LOCALE = "zh"

# 支持的语言 / Supported locales
SUPPORTED_LOCALES = ["en", "zh"]

# 语言文件路径 / Path to locale files
LOCALE_DIR = Path(__file__).parent.parent / "i18n"

# 当前语言 / Current locale
_current_locale: str = DEFAULT_LOCALE

# 翻译缓存 / Translation cache
_translations: dict[str, Any] = {}


def get_locale() -> str:
    """获取当前语言
    Get the current locale.
    """
    return _current_locale


def set_locale(locale: str) -> None:
    """设置当前语言并加载翻译
    Set the current locale and load translations.

    Args:
        locale: 语言代码 (例如 'en', 'zh') / Locale code (e.g., 'en', 'zh')
    """
    global _current_locale, _translations

    if locale not in SUPPORTED_LOCALES:
        logger.warning(t("i18n.unsupported_locale", locale=locale, default=DEFAULT_LOCALE))
        locale = DEFAULT_LOCALE

    _current_locale = locale
    _translations = _load_translations(locale)
    logger.info(t("i18n.locale_set", locale=locale))


def _load_translations(locale: str) -> dict[str, Any]:
    """加载指定语言的翻译
    Load translations for the given locale.

    Args:
        locale: 语言代码 / Locale code

    Returns:
        翻译字典 / Dictionary of translations
    """
    locale_file = LOCALE_DIR / f"{locale}.json"

    if not locale_file.exists():
        logger.warning(t("i18n.locale_file_not_found", file=str(locale_file)))
        return {}

    try:
        with open(locale_file, "r", encoding="utf-8") as f:
            translations = json.load(f)
        logger.info(t("i18n.translations_loaded", count=len(translations), locale=locale))
        return translations
    except Exception as e:
        logger.error(t("i18n.load_failed", locale=locale, error=e))
        return {}


def translate(key: str, **kwargs: Any) -> str:
    """将字符串键翻译为当前语言
    Translate a string key to the current locale.

    Args:
        key: 翻译键 (点分隔的翻译字典路径) / Translation key (dot-separated path into translations dict)
        **kwargs: 翻译字符串的格式参数 / Format arguments for the translation string

    Returns:
        翻译后的字符串，如果找不到翻译则返回键本身
        Translated string, or the key itself if translation not found
    """
    # Navigate the translations dict using dot-separated key
    parts = key.split(".")
    current = _translations

    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            # 找不到翻译，返回键本身
            # Translation not found, return the key
            return key

    if isinstance(current, str):
        if kwargs:
            try:
                return current.format(**kwargs)
            except (KeyError, IndexError, ValueError) as e:
                logger.warning("i18n.format_failed: key=%s error=%s", key, e)
                return current
        return current

    # 不是字符串，返回键本身
    # Not a string, return the key
    return key


def t(key: str, **kwargs: Any) -> str:
    """translate() 的简写
    Shorthand for translate().
    """
    return translate(key, **kwargs)


def init_i18n() -> None:
    """从环境变量初始化 i18n
    Initialize i18n from environment variables.

    读取 ASR_LOCALE 环境变量来确定语言
    Reads ASR_LOCALE environment variable to determine locale.
    如果未设置或无效则回退到 'zh'
    Falls back to 'zh' if not set or invalid.
    """
    locale = os.getenv("ASR_LOCALE", DEFAULT_LOCALE)
    set_locale(locale)


# 模块导入时初始化 i18n
# Initialize i18n on module import
init_i18n()
