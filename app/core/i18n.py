"""Internationalization (i18n) support for AsrServe.

Uses ASR_LOCALE environment variable to select language at startup.
Supported locales: en (English, default), zh (Chinese).
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default locale
DEFAULT_LOCALE = "en"

# Supported locales
SUPPORTED_LOCALES = ["en", "zh"]

# Path to locale files
LOCALE_DIR = Path(__file__).parent.parent / "i18n"

# Current locale
_current_locale: str = DEFAULT_LOCALE

# Translation cache
_translations: dict[str, Any] = {}


def get_locale() -> str:
    """Get the current locale."""
    return _current_locale


def set_locale(locale: str) -> None:
    """Set the current locale and load translations.

    Args:
        locale: Locale code (e.g., 'en', 'zh')
    """
    global _current_locale, _translations

    if locale not in SUPPORTED_LOCALES:
        logger.warning(f"Unsupported locale '{locale}', falling back to '{DEFAULT_LOCALE}'")
        locale = DEFAULT_LOCALE

    _current_locale = locale
    _translations = _load_translations(locale)
    logger.info(f"Locale set to: {locale}")


def _load_translations(locale: str) -> dict[str, Any]:
    """Load translations for the given locale.

    Args:
        locale: Locale code

    Returns:
        Dictionary of translations
    """
    locale_file = LOCALE_DIR / f"{locale}.json"

    if not locale_file.exists():
        logger.warning(f"Locale file not found: {locale_file}")
        return {}

    try:
        with open(locale_file, "r", encoding="utf-8") as f:
            translations = json.load(f)
        logger.info(f"Loaded {len(translations)} translations for locale '{locale}'")
        return translations
    except Exception as e:
        logger.error(f"Failed to load translations for locale '{locale}': {e}")
        return {}


def translate(key: str, **kwargs: Any) -> str:
    """Translate a string key to the current locale.

    Args:
        key: Translation key (dot-separated path into translations dict)
        **kwargs: Format arguments for the translation string

    Returns:
        Translated string, or the key itself if translation not found
    """
    # Navigate the translations dict using dot-separated key
    parts = key.split(".")
    current = _translations

    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            # Translation not found, return the key
            return key

    if isinstance(current, str):
        if kwargs:
            try:
                return current.format(**kwargs)
            except (KeyError, IndexError, ValueError) as e:
                logger.warning(f"Failed to format translation for key '{key}': {e}")
                return current
        return current

    # Not a string, return the key
    return key


def t(key: str, **kwargs: Any) -> str:
    """Shorthand for translate()."""
    return translate(key, **kwargs)


def init_i18n() -> None:
    """Initialize i18n from environment variables.

    Reads ASR_LOCALE environment variable to determine locale.
    Falls back to 'en' if not set or invalid.
    """
    locale = os.getenv("ASR_LOCALE", DEFAULT_LOCALE)
    set_locale(locale)


# Initialize i18n on module import
init_i18n()
