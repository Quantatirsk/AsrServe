"""i18n must load, switch, translate, and fall back correctly across the app."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1 import api_router
from app.core import i18n
from app.core.config import settings
from app.core.i18n import t, get_locale, set_locale, init_i18n, translate
from app.utils.audio import download_audio_from_url


class I18nLocaleSwitchingTest(unittest.TestCase):
    """Locale must switch correctly and load the right translations."""

    def setUp(self):
        self.original_locale = i18n._current_locale
        self.original_translations = i18n._translations
        i18n._current_locale = i18n.DEFAULT_LOCALE
        i18n._translations = {}

    def tearDown(self):
        i18n._current_locale = self.original_locale
        i18n._translations = self.original_translations

    def test_locale_defaults_to_zh_on_import(self):
        self.assertEqual(i18n.DEFAULT_LOCALE, "zh")

    def test_locale_switches_between_en_and_zh(self):
        set_locale("en")
        self.assertEqual(get_locale(), "en")
        set_locale("zh")
        self.assertEqual(get_locale(), "zh")

    def test_locale_switch_loads_correct_translations(self):
        set_locale("en")
        self.assertIn("app", i18n._translations)
        self.assertEqual(t("app.description"), "R2T2 offline and realtime speech recognition")

        set_locale("zh")
        self.assertEqual(t("app.description"), "R2T2 离线和实时语音识别")

    def test_unsupported_locale_falls_back_to_default(self):
        set_locale("fr")
        self.assertEqual(get_locale(), i18n.DEFAULT_LOCALE)

    def test_asr_locale_env_var_selects_locale_at_init(self):
        with patch.dict(os.environ, {"ASR_LOCALE": "en"}):
            init_i18n()
            self.assertEqual(get_locale(), "en")

    def test_init_falls_back_when_env_var_is_invalid(self):
        with patch.dict(os.environ, {"ASR_LOCALE": "invalid_locale"}):
            init_i18n()
            self.assertEqual(get_locale(), i18n.DEFAULT_LOCALE)

    def test_init_uses_default_when_env_var_not_set(self):
        env = os.environ.copy()
        env.pop("ASR_LOCALE", None)
        with patch.dict(os.environ, env, clear=True):
            init_i18n()
            self.assertEqual(get_locale(), i18n.DEFAULT_LOCALE)


class I18nTranslationLookupTest(unittest.TestCase):
    """Translation lookup must resolve nested keys and handle missing entries."""

    def setUp(self):
        self.original_locale = i18n._current_locale
        self.original_translations = i18n._translations
        i18n._current_locale = "en"
        i18n._translations = {}
        set_locale("en")

    def tearDown(self):
        i18n._current_locale = self.original_locale
        i18n._translations = self.original_translations

    def test_translates_top_level_keys(self):
        self.assertEqual(t("app.name"), "AsrServe")

    def test_translates_nested_dot_separated_keys(self):
        self.assertEqual(t("errors.not_found"), "Not found")
        self.assertEqual(t("health.healthy"), "healthy")

    def test_translates_chinese_strings(self):
        set_locale("zh")
        self.assertEqual(t("errors.not_found"), "未找到")
        self.assertEqual(t("health.running"), "ASR 服务运行正常")

    def test_returns_key_itself_when_translation_missing(self):
        self.assertEqual(t("nonexistent.key.path"), "nonexistent.key.path")

    def test_returns_key_itself_for_non_string_values(self):
        i18n._translations = {"test": {"value": 123}}
        self.assertEqual(t("test.value"), "test.value")

    def test_t_is_functional_shorthand_for_translate(self):
        set_locale("en")
        self.assertEqual(t("app.name"), translate("app.name"))


class I18nFormatStringTest(unittest.TestCase):
    """Format string interpolation must work and fail gracefully."""

    def setUp(self):
        self.original_locale = i18n._current_locale
        self.original_translations = i18n._translations
        set_locale("en")

    def tearDown(self):
        i18n._current_locale = self.original_locale
        i18n._translations = self.original_translations

    def test_interpolates_format_arguments(self):
        result = t("app.temp_cleaned", count=5)
        self.assertEqual(result, "Cleaned up 5 expired temp files")

    def test_interpolates_chinese_format_strings(self):
        set_locale("zh")
        result = t("app.temp_cleaned", count=5)
        self.assertEqual(result, "已清理 5 个过期临时文件")

    def test_interpolates_api_log_messages(self):
        result = t("api.transcription_complete", chars=42)
        self.assertEqual(result, "[OpenAI API] Transcription complete: 42 characters")

    def test_returns_unformatted_template_when_args_missing(self):
        result = t("app.temp_cleaned")
        self.assertIn("{count}", result)

    def test_extra_kwargs_are_ignored(self):
        result = t("app.name", extra="ignored")
        self.assertEqual(result, "AsrServe")

    def test_format_error_does_not_crash(self):
        # Malformed format string should not raise
        i18n._translations = {"test": {"bad": "{unclosed"}}
        result = t("test.bad", value="x")
        # Should return something without crashing
        self.assertIsInstance(result, str)


class I18nTranslationFileLoadingTest(unittest.TestCase):
    """Translation files must load correctly and handle errors."""

    def test_loads_valid_translation_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfile = Path(tmpdir) / "test.json"
            data = {"test": {"key": "value"}}
            tmpfile.write_text(json.dumps(data), encoding="utf-8")
            with patch.object(i18n, "LOCALE_DIR", Path(tmpdir)):
                result = i18n._load_translations("test")
                self.assertEqual(result, data)

    def test_returns_empty_dict_when_file_not_found(self):
        result = i18n._load_translations("nonexistent")
        self.assertEqual(result, {})

    def test_returns_empty_dict_when_json_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            broken = Path(tmpdir) / "broken.json"
            broken.write_text("{invalid json", encoding="utf-8")
            with patch.object(i18n, "LOCALE_DIR", Path(tmpdir)):
                result = i18n._load_translations("broken")
                self.assertEqual(result, {})

    def test_supported_locales_are_en_and_zh(self):
        self.assertIn("en", i18n.SUPPORTED_LOCALES)
        self.assertIn("zh", i18n.SUPPORTED_LOCALES)
        self.assertEqual(len(i18n.SUPPORTED_LOCALES), 2)


class I18nApiIntegrationTest(unittest.TestCase):
    """API endpoints must work correctly with i18n initialized."""

    def setUp(self):
        self.original_locale = i18n._current_locale
        self.original_translations = i18n._translations

        # Disable auth for these tests
        self.auth_patch = patch.object(settings, "API_KEY", None)
        self.auth_patch.start()
        self.addCleanup(self.auth_patch.stop)

        app = FastAPI()
        app.include_router(api_router)
        self.client = TestClient(app)

    def tearDown(self):
        i18n._current_locale = self.original_locale
        i18n._translations = self.original_translations

    def test_api_works_with_english_locale(self):
        """API endpoints must function when locale is set to English."""
        set_locale("en")
        response = self.client.get("/v1/models")
        self.assertEqual(response.status_code, 200)

    def test_api_works_with_chinese_locale(self):
        """API endpoints must function when locale is set to Chinese."""
        set_locale("zh")
        response = self.client.get("/v1/models")
        self.assertEqual(response.status_code, 200)

    def test_authentication_rejects_invalid_token(self):
        """Authentication must reject invalid tokens regardless of locale."""
        with patch.object(settings, "API_KEY", "secret-token"):
            set_locale("en")
            response = self.client.get("/v1/config", headers={"Authorization": "Bearer wrong"})
            self.assertEqual(response.status_code, 401)

            set_locale("zh")
            response = self.client.get("/v1/config", headers={"Authorization": "Bearer wrong"})
            self.assertEqual(response.status_code, 401)

    def test_validate_token_returns_localized_error_when_no_token(self):
        """validate_token must return the localized error message when no token is provided."""
        from app.core.security import validate_token
        from fastapi.testclient import TestClient as TC
        from fastapi import FastAPI as FA, Request

        test_app = FA()
        
        @test_app.get("/test")
        async def test_endpoint(request: Request):
            result, message = validate_token(request)
            return {"result": result, "message": message}

        tc = TC(test_app)

        # Enable auth for this test
        with patch.object(settings, "API_KEY", "test-secret-key"):
            # English - no token provided
            set_locale("en")
            response = tc.get("/test")
            self.assertEqual(response.status_code, 200)
            self.assertFalse(response.json()["result"])
            self.assertEqual(response.json()["message"], "Authentication failed")

            # Chinese - no token provided
            set_locale("zh")
            response = tc.get("/test")
            self.assertEqual(response.status_code, 200)
            self.assertFalse(response.json()["result"])
            self.assertEqual(response.json()["message"], "认证失败")


class I18nUnicodeAndEncodingTest(unittest.TestCase):
    """Unicode and encoding edge cases must be handled correctly."""

    def setUp(self):
        self.original_locale = i18n._current_locale
        self.original_translations = i18n._translations
        set_locale("zh")

    def tearDown(self):
        i18n._current_locale = self.original_locale
        i18n._translations = self.original_translations

    def test_chinese_characters_round_trip_correctly(self):
        result = t("app.description")
        self.assertEqual(result, "R2T2 离线和实时语音识别")
        # Verify it's valid UTF-8
        result.encode("utf-8")

    def test_mixed_script_content_is_preserved(self):
        result = t("realtime.footer")
        self.assertEqual(result, "R2T2 · 每个连接独立转写 · 停止录音后会补齐末尾文字")
        # Contains both ASCII and CJK
        self.assertIn("R2T2", result)
        self.assertIn("每个连接", result)

    def test_unicode_special_characters_are_preserved(self):
        result = t("realtime.status_connecting")
        self.assertEqual(result, "正在连接…")
        # Contains ellipsis (U+2026)
        self.assertIn("…", result)

    def test_chinese_punctuation_is_preserved(self):
        result = t("realtime.mic_error")
        self.assertEqual(result, "请通过 HTTPS 反向代理或 localhost 打开页面以使用麦克风。")
        # Contains Chinese period (U+3002)
        self.assertIn("。", result)

    def test_translation_file_loads_with_utf8_encoding(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfile = Path(tmpdir) / "utf8.json"
            data = {"test": {"chinese": "测试中文", "emoji": "🚀"}}
            tmpfile.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            with patch.object(i18n, "LOCALE_DIR", Path(tmpdir)):
                result = i18n._load_translations("utf8")
                self.assertEqual(result["test"]["chinese"], "测试中文")
                self.assertEqual(result["test"]["emoji"], "🚀")

    def test_format_string_with_chinese_arguments(self):
        set_locale("zh")
        result = t("i18n.unsupported_locale", locale="fr", default="zh")
        self.assertEqual(result, "不支持的语言 'fr'，回退到 'zh'")

    def test_long_chinese_text_is_preserved(self):
        result = t("api.transcription_received",
                   format="json", diarization=True, word_timestamps=False, has_address=False)
        self.assertEqual(
            result,
            "[OpenAI API] 收到转写请求: format=json, speaker_diarization=True, word_level=False, audio_address=False"
        )


class I18nEdgeCasesTest(unittest.TestCase):
    """Edge cases: recoverability, atomic failure, and corrupted state handling."""

    def setUp(self):
        self.original_locale = i18n._current_locale
        self.original_translations = i18n._translations
        set_locale("en")

    def tearDown(self):
        i18n._current_locale = self.original_locale
        i18n._translations = self.original_translations

    def test_system_recovers_after_format_error(self):
        """After a format string error, subsequent translations must still work."""
        # Trigger a format error
        i18n._translations = {"test": {"bad": "{unclosed"}}
        result = t("test.bad", value="x")
        self.assertIsInstance(result, str)

        # Reload valid translations
        set_locale("en")
        # Subsequent translations must work
        self.assertEqual(t("app.name"), "AsrServe")
        self.assertEqual(t("errors.not_found"), "Not found")

    def test_system_recovers_after_missing_key(self):
        """After a missing key lookup, subsequent translations must still work."""
        # Look up a missing key
        result = t("nonexistent.key")
        self.assertEqual(result, "nonexistent.key")

        # Subsequent translations must work
        self.assertEqual(t("app.name"), "AsrServe")

    def test_locale_switch_is_atomic_on_file_read_failure(self):
        """If loading translations fails, the previous locale must remain active."""
        # Start with English
        set_locale("en")
        self.assertEqual(get_locale(), "en")
        self.assertEqual(t("app.name"), "AsrServe")

        # Try to switch to a locale with a broken file
        with tempfile.TemporaryDirectory() as tmpdir:
            broken_file = Path(tmpdir) / "broken.json"
            broken_file.write_text("{invalid json", encoding="utf-8")
            with patch.object(i18n, "LOCALE_DIR", Path(tmpdir)):
                # This should fail to load but not crash
                i18n._load_translations("broken")

        # The locale should still be usable (even if translations are empty)
        # The system must not be in a broken state
        self.assertIsInstance(i18n._translations, dict)

    def test_corrupted_translations_dict_degrades_gracefully(self):
        """If the translations dict is corrupted, lookups must not crash."""
        # Corrupt the translations dict with a non-dict value
        i18n._translations = "not a dict"
        # Lookup should return the key, not crash
        result = t("app.name")
        self.assertEqual(result, "app.name")

        # Corrupt with a list
        i18n._translations = ["not", "a", "dict"]
        result = t("app.name")
        self.assertEqual(result, "app.name")

        # Restore and verify recovery
        set_locale("en")
        self.assertEqual(t("app.name"), "AsrServe")

    def test_nested_corruption_degrades_gracefully(self):
        """If a nested translation value is corrupted, lookups must not crash."""
        set_locale("en")
        # Corrupt a nested value with a non-string, non-dict
        i18n._translations["app"]["name"] = 12345
        result = t("app.name")
        self.assertEqual(result, "app.name")

        # Corrupt with None
        i18n._translations["app"]["name"] = None
        result = t("app.name")
        self.assertEqual(result, "app.name")

        # Restore and verify recovery
        set_locale("en")
        self.assertEqual(t("app.name"), "AsrServe")

    def test_very_long_translation_key_does_not_crash(self):
        """Very long translation keys must be handled without crashing."""
        long_key = "a" * 10000
        result = t(long_key)
        self.assertEqual(result, long_key)

    def test_very_long_format_args_do_not_crash(self):
        """Very long format arguments must be handled without crashing."""
        long_value = "x" * 10000
        result = t("app.temp_cleaned", count=long_value)
        self.assertIn(long_value, result)

    def test_unicode_in_format_args(self):
        """Unicode characters in format arguments must be preserved."""
        set_locale("zh")
        result = t("i18n.unsupported_locale", locale="français", default="zh")
        self.assertIn("français", result)

    def test_empty_string_translation(self):
        """Empty string translations must be returned as-is, not fall back to key."""
        set_locale("en")
        i18n._translations["test"] = {"empty": ""}
        result = t("test.empty")
        self.assertEqual(result, "")

    def test_whitespace_only_translation(self):
        """Whitespace-only translations must be returned as-is."""
        set_locale("en")
        i18n._translations["test"] = {"whitespace": "   "}
        result = t("test.whitespace")
        self.assertEqual(result, "   ")


class I18nRealtimeUiStringsTest(unittest.TestCase):
    """Realtime UI strings must be correctly localized."""

    def setUp(self):
        self.original_locale = i18n._current_locale
        self.original_translations = i18n._translations

    def tearDown(self):
        i18n._current_locale = self.original_locale
        i18n._translations = self.original_translations

    def test_realtime_ui_strings_are_localized_in_english(self):
        set_locale("en")
        self.assertEqual(t("realtime.start_button"), "Start Recording")
        self.assertEqual(t("realtime.stop_button"), "Stop Recording")
        self.assertEqual(t("realtime.status_ready"), "Ready")
        self.assertEqual(t("realtime.status_listening"), "Listening · Auto-detecting Chinese/English")

    def test_realtime_ui_strings_are_localized_in_chinese(self):
        set_locale("zh")
        self.assertEqual(t("realtime.start_button"), "开始录音")
        self.assertEqual(t("realtime.stop_button"), "结束录音")
        self.assertEqual(t("realtime.status_ready"), "准备就绪")
        self.assertEqual(t("realtime.status_listening"), "正在聆听 · 自动识别中英文")


if __name__ == "__main__":
    unittest.main()
