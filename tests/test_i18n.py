"""i18n module must load, switch, translate, and fall back correctly."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.core import i18n


class I18nTest(unittest.TestCase):
    def setUp(self):
        # Save original state
        self.original_locale = i18n._current_locale
        self.original_translations = i18n._translations
        # Reset to known state
        i18n._current_locale = i18n.DEFAULT_LOCALE
        i18n._translations = {}

    def tearDown(self):
        # Restore original state
        i18n._current_locale = self.original_locale
        i18n._translations = self.original_translations

    def test_get_locale_returns_current(self):
        i18n._current_locale = "en"
        self.assertEqual(i18n.get_locale(), "en")
        i18n._current_locale = "zh"
        self.assertEqual(i18n.get_locale(), "zh")

    def test_set_locale_loads_translations(self):
        i18n.set_locale("en")
        self.assertEqual(i18n.get_locale(), "en")
        self.assertIn("app", i18n._translations)
        self.assertIn("i18n", i18n._translations)

    def test_set_locale_switches_back_and_forth(self):
        i18n.set_locale("en")
        self.assertEqual(i18n.get_locale(), "en")
        self.assertEqual(i18n.t("app.name"), "AsrServe")

        i18n.set_locale("zh")
        self.assertEqual(i18n.get_locale(), "zh")
        self.assertEqual(i18n.t("app.description"), "R2T2 离线和实时语音识别")

    def test_set_locale_unsupported_falls_back_to_default(self):
        i18n.set_locale("fr")
        self.assertEqual(i18n.get_locale(), i18n.DEFAULT_LOCALE)

    def test_translate_returns_translation(self):
        i18n.set_locale("en")
        self.assertEqual(i18n.t("app.name"), "AsrServe")
        self.assertEqual(i18n.t("errors.not_found"), "Not found")

    def test_translate_chinese(self):
        i18n.set_locale("zh")
        self.assertEqual(i18n.t("app.name"), "AsrServe")
        self.assertEqual(i18n.t("errors.not_found"), "未找到")

    def test_translate_missing_key_returns_key(self):
        i18n.set_locale("en")
        self.assertEqual(i18n.t("nonexistent.key"), "nonexistent.key")

    def test_translate_nested_key(self):
        i18n.set_locale("en")
        self.assertEqual(i18n.t("api.transcription_complete", chars=42), "[OpenAI API] Transcription complete: 42 characters")

    def test_translate_format_string_interpolation(self):
        i18n.set_locale("en")
        result = i18n.t("app.temp_cleaned", count=5)
        self.assertEqual(result, "Cleaned up 5 expired temp files")

    def test_translate_format_string_chinese(self):
        i18n.set_locale("zh")
        result = i18n.t("app.temp_cleaned", count=5)
        self.assertEqual(result, "已清理 5 个过期临时文件")

    def test_translate_format_error_returns_unformatted(self):
        i18n.set_locale("en")
        # Missing required format argument
        result = i18n.t("app.temp_cleaned")
        # Should return the template without crashing
        self.assertIn("{count}", result)

    def test_translate_extra_kwargs_ignored(self):
        i18n.set_locale("en")
        result = i18n.t("app.name", extra="ignored")
        self.assertEqual(result, "AsrServe")

    def test_translate_non_string_value_returns_key(self):
        # Manually set a non-string value in translations
        i18n._translations = {"test": {"value": 123}}
        self.assertEqual(i18n.t("test.value"), "test.value")

    def test_t_is_shorthand_for_translate(self):
        i18n.set_locale("en")
        self.assertEqual(i18n.t("app.name"), i18n.translate("app.name"))

    def test_init_i18n_reads_environment_variable(self):
        with patch.dict(os.environ, {"ASR_LOCALE": "en"}):
            i18n.init_i18n()
            self.assertEqual(i18n.get_locale(), "en")

    def test_init_i18n_defaults_when_not_set(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove ASR_LOCALE if it exists
            os.environ.pop("ASR_LOCALE", None)
            i18n.init_i18n()
            self.assertEqual(i18n.get_locale(), i18n.DEFAULT_LOCALE)

    def test_init_i18n_invalid_falls_back(self):
        with patch.dict(os.environ, {"ASR_LOCALE": "invalid_locale"}):
            i18n.init_i18n()
            self.assertEqual(i18n.get_locale(), i18n.DEFAULT_LOCALE)

    def test_load_translations_file_not_found(self):
        result = i18n._load_translations("nonexistent")
        self.assertEqual(result, {})

    def test_load_translations_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfile = Path(tmpdir) / "test.json"
            tmpfile.write_text("not valid json", encoding="utf-8")
            with patch.object(i18n, "LOCALE_DIR", Path(tmpdir)):
                # Write a broken file with the right name
                broken = Path(tmpdir) / "nonexistent.json"
                broken.write_text("{invalid", encoding="utf-8")
                result = i18n._load_translations("nonexistent")
                self.assertEqual(result, {})

    def test_load_translations_valid_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfile = Path(tmpdir) / "test.json"
            data = {"test": {"key": "value"}}
            tmpfile.write_text(json.dumps(data), encoding="utf-8")
            with patch.object(i18n, "LOCALE_DIR", Path(tmpdir)):
                result = i18n._load_translations("test")
                self.assertEqual(result, data)

    def test_supported_locales_constant(self):
        self.assertIn("en", i18n.SUPPORTED_LOCALES)
        self.assertIn("zh", i18n.SUPPORTED_LOCALES)

    def test_default_locale_is_zh(self):
        self.assertEqual(i18n.DEFAULT_LOCALE, "zh")

    def test_realtime_ui_strings_en(self):
        i18n.set_locale("en")
        self.assertEqual(i18n.t("realtime.start_button"), "Start Recording")
        self.assertEqual(i18n.t("realtime.stop_button"), "Stop Recording")
        self.assertEqual(i18n.t("realtime.status_ready"), "Ready")

    def test_realtime_ui_strings_zh(self):
        i18n.set_locale("zh")
        self.assertEqual(i18n.t("realtime.start_button"), "开始录音")
        self.assertEqual(i18n.t("realtime.stop_button"), "结束录音")
        self.assertEqual(i18n.t("realtime.status_ready"), "准备就绪")

    def test_health_strings(self):
        i18n.set_locale("en")
        self.assertEqual(i18n.t("health.healthy"), "healthy")
        self.assertEqual(i18n.t("health.running"), "ASR service is running normally")

        i18n.set_locale("zh")
        self.assertEqual(i18n.t("health.healthy"), "healthy")
        self.assertEqual(i18n.t("health.running"), "ASR 服务运行正常")


if __name__ == "__main__":
    unittest.main()
