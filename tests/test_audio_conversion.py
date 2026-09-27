"""Regression checks using real FFmpeg decoding."""

import shutil
import subprocess
import tempfile
import unittest
import warnings
from pathlib import Path

import soundfile as sf

from app.core.exceptions import DefaultServerErrorException
from app.utils.audio import normalize_audio_for_asr


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required")
class AudioConversionTests(unittest.TestCase):
    def test_formats_normalize_without_decoder_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for extension in ("wav", "m4a", "mp3", "pcm"):
                with self.subTest(extension=extension):
                    source = Path(directory) / f"input.{extension}"
                    raw_options = ["-f", "s16le"] if extension == "pcm" else []
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-v",
                            "error",
                            "-f",
                            "lavfi",
                            "-i",
                            "sine=frequency=440:duration=0.5",
                            "-ar",
                            "16000" if extension == "pcm" else "44100",
                            "-ac",
                            "1" if extension == "pcm" else "2",
                            *raw_options,
                            str(source),
                        ],
                        check=True,
                        capture_output=True,
                    )
                    # Uploaded files may have a misleading extension.
                    if extension == "m4a":
                        source = source.rename(Path(directory) / "disguised.wav")
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error", message="PySoundFile failed.*")
                        warnings.filterwarnings(
                            "error",
                            message=".*__audioread_load",
                            category=FutureWarning,
                        )
                        normalized = normalize_audio_for_asr(str(source))
                    info = sf.info(normalized.path)
                    self.assertEqual(info.samplerate, 16000)
                    self.assertEqual(info.channels, 1)
                    self.assertAlmostEqual(info.duration, 0.5, delta=0.06)
                    self.assertTrue(source.exists())

    def test_invalid_container_is_rejected_and_output_cleaned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "broken.mp3"
            source.write_bytes(b"not an audio file" * 100)
            with self.assertRaises(DefaultServerErrorException):
                normalize_audio_for_asr(str(source))
            self.assertEqual(list(Path(directory).iterdir()), [source])


if __name__ == "__main__":
    unittest.main()
