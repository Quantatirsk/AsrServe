"""Measure the application's real CPU diarizer, without ASR or network calls."""

import argparse
import asyncio
import hashlib
import json
import platform
import resource
import statistics
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from app.core.config import settings
from app.services.realtime.speakers import SpeakerStream
from app.utils.speaker_diarizer import get_speaker_diarizer


async def stream(pcm: bytes) -> dict:
    session = SpeakerStream()
    assert session.model is not None, "Diarizer failed to load"
    chunks = []
    started = time.perf_counter()
    for offset in range(0, len(pcm), 5120):  # Replay 160 ms packets without sleeping.
        session.feed(pcm[offset : offset + 5120])
        before = session.frames
        tick = time.perf_counter()
        await session.advance()
        assert session.model is not None, "Streaming inference failed"
        if session.frames > before:
            chunks.append(time.perf_counter() - tick)
    session.ended = True
    await session.advance()
    assert session.model is not None and session.finished
    return {
        "seconds": time.perf_counter() - started,
        "frames": session.frames,
        "active_chunks": len(chunks),
        "chunk_p50_ms": float(np.percentile(chunks, 50) * 1000),
        "chunk_p95_ms": float(np.percentile(chunks, 95) * 1000),
        "chunk_max_ms": max(chunks) * 1000,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4, 6])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or any(n < 1 for n in args.threads):
        parser.error("repeats and threads must be positive")
    audio, rate = sf.read(args.audio, dtype="float32")
    if rate != 16000 or audio.ndim != 1 or len(audio) < rate * 2:
        parser.error("use mono 16 kHz audio at least two seconds long")
    assert np.isfinite(audio).all()
    pcm = (np.clip(audio, -1, 32767 / 32768) * 32768).astype("<i2").tobytes()
    settings.DEVICE = "cpu"
    torch.set_num_threads(args.threads[0])
    diarizer = get_speaker_diarizer()
    started = time.perf_counter()
    diarizer.warmup()
    report = {
        "platform": platform.platform(),
        "torch": torch.__version__,
        "dtype": str(diarizer._model.dtype),
        "device": str(diarizer._model.device),
        "audio": str(args.audio),
        "audio_sha256": hashlib.sha256(args.audio.read_bytes()).hexdigest(),
        "audio_seconds": len(audio) / rate,
        "model_load_seconds": time.perf_counter() - started,
        "runs": [],
    }
    for threads in args.threads:
        torch.set_num_threads(threads)
        # Warm both actual application paths; exclude cold decoder/kernel costs.
        diarizer.diarize(str(args.audio))
        asyncio.run(stream(pcm))
        offline, streaming = [], []
        for _ in range(args.repeats):
            started = time.perf_counter()
            result = diarizer.diarize(str(args.audio))
            offline.append(time.perf_counter() - started)
            assert result.segments and np.isfinite(result.probabilities).all()
            streaming.append(asyncio.run(stream(pcm)))
        row = {
            "threads": threads,
            "offline_seconds": offline,
            "offline_rtf": statistics.median(offline) / report["audio_seconds"],
            "speakers": len({s.speaker_id for s in result.segments}),
            "streaming": streaming,
            "streaming_rtf": statistics.median(s["seconds"] for s in streaming)
            / report["audio_seconds"],
        }
        report["runs"].append(row)
        print(json.dumps(row), flush=True)
    # ru_maxrss is bytes on macOS and KiB on Linux; process peak, not model-only.
    report["process_peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
        1024**2 if platform.system() == "Darwin" else 1024
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
