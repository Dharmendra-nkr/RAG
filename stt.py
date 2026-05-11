"""Simple local STT helpers using faster-whisper.

Functions:
- `record_audio(output_path, duration, sample_rate)` : record a short WAV file
- `transcribe_audio(path, model_size, device, language)` : transcribe audio to text

This module keeps dependencies minimal: `faster-whisper`, `sounddevice`, `scipy`, `numpy`.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from scipy import signal

_WHISPER_MODELS: dict[str, object] = {}


def record_audio(output_path: str | Path = "recording.wav", duration: float = 5.0, sample_rate: int = 16000) -> str:
    """Record `duration` seconds from the default input device and save as WAV.

    Returns the path to the written WAV file.
    """
    output_path = str(output_path)
    try:
        frames = int(duration * sample_rate)
        print(f"Recording {duration}s ({frames} frames) at {sample_rate} Hz...")
        recording = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="int16")
        sd.wait()
        wavfile.write(output_path, sample_rate, recording)
        print(f"Saved recording to {output_path}")
        return output_path
    except Exception as exc:
        raise RuntimeError(f"Recording failed: {exc}") from exc


def _normalize_wav(input_path: str, target_rate: int = 16000) -> str:
    """Ensure audio is mono and at `target_rate`. Returns path to normalized WAV.

    If no change is needed, returns the original path.
    """
    sr, data = wavfile.read(input_path)
    if data.ndim > 1:
        data = data.mean(axis=1)

    if sr != target_rate:
        num_samples = round(len(data) * float(target_rate) / sr)
        data = signal.resample(data, num_samples)

    # convert to int16
    if data.dtype != np.int16:
        # scale floats in -1..1 to int16
        if np.issubdtype(data.dtype, np.floating):
            max_val = np.max(np.abs(data)) or 1.0
            data = (data / max_val) * 32767
        data = data.astype(np.int16)

    tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
    wavfile.write(tmp, target_rate, data)
    return str(tmp)


def transcribe_audio(path: str, model_size: str = "small", device: str = "cpu", language: Optional[str] = None) -> str:
    """Transcribe an audio file using `faster-whisper`.

    Returns the transcription text.
    """
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("faster-whisper is required for transcription") from exc

    norm_path = _normalize_wav(path, target_rate=16000)

    key = f"{model_size}:{device}"
    model = _WHISPER_MODELS.get(key)
    if model is None:
        # On CPU, float32 avoids repeated float16 fallback warnings.
        compute_type = "float32" if device == "cpu" else "float16"
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        _WHISPER_MODELS[key] = model

    segments, _ = model.transcribe(norm_path, language=language) if language else model.transcribe(norm_path)
    text = " ".join((segment.text or "").strip() for segment in segments).strip()
    return text.strip()
