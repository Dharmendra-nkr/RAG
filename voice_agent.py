"""Simple voice agent that records or loads audio, transcribes, queries RAG, and prints the answer.

Usage examples:
  python voice_agent.py --duration 4
  python voice_agent.py --file example.wav

By default it prints the RAG answer. TTS can be added later (ElevenLabs) by implementing `speak_text`.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import tempfile
import sys
import re
import time
from pathlib import Path
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

import stt
import rag_pipeline


def _configure_console() -> None:
    """Make stdout/stderr line-buffered so progress appears as it happens."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True, write_through=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple voice agent: record/transcribe -> RAG -> print")
    parser.add_argument("--file", help="Path to a WAV file to transcribe (skip recording)")
    parser.add_argument("--duration", type=float, default=5.0, help="Recording duration in seconds (when not using --file)")
    parser.add_argument(
        "--countdown",
        type=int,
        default=3,
        help="Seconds to wait before recording starts so you can get ready to speak.",
    )
    parser.add_argument("--input-device", type=int, help="Optional sounddevice input device index")
    parser.add_argument("--model", default="small", help="Whisper model size (small, base, medium, etc.)")
    parser.add_argument("--device", default="cpu", help="Device for whisper (cpu, cuda)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of context chunks to retrieve for RAG")
    parser.add_argument("--tts", action="store_true", help="(optional) speak the answer using ElevenLabs if configured")
    parser.add_argument("--tts-voice-id", help="Optional ElevenLabs voice ID override")
    parser.add_argument("--tts-model-id", help="Optional ElevenLabs model ID override")
    parser.add_argument("--tts-output-format", help="Optional ElevenLabs output format override")
    parser.add_argument("--list-devices", action="store_true", help="Show audio devices and exit")
    return parser.parse_args()


def speak_text(
    text: str,
    voice_id: str | None = None,
    model_id: str | None = None,
    output_format: str | None = None,
) -> None:
    """Speak text with ElevenLabs when configured, otherwise fall back locally."""
    voice_text = format_for_voiceover(text)
    api_key = get_elevenlabs_api_key()
    if api_key:
        try:
            audio_path = synthesize_elevenlabs_voiceover(
                voice_text,
                api_key=api_key,
                voice_id=voice_id,
                model_id=model_id,
                output_format=output_format,
            )
            print(f"ElevenLabs voiceover saved to: {audio_path}", flush=True)
            play_audio_file(audio_path)
            return
        except Exception as exc:
            print(f"ElevenLabs TTS failed, falling back to local speech: {exc}")

    try:
        import pyttsx3
    except Exception:
        print(voice_text)
        return

    engine = pyttsx3.init()
    engine.say(voice_text)
    engine.runAndWait()


def get_elevenlabs_api_key() -> str | None:
    """Read the ElevenLabs API key from either supported env var name."""
    return os.getenv("ELEVENLABS_API_KEY") or os.getenv("ELEVEN_LABS_API_KEY")


def format_for_voiceover(text: str) -> str:
    """Turn raw model output into a cleaner narration script."""
    cleaned = text.strip()
    cleaned = re.sub(r"```.*?```", " ", cleaned, flags=re.S)
    cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", cleaned)
    cleaned = re.sub(r"^\s*[-*+]\s+", "", cleaned, flags=re.M)
    cleaned = re.sub(r"^\s*\d+\.\s+", "", cleaned, flags=re.M)
    cleaned = re.sub(r"[#_>*~]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def synthesize_elevenlabs_voiceover(
    text: str,
    api_key: str,
    voice_id: str | None = None,
    model_id: str | None = None,
    output_format: str | None = None,
) -> Path:
    """Generate an ElevenLabs MP3 voiceover for the given text."""
    if not text:
        raise ValueError("No text was provided for voiceover synthesis.")

    voice_id = voice_id or os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
    model_id = model_id or os.getenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5")
    output_format = output_format or os.getenv("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128")

    # ElevenLabs text-to-speech endpoint: https://api.elevenlabs.io/v1/text-to-speech/:voice_id
    url = (
        f"https://api.elevenlabs.io/v1/text-to-speech/"
        f"{urllib_parse.quote(voice_id, safe='')}"
        f"?output_format={urllib_parse.quote(output_format, safe='')}"
    )
    payload = json.dumps(
        {
            "text": text,
            "model_id": model_id,
        }
    ).encode("utf-8")
    request = urllib_request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
    )

    try:
        with urllib_request.urlopen(request, timeout=120) as response:
            audio_bytes = response.read()
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
        raise RuntimeError(f"ElevenLabs API request failed with HTTP {exc.code}: {detail or exc.reason}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"ElevenLabs API request failed: {exc.reason}") from exc

    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    try:
        tmp.write(audio_bytes)
        tmp.flush()
    finally:
        tmp.close()

    return Path(tmp.name)


def play_audio_file(path: Path) -> None:
    """Play a generated audio file on the local machine when possible."""
    if sys.platform.startswith("win") and hasattr(os, "startfile"):
        os.startfile(str(path))
        return

    import webbrowser

    webbrowser.open(path.resolve().as_uri())


def main() -> None:
    _configure_console()
    args = parse_args()
    cleanup_audio = False

    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return

    if args.input_device is not None:
        import sounddevice as sd

        sd.default.device = (args.input_device, None)

    if args.file:
        audio_path = args.file
        if not Path(audio_path).exists():
            raise SystemExit(f"Audio file not found: {audio_path}")
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        if args.countdown > 0:
            prompt = f"Get ready. Recording starts in {args.countdown} seconds."
            print(prompt, flush=True)
            for remaining in range(args.countdown, 0, -1):
                print(f"Starting in {remaining}...", flush=True)
                time.sleep(1)
            print("Recording now. Speak your question.", flush=True)
            if args.tts:
                speak_text(
                    "Recording now. Speak your question.",
                    args.tts_voice_id,
                    args.tts_model_id,
                    args.tts_output_format,
                )
        audio_path = stt.record_audio(tmp.name, duration=args.duration, input_device=args.input_device)
        cleanup_audio = True

    try:
        print("Transcribing...")
        transcript = stt.transcribe_audio(audio_path, model_size=args.model, device=args.device)
        print("Transcript:")
        print(transcript or "(no speech detected)")

        if not transcript:
            print("No speech was detected. Try a longer duration, a quieter room, or a different input device.", flush=True)
            return

        print("Querying RAG...")
        answer = rag_pipeline.answer_query(transcript, top_k=args.top_k)
        print("Answer:")
        print(answer)

        if args.tts:
            speak_text(answer, args.tts_voice_id, args.tts_model_id, args.tts_output_format)
    finally:
        if cleanup_audio:
            with contextlib.suppress(OSError):
                Path(audio_path).unlink()


if __name__ == "__main__":
    main()
