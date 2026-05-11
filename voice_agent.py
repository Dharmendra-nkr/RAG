"""Simple voice agent that records or loads audio, transcribes, queries RAG, and prints the answer.

Usage examples:
  python voice_agent.py --duration 4
  python voice_agent.py --file example.wav

By default it prints the RAG answer. TTS can be added later (ElevenLabs) by implementing `speak_text`.
"""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import stt
import rag_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple voice agent: record/transcribe -> RAG -> print")
    parser.add_argument("--file", help="Path to a WAV file to transcribe (skip recording)")
    parser.add_argument("--duration", type=float, default=5.0, help="Recording duration in seconds (when not using --file)")
    parser.add_argument("--model", default="small", help="Whisper model size (small, base, medium, etc.)")
    parser.add_argument("--device", default="cpu", help="Device for whisper (cpu, cuda)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of context chunks to retrieve for RAG")
    parser.add_argument("--tts", action="store_true", help="(optional) speak the answer using ElevenLabs if configured")
    return parser.parse_args()


def speak_text(text: str) -> None:
    """Placeholder for TTS. Implement ElevenLabs TTS in `tts.py` or here when ready."""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("ELEVENLABS_API_KEY not set; skipping TTS.")
        return
    print("TTS requested but no TTS integration is implemented in this helper.")


def main() -> None:
    args = parse_args()

    if args.file:
        audio_path = args.file
        if not Path(audio_path).exists():
            raise SystemExit(f"Audio file not found: {audio_path}")
    else:
        tmp = tempfile.mkstemp(suffix=".wav")[1]
        audio_path = stt.record_audio(tmp, duration=args.duration)

    print("Transcribing...")
    transcript = stt.transcribe_audio(audio_path, model_size=args.model, device=args.device)
    print("Transcript:")
    print(transcript or "(no speech detected)")

    if not transcript:
        return

    print("Querying RAG...")
    answer = rag_pipeline.answer_query(transcript, top_k=args.top_k)
    print("Answer:")
    print(answer)

    if args.tts:
        speak_text(answer)


if __name__ == "__main__":
    main()
