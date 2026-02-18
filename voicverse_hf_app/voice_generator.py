"""
Voice Generation Module for VoiceVerse Sprint
Multi-voice TTS supporting edge-tts (primary) and gTTS (fallback).

Edge-TTS voices (Microsoft Azure Neural TTS - FREE, no API key needed):
- en-US-GuyNeural      → male voice 1
- en-US-JennyNeural    → female voice 1
- en-GB-RyanNeural     → British male
- en-GB-SoniaNeural    → British female
- en-AU-NatashaNeural  → Australian female
"""

import asyncio
import os
import tempfile
from typing import Dict, List, Optional


# ── Voice Mapping ──────────────────────────────────────────────────────────────

VOICE_MAP = {
    # Podcast voices
    "Host A": "en-US-GuyNeural",
    "Host B": "en-US-JennyNeural",
    # Debate voices
    "Speaker 1 (Pro)": "en-GB-RyanNeural",
    "Speaker 2 (Con)": "en-AU-NatashaNeural",
    # Monologue voices
    "Narrator": "en-US-JennyNeural",
    "Anchor": "en-US-GuyNeural",
    "Professor": "en-GB-RyanNeural",
    # Fallback
    "default_male": "en-US-GuyNeural",
    "default_female": "en-US-JennyNeural",
}

# Style-to-voice rate and pitch (edge-tts SSML prosody)
STYLE_PROSODY = {
    "podcast": {"rate": "+5%", "pitch": "+0Hz"},
    "debate": {"rate": "+10%", "pitch": "+2Hz"},
    "storytelling": {"rate": "-5%", "pitch": "-3Hz"},
    "news": {"rate": "+0%", "pitch": "+0Hz"},
    "lecture": {"rate": "-10%", "pitch": "-2Hz"},
}


# ── Edge TTS (Primary) ────────────────────────────────────────────────────────

async def _synthesize_edge_tts(text: str, voice: str, rate: str = "+0%", pitch: str = "+0Hz") -> bytes:
    """
    Generate audio using edge-tts (Microsoft Azure Neural TTS, free, no key needed).
    Returns raw MP3 bytes.
    """
    import edge_tts

    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)

    audio_bytes = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_bytes += chunk["data"]

    return audio_bytes


def synthesize_edge_tts(text: str, voice: str, rate: str = "+0%", pitch: str = "+0Hz") -> bytes:
    """Sync wrapper for edge-tts."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_synthesize_edge_tts(text, voice, rate, pitch))
        loop.close()
        return result
    except RuntimeError:
        # If loop already running (Jupyter/Gradio)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, _synthesize_edge_tts(text, voice, rate, pitch))
            return future.result()


# ── gTTS (Fallback) ───────────────────────────────────────────────────────────

def synthesize_gtts(text: str, lang: str = "en", slow: bool = False) -> bytes:
    """Generate audio using Google Text-to-Speech (requires internet, free)."""
    from gtts import gTTS
    import io

    tts = gTTS(text=text, lang=lang, slow=slow)
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer.read()


# ── Audio Concatenation ───────────────────────────────────────────────────────

def concatenate_audio_files(audio_files: List[str], output_path: str) -> str:
    """Concatenate multiple MP3 files into one using pydub."""
    try:
        from pydub import AudioSegment

        combined = AudioSegment.empty()
        silence = AudioSegment.silent(duration=400)  # 400ms pause between segments

        for audio_file in audio_files:
            segment = AudioSegment.from_mp3(audio_file)
            combined += segment + silence

        combined.export(output_path, format="mp3")
        return output_path

    except ImportError:
        # Fallback: concatenate raw bytes (won't have proper pauses but works)
        print("⚠️ pydub not available, using raw concatenation")
        with open(output_path, "wb") as out_f:
            for audio_file in audio_files:
                with open(audio_file, "rb") as in_f:
                    out_f.write(in_f.read())
        return output_path


# ── Main Voice Generation ─────────────────────────────────────────────────────

def generate_audio(
    segments: List[Dict],
    style: str = "podcast",
    output_dir: Optional[str] = None,
    progress_callback=None,
) -> str:
    """
    Generate audio for a list of script segments.

    Args:
        segments: List of {"speaker": str, "text": str}
        style: Content style for prosody settings
        output_dir: Directory to save temp files (uses tempdir if None)
        progress_callback: Optional function(step, total, message) for UI updates

    Returns:
        Path to the final combined MP3 file.
    """
    if not segments:
        raise ValueError("No script segments provided.")

    if output_dir is None:
        output_dir = tempfile.mkdtemp()

    prosody = STYLE_PROSODY.get(style, {"rate": "+0%", "pitch": "+0Hz"})
    temp_files = []
    total = len(segments)

    use_edge_tts = _check_edge_tts()

    for i, segment in enumerate(segments):
        speaker = segment.get("speaker", "default_male")
        text = segment.get("text", "").strip()

        if not text:
            continue

        if progress_callback:
            progress_callback(i + 1, total, f"🎙️ Generating voice for: {speaker}...")

        # Get voice for this speaker
        voice = VOICE_MAP.get(speaker, VOICE_MAP["default_male"])

        # Generate audio chunk
        segment_path = os.path.join(output_dir, f"segment_{i:03d}.mp3")

        if use_edge_tts:
            try:
                audio_bytes = synthesize_edge_tts(
                    text=text,
                    voice=voice,
                    rate=prosody["rate"],
                    pitch=prosody["pitch"],
                )
                with open(segment_path, "wb") as f:
                    f.write(audio_bytes)
            except Exception as e:
                print(f"edge-tts failed for segment {i}: {e}. Falling back to gTTS.")
                audio_bytes = synthesize_gtts(text)
                with open(segment_path, "wb") as f:
                    f.write(audio_bytes)
        else:
            # Use gTTS
            audio_bytes = synthesize_gtts(text)
            with open(segment_path, "wb") as f:
                f.write(audio_bytes)

        temp_files.append(segment_path)

    if not temp_files:
        raise ValueError("No audio segments were generated.")

    # Combine all segments
    output_path = os.path.join(output_dir, "voiceverse_output.mp3")
    concatenate_audio_files(temp_files, output_path)

    return output_path


def _check_edge_tts() -> bool:
    """Check if edge-tts is available."""
    try:
        import edge_tts
        return True
    except ImportError:
        return False


# ── Voice Preview ─────────────────────────────────────────────────────────────

def generate_voice_preview(style: str) -> Optional[str]:
    """Generate a short voice preview for the selected style."""
    preview_texts = {
        "podcast": {
            "Host A": "Welcome to VoiceVerse! I'm your host, and today we have an exciting episode.",
            "Host B": "That's right! And I can't wait to dive into today's fascinating topic.",
        },
        "debate": {
            "Speaker 1 (Pro)": "I firmly believe that the evidence supports our position on this matter.",
            "Speaker 2 (Con)": "While that perspective has merit, I must respectfully disagree with the conclusion.",
        },
        "storytelling": {
            "Narrator": "Once upon a time, in a world shaped by ideas, a discovery changed everything...",
        },
        "news": {
            "Anchor": "Good evening. Tonight's top story: A breakthrough in understanding has researchers excited.",
        },
        "lecture": {
            "Professor": "Today we'll explore three fundamental concepts that form the foundation of this subject.",
        },
    }

    segments = []
    for speaker, text in preview_texts.get(style, {}).items():
        segments.append({"speaker": speaker, "text": text})

    if not segments:
        return None

    try:
        return generate_audio(segments, style=style)
    except Exception as e:
        print(f"Preview generation failed: {e}")
        return None
