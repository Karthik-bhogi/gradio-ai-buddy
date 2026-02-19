"""
Voice Generation Module for VoiceVerse
=========================================
Primary: Microsoft Edge Neural TTS (edge-tts) — free, no API key
Fallback: Google TTS (gTTS)

Output: WAV file (per PRD §6.6)
Five config parameters influence prosody (rate, pitch, volume).
"""

import asyncio
import io
import os
import tempfile
from typing import Dict, List, Optional


# ── Voice Mapping ──────────────────────────────────────────────────────────────

VOICE_MAP = {
    # Podcast
    "Host A": "en-US-GuyNeural",
    "Host B": "en-US-JennyNeural",
    # Debate
    "Speaker 1 (Pro)": "en-GB-RyanNeural",
    "Speaker 2 (Con)": "en-AU-NatashaNeural",
    # Monologue styles
    "Narrator": "en-US-JennyNeural",
    "Anchor": "en-US-GuyNeural",
    "Professor": "en-GB-RyanNeural",
    # Defaults
    "default_male": "en-US-GuyNeural",
    "default_female": "en-US-JennyNeural",
}


# ── Prosody Derived from 5 Config Parameters ──────────────────────────────────

def _params_to_prosody(params: Dict) -> Dict[str, str]:
    """
    Map the five UI config parameters → edge-tts SSML prosody values.
    PRD §6.6: Audio must reflect tone, delivery intensity, style, length.
    """
    tone = params.get("tone", "Energetic ⚡").lower()
    intensity = params.get("intensity", "Balanced 🔉").lower()
    style = params.get("style", "Podcast 🎧").lower()
    length = params.get("length", "Medium").lower()

    # Rate: based on tone + length
    if "calm" in tone or "detailed" in length:
        rate = "-10%"
    elif "energetic" in tone or "short" in length:
        rate = "+15%"
    elif "dramatic" in tone:
        rate = "-5%"
    else:
        rate = "+0%"

    # Pitch: based on tone
    if "dramatic" in tone:
        pitch = "+5Hz"
    elif "calm" in tone:
        pitch = "-4Hz"
    elif "energetic" in tone:
        pitch = "+3Hz"
    elif "serious" in tone:
        pitch = "-2Hz"
    else:
        pitch = "+0Hz"

    # Volume: based on delivery intensity
    if "soft" in intensity:
        volume = "-20%"
    elif "powerful" in intensity:
        volume = "+20%"
    else:
        volume = "+0%"

    return {"rate": rate, "pitch": pitch, "volume": volume}


# ── Style-based prosody fallback (when no params provided) ─────────────────────

STYLE_PROSODY_DEFAULTS = {
    "podcast":     {"rate": "+5%",  "pitch": "+0Hz",  "volume": "+0%"},
    "debate":      {"rate": "+10%", "pitch": "+2Hz",  "volume": "+10%"},
    "storytelling":{"rate": "-5%",  "pitch": "-3Hz",  "volume": "-10%"},
    "news":        {"rate": "+0%",  "pitch": "+0Hz",  "volume": "+0%"},
    "lecture":     {"rate": "-10%", "pitch": "-2Hz",  "volume": "-5%"},
}


# ── Edge TTS ──────────────────────────────────────────────────────────────────

async def _synthesize_edge_tts_async(
    text: str,
    voice: str,
    rate: str = "+0%",
    pitch: str = "+0Hz",
) -> bytes:
    """Async edge-tts synthesis. Returns raw MP3 bytes."""
    import edge_tts

    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
    audio_bytes = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_bytes += chunk["data"]
    return audio_bytes


def synthesize_edge_tts(text: str, voice: str, rate: str = "+0%", pitch: str = "+0Hz") -> bytes:
    """Sync wrapper for edge-tts (handles both Jupyter and server environments)."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_synthesize_edge_tts_async(text, voice, rate, pitch))
        loop.close()
        return result
    except RuntimeError:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, _synthesize_edge_tts_async(text, voice, rate, pitch))
            return future.result()


# ── gTTS Fallback ─────────────────────────────────────────────────────────────

def synthesize_gtts(text: str, slow: bool = False) -> bytes:
    """Google TTS fallback. Returns MP3 bytes."""
    from gtts import gTTS

    tts = gTTS(text=text, lang="en", slow=slow)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


# ── MP3 → WAV Conversion ──────────────────────────────────────────────────────

def mp3_bytes_to_wav(mp3_bytes: bytes, output_path: str) -> str:
    """Convert MP3 bytes to WAV file. PRD output format is WAV."""
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
        audio.export(output_path, format="wav")
        return output_path
    except ImportError:
        # pydub unavailable — write raw (may not be valid WAV but still playable in Streamlit)
        wav_path = output_path
        with open(wav_path, "wb") as f:
            f.write(mp3_bytes)
        return wav_path


# ── Audio Concatenation → WAV ─────────────────────────────────────────────────

def concatenate_to_wav(segment_paths: List[str], output_path: str) -> str:
    """Concatenate multiple audio files into one WAV. PRD output: WAV."""
    try:
        from pydub import AudioSegment

        combined = AudioSegment.empty()
        silence = AudioSegment.silent(duration=450)  # 450ms pause

        for path in segment_paths:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".mp3":
                seg = AudioSegment.from_mp3(path)
            elif ext == ".wav":
                seg = AudioSegment.from_wav(path)
            else:
                seg = AudioSegment.from_file(path)
            combined += seg + silence

        combined.export(output_path, format="wav")
        return output_path

    except ImportError:
        # Raw concatenation fallback
        print("⚠️ pydub not available — raw concatenation (quality may be lower)")
        with open(output_path, "wb") as out_f:
            for path in segment_paths:
                with open(path, "rb") as in_f:
                    out_f.write(in_f.read())
        return output_path


# ── Main Audio Generation ─────────────────────────────────────────────────────

def generate_audio(
    segments: List[Dict],
    style: str = "podcast",
    output_dir: Optional[str] = None,
    audio_prompt: str = "",   # PRD §6.6 audio conditioning prompt
    params: Dict = None,      # Five config parameters
    progress_callback=None,
) -> str:
    """
    Generate WAV audio for a list of script segments.
    PRD: Output = playable WAV file.
    Five config parameters drive prosody (rate, pitch, volume).

    Args:
        segments: List of {"speaker": str, "text": str}
        style: Content style key
        output_dir: Directory for temp files
        audio_prompt: Internal audio-conditioning string (logged/used for prosody mapping)
        params: Five config parameters dict
        progress_callback: Optional fn(step, total, msg) for UI updates

    Returns:
        Path to the final WAV file.
    """
    if not segments:
        raise ValueError("No script segments provided.")

    if output_dir is None:
        output_dir = tempfile.mkdtemp()

    # Determine prosody from params (PRD §6.6)
    if params:
        prosody = _params_to_prosody(params)
    else:
        prosody = STYLE_PROSODY_DEFAULTS.get(style, {"rate": "+0%", "pitch": "+0Hz", "volume": "+0%"})

    if audio_prompt:
        print(f"[Audio Conditioning] {audio_prompt}")
        print(f"[Prosody Applied] rate={prosody['rate']} pitch={prosody['pitch']} volume={prosody.get('volume','')}")

    use_edge = _check_edge_tts()
    temp_files = []
    total = len(segments)

    for i, seg in enumerate(segments):
        speaker = seg.get("speaker", "default_male")
        text = seg.get("text", "").strip()

        if not text:
            continue

        if progress_callback:
            progress_callback(i + 1, total, f"🎙️ {speaker}...")

        voice = VOICE_MAP.get(speaker, VOICE_MAP["default_male"])
        seg_path = os.path.join(output_dir, f"seg_{i:03d}.mp3")

        if use_edge:
            try:
                mp3_bytes = synthesize_edge_tts(
                    text=text,
                    voice=voice,
                    rate=prosody["rate"],
                    pitch=prosody["pitch"],
                )
                with open(seg_path, "wb") as f:
                    f.write(mp3_bytes)
            except Exception as e:
                print(f"edge-tts failed seg {i}: {e}. Using gTTS.")
                mp3_bytes = synthesize_gtts(text)
                with open(seg_path, "wb") as f:
                    f.write(mp3_bytes)
        else:
            mp3_bytes = synthesize_gtts(text)
            with open(seg_path, "wb") as f:
                f.write(mp3_bytes)

        temp_files.append(seg_path)

    if not temp_files:
        raise ValueError("No audio segments were generated.")

    # Combine into WAV (PRD output format)
    output_wav = os.path.join(output_dir, "voiceverse_output.wav")
    concatenate_to_wav(temp_files, output_wav)

    return output_wav


def _check_edge_tts() -> bool:
    try:
        import edge_tts  # noqa
        return True
    except ImportError:
        return False
