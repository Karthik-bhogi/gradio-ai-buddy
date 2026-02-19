"""
VoiceVerse — Streamlit Application (PRD-Compliant)
===================================================
Single-screen, full-width layout.
Mutually exclusive input modes (text OR document).
Five emoji-based configuration parameters.
Three selectable summarization models.
Config-driven audio generation (WAV output).

Deployment: Hugging Face Spaces (Streamlit SDK)
"""

import os
import tempfile
import traceback

import streamlit as st

from rag_pipeline import RAGPipeline
from script_generator import (
    STYLE_CONFIG,
    build_prompt_with_params,
    generate_script_with_model,
    parse_script_to_segments,
    build_audio_conditioning_prompt,
)
from voice_generator import generate_audio, VOICE_MAP


# ── Page Configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🎙️ VoiceVerse",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── Custom CSS: Full-Screen, Minimal, Clean ────────────────────────────────────

st.markdown("""
<style>
/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 100%; }

/* Header */
.vv-header {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 14px;
    padding: 1.4rem 2rem 1.2rem;
    margin-bottom: 1.2rem;
    text-align: center;
}
.vv-header h1 {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #e94560, #f5a623, #7ED321);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.3rem 0;
}
.vv-header p {
    color: #ccc;
    font-size: 1rem;
    margin: 0;
}

/* Panel labels */
.panel-label {
    font-weight: 700;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #e94560;
    margin-bottom: 0.3rem;
}

/* Active input indicator */
.input-active {
    border-left: 3px solid #4CAF50;
    padding-left: 8px;
    margin-bottom: 0.6rem;
}
.input-inactive {
    opacity: 0.45;
    pointer-events: none;
}

/* Status messages */
.status-ok  { color: #4CAF50; font-weight: 600; }
.status-err { color: #e94560; font-weight: 600; }
.status-run { color: #f5a623; font-weight: 600; }

/* Divider */
.vv-divider { border-top: 1px solid #2a2a3a; margin: 0.8rem 0; }

/* Script box */
.script-box {
    background: #0e1117;
    border: 1px solid #2a2a3a;
    border-radius: 10px;
    padding: 1rem;
    font-family: monospace;
    font-size: 0.85rem;
    white-space: pre-wrap;
    max-height: 320px;
    overflow-y: auto;
    color: #e0e0e0;
}

/* Synthetic label */
.synthetic-label {
    background: #1a1a2e;
    border: 1px solid #e94560;
    border-radius: 6px;
    padding: 0.3rem 0.7rem;
    font-size: 0.75rem;
    color: #e94560;
    display: inline-block;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="vv-header">
  <h1>🎙️ VoiceVerse</h1>
  <p>Transform documents or text into AI-generated structured audio &nbsp;|&nbsp;
     RAG · Multi-Model · Neural TTS</p>
</div>
""", unsafe_allow_html=True)


# ── Session State Init ─────────────────────────────────────────────────────────

if "rag" not in st.session_state:
    st.session_state.rag = RAGPipeline()
if "script_text" not in st.session_state:
    st.session_state.script_text = ""
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
if "doc_ingested" not in st.session_state:
    st.session_state.doc_ingested = False
if "status_msg" not in st.session_state:
    st.session_state.status_msg = ""
if "status_type" not in st.session_state:
    st.session_state.status_type = "ok"  # ok | err | run


# ── Layout: Left Panel | Right Panel ──────────────────────────────────────────

left_col, right_col = st.columns([1, 1], gap="large")


# ══════════════════════════════════════════════════════════════════════════════
# LEFT PANEL — Input + Configuration + Model + Generate
# ══════════════════════════════════════════════════════════════════════════════

with left_col:

    # ── Input Mode (Mutually Exclusive) ────────────────────────────────────────

    st.markdown('<div class="panel-label">📥 Input Source (choose one)</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📂 Upload Document (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        help="Uploading a document disables the text box.",
        key="file_uploader",
    )

    file_is_active = uploaded_file is not None
    text_is_active = not file_is_active

    # Text input — disabled if a file is uploaded
    direct_text = st.text_area(
        "✏️ Or paste text directly",
        height=110,
        placeholder="Paste article, notes, or any text here...",
        disabled=file_is_active,
        help="Disabled when a document is uploaded.",
        key="direct_text_input",
    )

    # Visual active indicator
    if file_is_active:
        st.markdown('<p style="color:#4CAF50;font-size:0.82rem;">✅ Document mode active — text input disabled</p>', unsafe_allow_html=True)
    elif direct_text:
        st.markdown('<p style="color:#4CAF50;font-size:0.82rem;">✅ Text mode active — upload disabled</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:#888;font-size:0.82rem;">⬆️ Upload a document OR paste text above</p>', unsafe_allow_html=True)

    st.markdown('<div class="vv-divider"></div>', unsafe_allow_html=True)

    # ── Model Selector ─────────────────────────────────────────────────────────

    st.markdown('<div class="panel-label">🤖 Summarization Model</div>', unsafe_allow_html=True)

    model_choice = st.radio(
        "Select model",
        options=[
            "⚡ Qwen/Qwen3-0.6B (Fast)",
            "🧠 meta-llama/Llama-3.1-8B-Instruct (High Quality)",
            "📘 openai-community/gpt2 (Baseline)",
        ],
        index=0,
        label_visibility="collapsed",
    )

    MODEL_MAP = {
        "⚡ Qwen/Qwen3-0.6B (Fast)": "Qwen/Qwen3-0.6B",
        "🧠 meta-llama/Llama-3.1-8B-Instruct (High Quality)": "meta-llama/Llama-3.1-8B-Instruct",
        "📘 openai-community/gpt2 (Baseline)": "openai-community/gpt2",
    }
    selected_model = MODEL_MAP[model_choice]

    st.markdown('<div class="vv-divider"></div>', unsafe_allow_html=True)

    # ── Five Configuration Parameters ─────────────────────────────────────────

    st.markdown('<div class="panel-label">⚙️ Configuration</div>', unsafe_allow_html=True)

    cfg_style = st.selectbox(
        "🎙️ Style",
        options=["Podcast 🎧", "Lecture 🎓", "Storytelling 📖", "Debate 🎭"],
        index=0,
    )

    cfg_tone = st.select_slider(
        "🎭 Tone",
        options=["Calm 😌", "Energetic ⚡", "Serious 🧐", "Dramatic 🎬"],
        value="Energetic ⚡",
    )

    cfg_length = st.radio(
        "⏱️ Length",
        options=["Short (1 min) ⏳", "Medium (3 min) ⌛", "Detailed (5+ min) 📜"],
        index=1,
        horizontal=True,
    )

    cfg_complexity = st.select_slider(
        "🧩 Complexity Level",
        options=["Beginner 🌱", "Intermediate 🌿", "Advanced 🌳"],
        value="Intermediate 🌿",
    )

    cfg_intensity = st.radio(
        "🎚️ Delivery Intensity",
        options=["Soft 🔈", "Balanced 🔉", "Powerful 🔊"],
        index=1,
        horizontal=True,
    )

    params = {
        "style": cfg_style,
        "tone": cfg_tone,
        "length": cfg_length,
        "complexity": cfg_complexity,
        "intensity": cfg_intensity,
    }

    st.markdown('<div class="vv-divider"></div>', unsafe_allow_html=True)

    # ── Topic (optional) ───────────────────────────────────────────────────────

    topic = st.text_input(
        "📝 Topic / Title (optional)",
        placeholder="e.g., Introduction to Quantum Computing",
    )

    # ── Generate Button ────────────────────────────────────────────────────────

    generate_btn = st.button(
        "🚀 Generate Script + Audio",
        type="primary",
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# GENERATION LOGIC (runs when button clicked)
# ══════════════════════════════════════════════════════════════════════════════

if generate_btn:
    st.session_state.script_text = ""
    st.session_state.audio_path = None
    st.session_state.status_msg = ""

    # --- Validate input ---
    if not file_is_active and not (direct_text and direct_text.strip()):
        st.session_state.status_msg = "❌ Please upload a document OR paste text before generating."
        st.session_state.status_type = "err"
    else:
        try:
            style_key = cfg_style.split(" ")[0].lower()  # "podcast", "lecture", "storytelling", "debate"

            # ── Step 1: RAG or direct text ─────────────────────────────────────
            with st.spinner("📄 Processing input..."):
                if file_is_active:
                    # Save uploaded file to temp location
                    suffix = "." + uploaded_file.name.split(".")[-1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    num_chunks, preview = st.session_state.rag.ingest(tmp_path)
                    context = st.session_state.rag.retrieve_for_style(style_key, top_k=4)
                    st.session_state.doc_ingested = True
                else:
                    # Direct text — skip retrieval, use as-is
                    context = direct_text.strip()
                    st.session_state.doc_ingested = False

            # ── Step 2: Script Generation ──────────────────────────────────────
            with st.spinner(f"✍️ Generating script with {selected_model}..."):
                prompt = build_prompt_with_params(style_key, context, topic, params)
                script = generate_script_with_model(prompt, selected_model)

                if not script or len(script.strip()) < 40:
                    raise ValueError("Generated script is too short. Try a different model or input.")

                st.session_state.script_text = script

            # ── Step 3: Audio Generation ───────────────────────────────────────
            with st.spinner("🎙️ Generating audio (this may take 20–60 seconds)..."):
                segments = parse_script_to_segments(script, style_key)
                if not segments:
                    raise ValueError("Could not parse script into voice segments.")

                audio_prompt = build_audio_conditioning_prompt(params)
                output_dir = tempfile.mkdtemp()
                audio_path = generate_audio(
                    segments=segments,
                    style=style_key,
                    output_dir=output_dir,
                    audio_prompt=audio_prompt,
                    params=params,
                )
                st.session_state.audio_path = audio_path

            st.session_state.status_msg = f"✅ Done! Script + audio generated using {selected_model}."
            st.session_state.status_type = "ok"

        except Exception as e:
            tb = traceback.format_exc()
            print(f"[VoiceVerse Error]\n{tb}")
            st.session_state.status_msg = f"❌ Error: {str(e)}"
            st.session_state.status_type = "err"


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL — Status + Script + Audio Output
# ══════════════════════════════════════════════════════════════════════════════

with right_col:

    # Status
    if st.session_state.status_msg:
        css_cls = {
            "ok": "status-ok",
            "err": "status-err",
            "run": "status-run",
        }.get(st.session_state.status_type, "status-ok")
        st.markdown(
            f'<div class="{css_cls}">{st.session_state.status_msg}</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="vv-divider"></div>', unsafe_allow_html=True)

    # Script output
    st.markdown('<div class="panel-label">📝 Generated Script</div>', unsafe_allow_html=True)

    if st.session_state.script_text:
        # Editable text area so user can tweak before re-generating audio
        edited_script = st.text_area(
            "Script (editable — modify and re-generate audio if needed)",
            value=st.session_state.script_text,
            height=280,
            label_visibility="collapsed",
            key="script_display",
        )
        st.session_state.script_text = edited_script

        # Re-generate audio from edited script
        if st.button("🔄 Re-generate Audio from Edited Script", use_container_width=True):
            with st.spinner("🎙️ Synthesizing edited script..."):
                try:
                    style_key = cfg_style.split(" ")[0].lower()
                    segments = parse_script_to_segments(edited_script, style_key)
                    audio_prompt = build_audio_conditioning_prompt(params)
                    output_dir = tempfile.mkdtemp()
                    audio_path = generate_audio(
                        segments=segments,
                        style=style_key,
                        output_dir=output_dir,
                        audio_prompt=audio_prompt,
                        params=params,
                    )
                    st.session_state.audio_path = audio_path
                    st.success("✅ Audio regenerated from edited script!")
                except Exception as e:
                    st.error(f"❌ {str(e)}")
    else:
        st.markdown(
            '<div class="script-box" style="color:#555;font-style:italic;">'
            'Your generated script will appear here...'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="vv-divider"></div>', unsafe_allow_html=True)

    # Audio output
    st.markdown('<div class="panel-label">🎧 Generated Audio</div>', unsafe_allow_html=True)

    if st.session_state.audio_path and os.path.exists(st.session_state.audio_path):
        with open(st.session_state.audio_path, "rb") as f:
            audio_bytes = f.read()

        st.audio(audio_bytes, format="audio/wav")

        st.download_button(
            label="⬇️ Download WAV",
            data=audio_bytes,
            file_name="voiceverse_output.wav",
            mime="audio/wav",
            use_container_width=True,
        )

        st.markdown(
            '<div class="synthetic-label">⚠️ AI-Generated Synthetic Audio — Not a real human voice</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="color:#555;font-style:italic;font-size:0.9rem;">'
            '🎵 Audio player will appear here after generation...</div>',
            unsafe_allow_html=True,
        )

    # Configuration Summary (shows what params were used)
    if st.session_state.script_text:
        with st.expander("🔍 Generation Config Used", expanded=False):
            st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| 🎙️ Style | `{cfg_style}` |
| 🎭 Tone | `{cfg_tone}` |
| ⏱️ Length | `{cfg_length}` |
| 🧩 Complexity | `{cfg_complexity}` |
| 🎚️ Intensity | `{cfg_intensity}` |
| 🤖 Model | `{selected_model}` |
| 📊 Mode | `{'Document RAG' if st.session_state.doc_ingested else 'Direct Text'}` |
""")


# ── Footer ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="text-align:center;color:#444;font-size:0.75rem;margin-top:1.5rem;border-top:1px solid #1e1e2e;padding-top:0.8rem;">
  VoiceVerse · PGDM & PGDM(BM) 25-27 · Maker Lab · Synthetic audio is AI-generated · No voice cloning of real individuals
</div>
""", unsafe_allow_html=True)
