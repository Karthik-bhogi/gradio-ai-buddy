"""
VoiceVerse Sprint — Main Gradio Application
==========================================
Transforms documents into AI-generated audio content (podcasts, debates, lectures, etc.)
using RAG + LLM script generation + Neural TTS.

Deployment: Hugging Face Spaces
Author: [Your Name]
"""

import os
import tempfile
import traceback
from typing import Optional, Tuple

import gradio as gr

from rag_pipeline import RAGPipeline
from script_generator import (
    STYLE_CONFIG,
    build_prompt,
    generate_script_with_api,
    parse_script_to_segments,
)
from voice_generator import generate_audio, VOICE_MAP


# ── Global State ───────────────────────────────────────────────────────────────

rag = RAGPipeline()
CURRENT_DOC_INFO = {"chunks": 0, "preview": ""}


# ── Helper Functions ───────────────────────────────────────────────────────────

def format_status(icon: str, message: str, is_error: bool = False) -> str:
    color = "#ff4444" if is_error else "#4CAF50"
    return f'<span style="color:{color}">{icon} {message}</span>'


def get_style_choices() -> list:
    return [(v["name"], k) for k, v in STYLE_CONFIG.items()]


# ── Step 1: Upload Document ────────────────────────────────────────────────────

def process_upload(file) -> Tuple[str, str]:
    """Ingest uploaded document into the RAG pipeline."""
    global CURRENT_DOC_INFO

    if file is None:
        return (
            format_status("❌", "No file uploaded.", is_error=True),
            "",
        )

    try:
        num_chunks, preview = rag.ingest(file.name)
        CURRENT_DOC_INFO = {"chunks": num_chunks, "preview": preview}

        status = format_status(
            "✅",
            f"Document processed! {num_chunks} knowledge chunks ready for retrieval.",
        )
        doc_preview = f"📄 **Document Preview (first 500 chars):**\n\n{preview}"
        return status, doc_preview

    except Exception as e:
        return (
            format_status("❌", f"Error processing document: {str(e)}", is_error=True),
            "",
        )


# ── Step 2: Generate Script ────────────────────────────────────────────────────

def generate_script(style: str, topic: str, api_key: str) -> Tuple[str, str]:
    """Retrieve context and generate a structured script."""
    if CURRENT_DOC_INFO["chunks"] == 0:
        return (
            format_status("❌", "Please upload a document first!", is_error=True),
            "",
        )

    try:
        # Step 1: Retrieve relevant context
        context = rag.retrieve_for_style(style, top_k=6)
        if not context:
            context = rag.get_full_context(max_chars=3000)

        # Step 2: Build prompt
        prompt = build_prompt(style, context, topic)

        # Step 3: Generate script
        status_msg = format_status("⏳", "Generating script with AI... (this may take 20-40 seconds)")

        script = generate_script_with_api(prompt, api_key=api_key.strip() if api_key else None)

        if not script or len(script) < 50:
            raise ValueError("Generated script is too short. Please try again.")

        status = format_status("✅", f"Script generated! Style: {STYLE_CONFIG[style]['name']}")
        return status, script

    except Exception as e:
        tb = traceback.format_exc()
        print(f"Script generation error:\n{tb}")
        return (
            format_status("❌", f"Script generation failed: {str(e)}", is_error=True),
            "",
        )


# ── Step 3: Generate Audio ────────────────────────────────────────────────────

def generate_audio_from_script(
    script: str,
    style: str,
    progress: gr.Progress = gr.Progress(),
) -> Tuple[str, Optional[str]]:
    """Convert script text to audio using Neural TTS."""
    if not script or len(script.strip()) < 20:
        return (
            format_status("❌", "Please generate a script first!", is_error=True),
            None,
        )

    try:
        progress(0, desc="🎙️ Preparing TTS pipeline...")

        # Parse script into segments
        segments = parse_script_to_segments(script, style)

        if not segments:
            raise ValueError("Could not parse script into voice segments.")

        total = len(segments)
        progress(0.1, desc=f"🎤 Synthesizing {total} voice segments...")

        # Generate audio with progress tracking
        def progress_cb(step, total_steps, msg):
            progress(0.1 + 0.8 * (step / total_steps), desc=msg)

        output_dir = tempfile.mkdtemp()
        audio_path = generate_audio(
            segments=segments,
            style=style,
            output_dir=output_dir,
            progress_callback=progress_cb,
        )

        progress(1.0, desc="✅ Audio generation complete!")

        num_segments = len(segments)
        voices_used = list({seg["speaker"] for seg in segments})
        status = format_status(
            "✅",
            f"Audio ready! {num_segments} segments, voices: {', '.join(voices_used)}",
        )
        return status, audio_path

    except Exception as e:
        tb = traceback.format_exc()
        print(f"Audio generation error:\n{tb}")
        return (
            format_status("❌", f"Audio generation failed: {str(e)}", is_error=True),
            None,
        )


# ── Full Pipeline (One-Click) ─────────────────────────────────────────────────

def run_full_pipeline(
    file,
    style: str,
    topic: str,
    api_key: str,
    progress: gr.Progress = gr.Progress(),
) -> Tuple[str, str, str, Optional[str]]:
    """Run the complete pipeline: Upload → RAG → Script → Audio."""

    # Step 1: Upload
    progress(0.0, desc="📄 Processing document...")
    upload_status, doc_preview = process_upload(file)

    if "Error" in upload_status or "No file" in upload_status:
        return upload_status, doc_preview, "", None

    # Step 2: Script
    progress(0.3, desc="✍️ Generating script...")
    script_status, script = generate_script(style, topic, api_key)

    if "failed" in script_status or "Please upload" in script_status:
        return script_status, doc_preview, script, None

    # Step 3: Audio
    progress(0.6, desc="🎙️ Generating audio...")
    audio_status, audio_path = generate_audio_from_script(script, style, progress)

    return audio_status, doc_preview, script, audio_path


# ── Gradio UI ─────────────────────────────────────────────────────────────────

CSS = """
/* ── VoiceVerse Custom Theme ── */
.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto;
}

.voiceverse-header {
    text-align: center;
    padding: 2rem 1rem 1rem;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    margin-bottom: 1.5rem;
    color: white;
}

.voiceverse-header h1 {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #e94560, #f5a623, #7ED321);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}

.voiceverse-header p {
    font-size: 1.1rem;
    opacity: 0.85;
    color: #ccc;
}

.step-card {
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem;
    background: #fafafa;
    margin-bottom: 1rem;
}

.step-label {
    font-weight: 700;
    color: #e94560;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}

.tab-nav button {
    font-weight: 600;
}

.audio-player {
    border-radius: 12px;
    overflow: hidden;
}

footer {
    display: none !important;
}
"""

HEADER_HTML = """
<div class="voiceverse-header">
    <h1>🎙️ VoiceVerse</h1>
    <p>Transform your documents into engaging AI-powered audio content</p>
    <p style="font-size:0.85rem; margin-top:0.5rem; opacity:0.7">
        📄 RAG-powered knowledge retrieval &nbsp;|&nbsp; 🤖 AI script generation &nbsp;|&nbsp; 🎵 Neural Text-to-Speech
    </p>
</div>
"""

STYLE_DESCRIPTIONS = "\n".join(
    [f"- **{v['name']}**: {v['description']}" for v in STYLE_CONFIG.values()]
)

# ── API Key Helper ─────────────────────────────────────────────────────────────

def get_api_key_from_env() -> str:
    """Auto-detect available API key from environment."""
    for env_var in ["TOGETHER_API_KEY", "GROQ_API_KEY", "HF_TOKEN"]:
        val = os.environ.get(env_var, "")
        if val:
            return val
    return ""


# ── Build UI ───────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(
        css=CSS,
        title="VoiceVerse — AI Document to Audio",
        theme=gr.themes.Soft(
            primary_hue="rose",
            secondary_hue="indigo",
            font=gr.themes.GoogleFont("Inter"),
        ),
    ) as demo:

        gr.HTML(HEADER_HTML)

        with gr.Tabs(elem_classes="tab-nav"):

            # ── TAB 1: Quick Generate (One-Click) ──────────────────────────────
            with gr.TabItem("⚡ Quick Generate"):
                gr.Markdown("### Upload your document and generate audio in one click!")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Step 1: Upload Document")
                        quick_file = gr.File(
                            label="📂 Upload PDF, TXT, or DOCX",
                            file_types=[".pdf", ".txt", ".docx"],
                        )

                        gr.Markdown("#### Step 2: Choose Style")
                        quick_style = gr.Dropdown(
                            choices=get_style_choices(),
                            value="podcast",
                            label="🎨 Output Style",
                            info=f"Available styles:\n{STYLE_DESCRIPTIONS}",
                        )

                        quick_topic = gr.Textbox(
                            label="📝 Topic/Title (optional)",
                            placeholder="e.g., 'Introduction to Machine Learning'",
                            max_lines=2,
                        )

                        quick_api_key = gr.Textbox(
                            label="🔑 LLM API Key (optional)",
                            placeholder="Together AI / Groq / HF Token — leave blank to use local model",
                            type="password",
                            value=get_api_key_from_env(),
                            info="Provide a Together AI or Groq key for best script quality.",
                        )

                        quick_btn = gr.Button(
                            "🚀 Generate Audio",
                            variant="primary",
                            size="lg",
                        )

                    with gr.Column(scale=1):
                        quick_status = gr.HTML(label="Status")
                        quick_preview = gr.Markdown(label="Document Preview")
                        quick_script = gr.Textbox(
                            label="📝 Generated Script",
                            lines=10,
                            interactive=True,
                            show_copy_button=True,
                        )
                        quick_audio = gr.Audio(
                            label="🎧 Generated Audio",
                            type="filepath",
                            elem_classes="audio-player",
                        )

                quick_btn.click(
                    fn=run_full_pipeline,
                    inputs=[quick_file, quick_style, quick_topic, quick_api_key],
                    outputs=[quick_status, quick_preview, quick_script, quick_audio],
                )

            # ── TAB 2: Step-by-Step ────────────────────────────────────────────
            with gr.TabItem("🔬 Step-by-Step"):
                gr.Markdown("### Fine-grained control over each pipeline stage")

                # Step 1: Upload
                with gr.Group(elem_classes="step-card"):
                    gr.HTML('<div class="step-label">Step 1 — Document Upload & RAG</div>')
                    with gr.Row():
                        step_file = gr.File(
                            label="📂 Upload File",
                            file_types=[".pdf", ".txt", ".docx"],
                        )
                        with gr.Column():
                            step_upload_btn = gr.Button("📤 Process Document", variant="primary")
                            step_upload_status = gr.HTML()
                            step_doc_preview = gr.Markdown()

                # Step 2: Script
                with gr.Group(elem_classes="step-card"):
                    gr.HTML('<div class="step-label">Step 2 — Script Generation</div>')
                    with gr.Row():
                        with gr.Column(scale=1):
                            step_style = gr.Dropdown(
                                choices=get_style_choices(),
                                value="podcast",
                                label="🎨 Output Style",
                            )
                            step_topic = gr.Textbox(
                                label="📝 Topic/Title (optional)",
                                placeholder="e.g., 'Quantum Computing Basics'",
                            )
                            step_api_key = gr.Textbox(
                                label="🔑 LLM API Key",
                                type="password",
                                placeholder="Together AI / Groq / HF Token",
                                value=get_api_key_from_env(),
                            )
                            step_script_btn = gr.Button("✍️ Generate Script", variant="primary")
                        with gr.Column(scale=1):
                            step_script_status = gr.HTML()
                            step_script = gr.Textbox(
                                label="📝 Script (editable — modify before generating audio!)",
                                lines=12,
                                interactive=True,
                                show_copy_button=True,
                            )

                # Step 3: Audio
                with gr.Group(elem_classes="step-card"):
                    gr.HTML('<div class="step-label">Step 3 — Voice Generation</div>')
                    with gr.Row():
                        with gr.Column(scale=1):
                            step_audio_btn = gr.Button("🎙️ Generate Audio", variant="primary")
                            step_audio_status = gr.HTML()
                        with gr.Column(scale=1):
                            step_audio = gr.Audio(
                                label="🎧 Output Audio",
                                type="filepath",
                                elem_classes="audio-player",
                            )

                # Wire up step-by-step
                step_upload_btn.click(
                    fn=process_upload,
                    inputs=[step_file],
                    outputs=[step_upload_status, step_doc_preview],
                )
                step_script_btn.click(
                    fn=generate_script,
                    inputs=[step_style, step_topic, step_api_key],
                    outputs=[step_script_status, step_script],
                )
                step_audio_btn.click(
                    fn=generate_audio_from_script,
                    inputs=[step_script, step_style],
                    outputs=[step_audio_status, step_audio],
                )

            # ── TAB 3: Style Guide ─────────────────────────────────────────────
            with gr.TabItem("📚 Style Guide"):
                gr.Markdown(f"""
## 🎙️ VoiceVerse — Style Guide

VoiceVerse supports **5 output styles**, each with a distinct voice setup:

---

### 🎙️ Podcast
- **Voices**: Host A (US Male) + Host B (US Female)
- **Format**: Back-and-forth dialogue
- **Best for**: General topic exploration, interviews, discussions

### ⚔️ Debate
- **Voices**: Speaker 1 Pro (British Male) + Speaker 2 Con (Australian Female)
- **Format**: Structured argument with rebuttals
- **Best for**: Controversial topics, two-sided analysis, academic papers

### 📖 Storytelling
- **Voices**: Narrator (US Female)
- **Format**: Immersive narrative prose
- **Best for**: Case studies, historical events, journey-based content

### 📰 News Report
- **Voices**: Anchor (US Male)
- **Format**: Professional broadcast script
- **Best for**: Research summaries, factual reports, announcements

### 🎓 Lecture
- **Voices**: Professor (British Male)
- **Format**: Educational explanation
- **Best for**: Textbook chapters, tutorials, learning materials

---

## 🔑 API Keys for Best Quality

For high-quality script generation, provide one of:
- **Together AI** (recommended): Get free key at [together.ai](https://together.ai)
- **Groq**: Get free key at [console.groq.com](https://console.groq.com)
- **HF Token**: From your Hugging Face account settings

Without an API key, VoiceVerse uses a local fallback model (lower quality).

## 🏗️ Technical Architecture

```
Document Upload (PDF/TXT/DOCX)
        ↓
   Text Extraction
        ↓
Sentence-Aware Chunking (500 chars, 100 overlap)
        ↓
Embedding (all-MiniLM-L6-v2 via sentence-transformers)
        ↓
  Semantic Retrieval (cosine similarity)
        ↓
Style-Adapted Prompt Construction
        ↓
LLM Script Generation (Mixtral/LLaMA via Together AI/Groq)
        ↓
Script Parsing → Voice Segments
        ↓
Neural TTS (Microsoft Edge TTS — en-US/GB/AU voices)
        ↓
Audio Concatenation (pydub)
        ↓
   🎧 Final MP3 Output
```

## 📋 Supported File Types
- **PDF** — Research papers, books, reports
- **TXT** — Plain text documents
- **DOCX** — Word documents
""")

            # ── TAB 4: About ───────────────────────────────────────────────────
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
## 🎙️ VoiceVerse Sprint

**Assignment**: Application Test 2 — VoiceVerse Sprint
**Program**: PGDM & PGDM(BM) 25-27
**Course**: Maker Lab

---

### What This System Does

VoiceVerse transforms uploaded documents into synthetic audio content through a 4-stage AI pipeline:

1. **Document Input & Knowledge Layer**: Extracts and chunks text from PDF/TXT/DOCX, embeds using sentence-transformers, and retrieves relevant passages via cosine similarity.

2. **Script Generation**: Uses LLM (Mixtral/LLaMA) with style-specific prompts to generate structured spoken scripts (intro → body → conclusion).

3. **Voice Generation**: Microsoft Edge Neural TTS with multi-voice support — different voices per speaker/style.

4. **User Interface**: Clean Gradio UI with one-click and step-by-step modes.

---

### Models & Tools Used

| Component | Model/Tool | Source |
|-----------|------------|--------|
| Embeddings | all-MiniLM-L6-v2 | Hugging Face / SBERT |
| LLM (primary) | Mixtral-8x7B-Instruct | Together AI |
| LLM (fallback) | LLaMA-3-8B | Groq |
| TTS (primary) | Microsoft Edge Neural TTS | edge-tts library |
| TTS (fallback) | Google Text-to-Speech | gTTS |
| Audio processing | pydub | PyPI |
| RAG retrieval | cosine similarity | numpy |

### "Wow Factor" Feature: Debate Mode ⚔️
Multi-voice Debate Mode generates structured arguments (Pro vs Con) with distinct voice personas — British male arguing pro, Australian female arguing con — creating an engaging intellectual dialogue from any document.

---

### Academic Integrity Statement
All tools and models are open-source or API-based and fully attributed above. Synthetic audio is AI-generated. No voice cloning of real individuals is performed.
""")

    return demo


# ── Launch ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
