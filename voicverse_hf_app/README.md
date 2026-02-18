---
title: VoiceVerse Sprint
emoji: 🎙️
colorFrom: rose
colorTo: indigo
sdk: gradio
sdk_version: "4.26.0"
app_file: app.py
pinned: false
license: mit
short_description: Transform documents into AI-powered audio content (RAG + TTS)
---

# 🎙️ VoiceVerse Sprint

> **PGDM & PGDM(BM) 25-27 | Maker Lab | Application Test 2**

Transform your documents into engaging AI-generated audio content — podcasts, debates, storytelling, news, and lectures — powered by RAG + LLM + Neural TTS.

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| 📄 Document Upload | PDF, TXT, DOCX supported |
| 🔍 RAG Pipeline | Semantic chunking + embedding + retrieval |
| ✍️ Script Generation | Style-adapted scripts via LLM (Mixtral/LLaMA) |
| 🎙️ Multi-Voice TTS | Microsoft Neural TTS with distinct voices per speaker |
| ⚔️ Debate Mode | Pro vs Con dual-voice argument generation *(Wow Feature)* |
| 🎨 5 Styles | Podcast, Debate, Storytelling, News, Lecture |

---

## 🏗️ Architecture

```
Document Upload (PDF/TXT/DOCX)
        ↓
   Text Extraction (PyMuPDF / pypdf / python-docx)
        ↓
Sentence-Aware Chunking (500 chars, 100 overlap)
        ↓
Dense Embeddings (all-MiniLM-L6-v2)
        ↓
Semantic Retrieval (cosine similarity, top-k chunks)
        ↓
Style-Adapted Prompt → LLM (Mixtral-8x7B via Together AI / Groq)
        ↓
Script Parsing → Speaker Segments
        ↓
Neural TTS (Microsoft Edge TTS: en-US/GB/AU voices)
        ↓
Audio Concatenation (pydub)
        ↓
🎧 Final MP3 — ready to play in the browser
```

---

## 🔑 API Keys (Optional)

For best script quality, set one of these as a **Space Secret**:

| Secret Name | Service | Get Key |
|-------------|---------|---------|
| `TOGETHER_API_KEY` | Together AI (recommended) | [together.ai](https://together.ai) |
| `GROQ_API_KEY` | Groq (fast, free) | [console.groq.com](https://console.groq.com) |
| `HF_TOKEN` | Hugging Face Inference | Your HF profile settings |

**Without any key**: Falls back to a local small model (lower quality but functional).

**TTS requires no API key** — Microsoft Edge TTS is free and works out of the box.

---

## 📦 Setup (Local)

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/voiceverse
cd voiceverse
pip install -r requirements.txt
python app.py
```

---

## 📁 Project Structure

```
voiceverse/
├── app.py                # Main Gradio application
├── rag_pipeline.py       # Document ingestion, chunking, embedding, retrieval
├── script_generator.py   # LLM-based script generation (5 styles)
├── voice_generator.py    # Multi-voice Neural TTS generation
├── requirements.txt      # Dependencies
└── README.md             # This file
```

---

## 🎭 Output Styles

| Style | Voices | Best For |
|-------|--------|---------|
| 🎙️ Podcast | Host A (US Male) + Host B (US Female) | Topic exploration, discussions |
| ⚔️ Debate | Speaker Pro (British Male) + Speaker Con (Australian Female) | Two-sided analysis |
| 📖 Storytelling | Narrator (US Female) | Case studies, narratives |
| 📰 News | Anchor (US Male) | Research summaries, reports |
| 🎓 Lecture | Professor (British Male) | Educational content |

---

## 🧠 Models & Attribution

| Component | Model | License |
|-----------|-------|---------|
| Embeddings | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Apache 2.0 |
| LLM Script Gen | [Mixtral-8x7B-Instruct](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) | Apache 2.0 |
| LLM Fallback | [LLaMA-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) | Meta License |
| TTS Primary | [edge-tts](https://github.com/rany2/edge-tts) (Microsoft Neural TTS) | MIT |
| TTS Fallback | [gTTS](https://gtts.readthedocs.io/) | MIT |
| Audio | [pydub](https://github.com/jiaaro/pydub) | MIT |

---

## ⚖️ Academic Integrity

- All models and tools are open-source or API-based, fully attributed above
- Audio is AI-generated and labeled as synthetic content
- No voice cloning of real individuals is performed
- Source material is acknowledged; generated content is grounded in uploaded documents

---

## 📊 Evaluation Coverage

| Rubric Component | Implementation |
|-----------------|---------------|
| End-to-End Execution (30%) | Upload → RAG → Script → Audio in single click |
| RAG Grounding (25%) | Semantic retrieval with cosine similarity, context-grounded prompts |
| Deployment & Stability (15%) | Error handling at every step, fallback chains for all components |
| Audio & Content Quality (10%) | Neural TTS (Microsoft Edge), structured scripts with intro/body/outro |
| User Experience (10%) | Clean Gradio UI, Quick + Step-by-Step modes, editable scripts |
| Wow Factor (10%) | ⚔️ Multi-voice Debate Mode with opposing personas |
