"""
Script Generation Module for VoiceVerse
=========================================
Supports three selectable HF models:
  - Qwen/Qwen3-0.6B             (Fast, no auth)
  - meta-llama/Llama-3.1-8B-Instruct  (High Quality, gated — needs HF_TOKEN)
  - openai-community/gpt2       (Baseline, no auth)

Five config parameters drive script structure + audio conditioning:
  style / tone / length / complexity / delivery intensity
"""

import os
import re
from typing import Dict, List


# ── Style Configurations ──────────────────────────────────────────────────────

STYLE_CONFIG = {
    "podcast": {
        "name": "🎙️ Podcast",
        "description": "Conversational two-host discussion",
        "voices": ["Host A", "Host B"],
        "tone": "engaging, conversational, curious",
        "format": "dialogue",
    },
    "debate": {
        "name": "⚔️ Debate",
        "description": "Structured argument between two opposing sides",
        "voices": ["Speaker 1 (Pro)", "Speaker 2 (Con)"],
        "tone": "formal, persuasive, analytical",
        "format": "dialogue",
    },
    "storytelling": {
        "name": "📖 Storytelling",
        "description": "Narrative narration with vivid descriptions",
        "voices": ["Narrator"],
        "tone": "dramatic, immersive, descriptive",
        "format": "monologue",
    },
    "news": {
        "name": "📰 News Report",
        "description": "Professional news broadcast style",
        "voices": ["Anchor"],
        "tone": "authoritative, clear, concise",
        "format": "monologue",
    },
    "lecture": {
        "name": "🎓 Lecture",
        "description": "Educational explanation with examples",
        "voices": ["Professor"],
        "tone": "instructive, clear, structured",
        "format": "monologue",
    },
}


# ── Audio Conditioning Prompt (from 5 params) ─────────────────────────────────

def build_audio_conditioning_prompt(params: Dict) -> str:
    """
    Build the internal audio-conditioning string from the five config parameters.
    PRD §6.6: 'Generate energetic podcast narration with powerful delivery...'
    """
    style = params.get("style", "").split(" ")[0].lower()
    tone = params.get("tone", "Balanced").split(" ")[0].lower()
    length = params.get("length", "Medium").split(" ")[0].lower()
    complexity = params.get("complexity", "Intermediate").split(" ")[0].lower()
    intensity = params.get("intensity", "Balanced").split(" ")[0].lower()

    # Map UI values → audio descriptors
    tone_map = {
        "calm": "calm, measured",
        "energetic": "energetic, upbeat",
        "serious": "serious, deliberate",
        "dramatic": "dramatic, expressive",
    }
    intensity_map = {
        "soft": "soft, gentle delivery",
        "balanced": "balanced, clear delivery",
        "powerful": "powerful, assertive delivery",
    }
    length_map = {
        "short": "concise narration",
        "medium": "moderate-length narration",
        "detailed": "detailed, thorough narration",
    }
    complexity_map = {
        "beginner": "simple vocabulary, easy to follow",
        "intermediate": "moderate vocabulary, clear articulation",
        "advanced": "advanced vocabulary, precise articulation",
    }

    tone_desc = tone_map.get(tone, "balanced")
    intensity_desc = intensity_map.get(intensity, "balanced delivery")
    length_desc = length_map.get(length, "moderate narration")
    complexity_desc = complexity_map.get(complexity, "clear articulation")

    return (
        f"Generate {tone_desc} {style} narration with {intensity_desc}. "
        f"{length_desc.capitalize()}. {complexity_desc.capitalize()}. "
        f"Expressive but professional."
    )


# ── Word Count Target from Length Param ───────────────────────────────────────

def _length_to_words(length_str: str) -> int:
    l = length_str.lower()
    if "short" in l or "1 min" in l:
        return 150
    elif "detailed" in l or "5+" in l:
        return 500
    else:
        return 250  # medium / 3 min


def _complexity_to_vocab(complexity_str: str) -> str:
    c = complexity_str.lower()
    if "beginner" in c:
        return "Use simple words and short sentences. Avoid jargon."
    elif "advanced" in c:
        return "Use precise, domain-specific vocabulary. Assume a knowledgeable audience."
    else:
        return "Use clear language with some technical terms. Define key concepts briefly."


def _tone_instruction(tone_str: str) -> str:
    t = tone_str.lower()
    if "calm" in t:
        return "Speak calmly and steadily. Measured pacing."
    elif "energetic" in t:
        return "Be enthusiastic and lively! High energy throughout."
    elif "serious" in t:
        return "Maintain a serious, professional demeanor throughout."
    elif "dramatic" in t:
        return "Use dramatic pauses and vivid language for emotional impact."
    return "Keep a balanced, engaging tone."


# ── Prompt Builder (style + context + 5 params) ────────────────────────────────

ANTI_HALLUCINATION = (
    "IMPORTANT: Only use information from the retrieved content below. "
    "If information is not present in the retrieved content, say so clearly. "
    "Do NOT add facts or details not present in the source material."
)

def build_prompt_with_params(
    style: str,
    context: str,
    topic: str = "",
    params: Dict = None,
) -> str:
    """Build the full LLM prompt incorporating all 5 config parameters."""
    if params is None:
        params = {}

    topic_line = f'Topic/Title: "{topic}".' if topic else ""
    word_count = _length_to_words(params.get("length", "Medium"))
    vocab_instruction = _complexity_to_vocab(params.get("complexity", "Intermediate"))
    tone_instruction = _tone_instruction(params.get("tone", "Energetic"))
    intensity = params.get("intensity", "Balanced 🔉").split(" ")[0]

    style_instructions = {
        "podcast": f"""You are a podcast script writer. Write an engaging podcast script.

{ANTI_HALLUCINATION}

{topic_line}

Retrieved Knowledge (base your script ONLY on this):
---
{context}
---

Write a podcast with TWO hosts: [Host A] and [Host B].

Required structure:
INTRO: [Host A] introduces the topic. [Host B] adds excitement.
BODY: 3–4 exchanges discussing key points from the material.
CONCLUSION: [Host A] summarizes. [Host B] closes with a call to action.

Format every line as:
[Host A]: <spoken text>
[Host B]: <spoken text>

Style instructions: {tone_instruction} {vocab_instruction}
Delivery: {intensity} intensity.
Target length: ~{word_count} words total. No bullet points. Conversational flow.""",

        "debate": f"""You are a debate script writer. Write a structured debate.

{ANTI_HALLUCINATION}

{topic_line}

Retrieved Knowledge (base arguments ONLY on this):
---
{context}
---

Debate structure with TWO speakers:
- [Speaker 1 (Pro)]: argues FOR the main thesis
- [Speaker 2 (Con)]: argues AGAINST or alternative view

Required structure:
OPENING STATEMENTS (2–3 sentences each)
MAIN ARGUMENT (2 exchanges each with evidence from source)
REBUTTAL (each addresses the other's point)
CLOSING STATEMENTS (each concludes)

Format:
[Speaker 1 (Pro)]: <text>
[Speaker 2 (Con)]: <text>

Style: {tone_instruction} {vocab_instruction} Delivery intensity: {intensity}.
Target: ~{word_count} words total. No bullet lists.""",

        "storytelling": f"""You are a narrative storyteller. Transform this material into a vivid story.

{ANTI_HALLUCINATION}

{topic_line}

Retrieved Knowledge (source material):
---
{context}
---

Write as a single Narrator. Required structure:
INTRO: Hook and topic introduction
BODY: Grounded explanation and narrative journey through the key ideas
CONCLUSION: Summary, key takeaway, and closing line

Style: {tone_instruction} Use vivid language and metaphors. {vocab_instruction}
Delivery intensity: {intensity}. Target: ~{word_count} words. Continuous prose, paragraph breaks only.""",

        "news": f"""You are a professional news anchor. Broadcast this as a news report.

{ANTI_HALLUCINATION}

{topic_line}

Briefing notes (ONLY use this):
---
{context}
---

Write for a SINGLE Anchor. Required structure:
INTRO: Hook and topic introduction
BODY: Key facts, context, and what this means
CONCLUSION: Summary, key takeaway, and professional sign-off

Style: {tone_instruction} Factual and clear. {vocab_instruction}
Delivery intensity: {intensity}. Target: ~{word_count} words. Continuous speech, no labels.""",

        "lecture": f"""You are a university professor. Deliver an educational lecture.

{ANTI_HALLUCINATION}

{topic_line}

Lecture notes (ONLY use this content):
---
{context}
---

Write for a SINGLE Professor. Required structure:
INTRO: Hook and topic introduction — what students will learn
BODY: 2–3 key concepts with explanations and examples from the source
CONCLUSION: Summary, key takeaway, and closing thought

Style: {tone_instruction} {vocab_instruction}
Delivery intensity: {intensity}. Target: ~{word_count} words. Engage the audience.""",
    }

    return style_instructions.get(style, style_instructions["podcast"])


# ── Model Loading & Generation ────────────────────────────────────────────────

def generate_script_with_model(prompt: str, model_id: str) -> str:
    """
    Generate script using the selected HF model.
    Models:
      - Qwen/Qwen3-0.6B                     → no token required
      - meta-llama/Llama-3.1-8B-Instruct    → requires HF_TOKEN env var
      - openai-community/gpt2               → no token required
    """
    hf_token = os.getenv("HF_TOKEN", "").strip() or None

    if "llama" in model_id.lower() and not hf_token:
        raise ValueError(
            "Llama-3.1-8B-Instruct requires a Hugging Face token with gated model access. "
            "Set HF_TOKEN in your Spaces secrets, or choose Qwen or GPT-2."
        )

    try:
        return _hf_transformers_generate(prompt, model_id, hf_token)
    except Exception as e:
        print(f"[{model_id}] Generation failed: {e}")
        # Fallback to inference API if local fails
        if hf_token:
            try:
                return _hf_inference_api(prompt, model_id, hf_token)
            except Exception as e2:
                print(f"[HF Inference API] Also failed: {e2}")
        raise RuntimeError(
            f"Script generation failed with {model_id}. "
            "Try a different model or check your HF_TOKEN."
        ) from e


def _hf_transformers_generate(prompt: str, model_id: str, token: str = None) -> str:
    """Load model locally and generate text."""
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch

    print(f"Loading model: {model_id}")

    kwargs = {"token": token} if token else {}

    if "gpt2" in model_id.lower():
        # GPT-2: text-generation pipeline
        gen = pipeline(
            "text-generation",
            model=model_id,
            max_new_tokens=600,
            do_sample=True,
            temperature=0.75,
            pad_token_id=50256,
            **kwargs,
        )
        out = gen(prompt[:1500])[0]["generated_text"]
        # Strip the prompt prefix
        result = out[len(prompt):].strip() if out.startswith(prompt[:100]) else out.strip()
        return result if result else out

    else:
        # Qwen / Llama — chat/instruct style
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, **kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            **kwargs,
        )
        model.eval()

        # Build chat messages
        messages = [
            {"role": "system", "content": "You are a professional audio script writer. Follow all instructions precisely."},
            {"role": "user", "content": prompt},
        ]

        # Use chat template if available
        if hasattr(tokenizer, "apply_chat_template"):
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            input_text = f"### Instruction:\n{prompt}\n\n### Response:\n"

        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        max_input = min(inputs["input_ids"].shape[1], 1500)
        inputs = {k: v[:, :max_input] for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=600,
                do_sample=True,
                temperature=0.75,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only new tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _hf_inference_api(prompt: str, model_id: str, token: str) -> str:
    """Fallback: HuggingFace Inference API."""
    import requests

    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": prompt[:2000],
        "parameters": {"max_new_tokens": 600, "temperature": 0.75, "return_full_text": False},
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=90)
    resp.raise_for_status()
    result = resp.json()
    if isinstance(result, list) and result:
        return result[0].get("generated_text", "").strip()
    return str(result)


# ── Script Parsing ────────────────────────────────────────────────────────────

def parse_script_to_segments(script: str, style: str) -> List[Dict]:
    """
    Parse the generated script into TTS-ready segments.
    Returns list of {speaker: str, text: str}.
    """
    segments = []
    config = STYLE_CONFIG.get(style, STYLE_CONFIG["podcast"])

    if config["format"] == "dialogue":
        # Match [Speaker]: text
        pattern = r'\[([^\]]+)\]:\s*(.+?)(?=\[[^\]]+\]:|$)'
        matches = re.findall(pattern, script, re.DOTALL)

        for speaker, text in matches:
            text = text.strip()
            if text and len(text) > 5:
                segments.append({"speaker": speaker.strip(), "text": text})

        # Fallback if parsing failed
        if not segments:
            lines = [l.strip() for l in script.split('\n') if l.strip() and len(l.strip()) > 10]
            voices = config["voices"]
            for i, line in enumerate(lines):
                segments.append({"speaker": voices[i % len(voices)], "text": line})

    else:
        # Monologue: split by paragraphs
        paragraphs = [p.strip() for p in script.split('\n\n') if p.strip() and len(p.strip()) > 10]
        narrator = config["voices"][0]
        for para in paragraphs:
            # Skip section headers like "INTRO:", "BODY:", "CONCLUSION:"
            clean = re.sub(r'^(INTRO|BODY|CONCLUSION)\s*:?\s*', '', para, flags=re.IGNORECASE).strip()
            if clean:
                segments.append({"speaker": narrator, "text": clean})

    return segments


# ── Build simple prompt (backwards compat) ────────────────────────────────────

def build_prompt(style: str, context: str, topic: str = "") -> str:
    """Simple wrapper kept for backward compatibility."""
    return build_prompt_with_params(style, context, topic, params={})
