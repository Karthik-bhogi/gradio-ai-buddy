"""
Script Generation Module for VoiceVerse Sprint
Generates structured spoken-word scripts from retrieved context.
Supports multiple output styles: podcast, debate, storytelling, news, lecture.
"""

import re
from typing import Dict, List, Tuple


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


# ── Prompt Templates ──────────────────────────────────────────────────────────

def build_prompt(style: str, context: str, topic: str = "") -> str:
    """Build the LLM prompt based on style."""
    topic_line = f'The topic/title is: "{topic}".' if topic else ""

    prompts = {
        "podcast": f"""You are a podcast script writer. Create an engaging podcast episode script.

{topic_line}

Source material (use this as your knowledge base, stay grounded in it):
{context}

Write a podcast script with TWO hosts: "Host A" and "Host B".
Structure:
- INTRO (Host A introduces topic, Host B adds excitement) 
- BODY (3-4 exchanges where hosts discuss key points from the material, ask questions, share insights)
- OUTRO (Host A summarizes takeaways, Host B closes with a call to action)

Format each line as:
[Host A]: <text>
[Host B]: <text>

Keep it natural, engaging, and grounded in the source material. Avoid jargon. 150-250 words total.""",

        "debate": f"""You are a debate script writer. Create a structured academic debate.

{topic_line}

Source material (base arguments on this):
{context}

Write a debate with TWO speakers:
- "Speaker 1 (Pro)": argues FOR the main thesis
- "Speaker 2 (Con)": argues AGAINST or offers an alternative view

Structure:
- OPENING STATEMENTS (each speaker 2-3 sentences)
- MAIN ARGUMENT ROUND (2 exchanges each, with evidence from the source)
- REBUTTAL (each speaker addresses the other's point)
- CLOSING STATEMENTS (each speaker concludes)

Format each line as:
[Speaker 1 (Pro)]: <text>
[Speaker 2 (Con)]: <text>

Be persuasive, use evidence from the source, 200-300 words total.""",

        "storytelling": f"""You are a storytelling narrator. Transform this material into an engaging story.

{topic_line}

Source material:
{context}

Write a narrative story based on the concepts or events in the source material.
Structure:
- OPENING: Set the scene dramatically
- JOURNEY: Walk through the key ideas as a narrative journey
- CLIMAX: The central insight or discovery
- RESOLUTION: What this means for the listener

Write as a single narrator. Use vivid language, metaphors, and a story arc.
Format as continuous prose with clear paragraph breaks.
150-250 words total.""",

        "news": f"""You are a professional news anchor. Report on this topic as a news broadcast.

{topic_line}

Source material (treat as your briefing notes):
{context}

Write a news broadcast script for a SINGLE ANCHOR.
Structure:
- HEADLINE: One impactful sentence
- LEAD: Who, What, When, Where, Why in 2-3 sentences
- DETAILS: Key facts and context from the source (3-4 sentences)
- EXPERT CONTEXT: What this means (2-3 sentences)
- SIGN-OFF: Professional closing

Keep it factual, clear, and authoritative. 150-200 words total.
Format as continuous speech (no [labels]).""",

        "lecture": f"""You are a university professor. Deliver a clear educational lecture on this topic.

{topic_line}

Source material (your lecture notes):
{context}

Write a lecture script for a SINGLE PROFESSOR.
Structure:
- WELCOME & OBJECTIVES: What students will learn today
- CONCEPT 1: First key idea with explanation and example
- CONCEPT 2: Second key idea building on the first
- CONCEPT 3: Third key idea or application
- SUMMARY & TAKEAWAY: Recap and why it matters

Use clear language, define terms, give examples. Engage the imaginary audience.
150-250 words total. Format as continuous speech.""",
    }

    return prompts.get(style, prompts["podcast"])


# ── Script Generation ─────────────────────────────────────────────────────────

def generate_script_with_api(prompt: str, api_key: str = None) -> str:
    """
    Generate script using available LLM API.
    Tries: Together AI → Groq → Hugging Face Inference → fallback template.
    """
    # Try Together AI (fast, free tier available)
    together_key = api_key or os.environ.get("TOGETHER_API_KEY", "")
    if together_key:
        try:
            return _together_generate(prompt, together_key)
        except Exception as e:
            print(f"Together AI failed: {e}")

    # Try Groq
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        try:
            return _groq_generate(prompt, groq_key)
        except Exception as e:
            print(f"Groq failed: {e}")

    # Try HF Inference API
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        try:
            return _hf_generate(prompt, hf_token)
        except Exception as e:
            print(f"HF Inference failed: {e}")

    # Final fallback: local transformers
    return _local_generate(prompt)


def _together_generate(prompt: str, api_key: str) -> str:
    import requests
    response = requests.post(
        "https://api.together.xyz/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 600,
            "temperature": 0.7,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def _groq_generate(prompt: str, api_key: str) -> str:
    import requests
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 600,
            "temperature": 0.7,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def _hf_generate(prompt: str, token: str) -> str:
    import requests
    response = requests.post(
        "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
        headers={"Authorization": f"Bearer {token}"},
        json={"inputs": prompt, "parameters": {"max_new_tokens": 600, "temperature": 0.7}},
        timeout=90,
    )
    response.raise_for_status()
    result = response.json()
    if isinstance(result, list):
        return result[0].get("generated_text", prompt)[len(prompt):].strip()
    return str(result)


def _local_generate(prompt: str) -> str:
    """Fallback: use a small local model via transformers pipeline."""
    try:
        from transformers import pipeline
        generator = pipeline(
            "text-generation",
            model="facebook/opt-125m",
            max_new_tokens=400,
            do_sample=True,
            temperature=0.8,
        )
        result = generator(prompt[:1000])[0]["generated_text"]
        return result[len(prompt):].strip()
    except Exception as e:
        print(f"Local generation failed: {e}")
        return _template_fallback(prompt)


def _template_fallback(prompt: str) -> str:
    """Emergency fallback with template-based script."""
    return """[Host A]: Welcome to today's episode! We're diving into some fascinating material that our listeners have been curious about.

[Host B]: That's right! And today's content is particularly interesting because it covers some really important concepts that affect many people.

[Host A]: Let's start from the beginning. Based on our research, the core idea here is about understanding complex systems and how they work together in practice.

[Host B]: Exactly. And what I find most compelling is how this applies to real-world situations. It's not just theory — these are practical insights.

[Host A]: One of the key points we want to highlight is that success in this area requires careful preparation and systematic thinking.

[Host B]: Well said. And for our listeners just getting started, the most important takeaway is to focus on fundamentals before moving to advanced concepts.

[Host A]: That wraps up today's episode. Thank you for joining us!

[Host B]: Until next time, keep learning and stay curious!"""


# ── Script Parsing ────────────────────────────────────────────────────────────

def parse_script_to_segments(script: str, style: str) -> List[Dict]:
    """
    Parse the generated script into segments for TTS.
    Returns list of {speaker, text} dicts.
    """
    segments = []
    config = STYLE_CONFIG.get(style, STYLE_CONFIG["podcast"])

    if config["format"] == "dialogue":
        # Parse [Speaker]: text format
        pattern = r'\[([^\]]+)\]:\s*(.+?)(?=\[[^\]]+\]:|$)'
        matches = re.findall(pattern, script, re.DOTALL)

        for speaker, text in matches:
            text = text.strip()
            if text:
                segments.append({"speaker": speaker.strip(), "text": text})

        # If parsing failed, treat as single narrator
        if not segments:
            lines = [l.strip() for l in script.split('\n') if l.strip()]
            voices = config["voices"]
            for i, line in enumerate(lines):
                segments.append({
                    "speaker": voices[i % len(voices)],
                    "text": line
                })
    else:
        # Monologue: split by paragraphs
        paragraphs = [p.strip() for p in script.split('\n\n') if p.strip()]
        narrator = config["voices"][0]
        for para in paragraphs:
            segments.append({"speaker": narrator, "text": para})

    return segments


import os
