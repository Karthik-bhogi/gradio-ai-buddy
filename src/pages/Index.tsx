import { useState } from "react";

const files = [
  {
    name: "app.py",
    icon: "🚀",
    description: "Main Gradio application — UI, pipeline orchestration, all tabs",
    size: "~250 lines",
    badge: "ENTRY POINT",
    badgeColor: "bg-rose-500",
  },
  {
    name: "rag_pipeline.py",
    icon: "🔍",
    description: "Document ingestion, chunking, sentence-transformers embedding, cosine retrieval",
    size: "~180 lines",
    badge: "RAG",
    badgeColor: "bg-indigo-500",
  },
  {
    name: "script_generator.py",
    icon: "✍️",
    description: "Style-based LLM prompts → structured scripts. Supports Together AI, Groq, HF, local fallback",
    size: "~200 lines",
    badge: "LLM",
    badgeColor: "bg-violet-500",
  },
  {
    name: "voice_generator.py",
    icon: "🎙️",
    description: "Multi-voice Neural TTS via edge-tts (Microsoft) + gTTS fallback + pydub audio concat",
    size: "~180 lines",
    badge: "TTS",
    badgeColor: "bg-emerald-500",
  },
  {
    name: "requirements.txt",
    icon: "📦",
    description: "All Python dependencies for Hugging Face Spaces deployment",
    size: "~20 lines",
    badge: "CONFIG",
    badgeColor: "bg-amber-500",
  },
  {
    name: "README.md",
    icon: "📄",
    description: "HF Spaces metadata + full documentation, architecture diagram, model attribution",
    size: "~120 lines",
    badge: "DOCS",
    badgeColor: "bg-sky-500",
  },
];

const pipeline = [
  { step: "1", label: "Upload", icon: "📂", desc: "PDF / TXT / DOCX" },
  { step: "2", label: "Chunk & Embed", icon: "🔍", desc: "all-MiniLM-L6-v2" },
  { step: "3", label: "Retrieve", icon: "🎯", desc: "Cosine similarity" },
  { step: "4", label: "Generate Script", icon: "✍️", desc: "Mixtral / LLaMA" },
  { step: "5", label: "Synthesize Voice", icon: "🎙️", desc: "Edge TTS (multi-voice)" },
  { step: "6", label: "Play Audio", icon: "🎧", desc: "MP3 output" },
];

const styles = [
  { name: "Podcast", icon: "🎙️", voices: "Host A (US Male) + Host B (US Female)", wow: false },
  { name: "Debate", icon: "⚔️", voices: "Speaker Pro (British Male) + Speaker Con (Australian Female)", wow: true },
  { name: "Storytelling", icon: "📖", voices: "Narrator (US Female)", wow: false },
  { name: "News Report", icon: "📰", voices: "Anchor (US Male)", wow: false },
  { name: "Lecture", icon: "🎓", voices: "Professor (British Male)", wow: false },
];

const rubric = [
  { component: "End-to-End Execution", weight: "30%", coverage: "One-click pipeline: Upload → RAG → Script → Audio" },
  { component: "RAG Grounding", weight: "25%", coverage: "Semantic retrieval, context-grounded prompts, min hallucination" },
  { component: "Deployment & Stability", weight: "15%", coverage: "Fallback chains for TTS & LLM, error handling at every step" },
  { component: "Audio & Content Quality", weight: "10%", coverage: "Neural TTS, structured intro/body/outro, engaging delivery" },
  { component: "User Experience", weight: "10%", coverage: "Quick + Step-by-Step modes, editable scripts, clean Gradio UI" },
  { component: "Wow Factor", weight: "10%", coverage: "⚔️ Multi-voice Debate Mode with opposing personas (Pro vs Con)" },
];

export default function Index() {
  const [copied, setCopied] = useState("");

  const copyCmd = (cmd: string, id: string) => {
    navigator.clipboard.writeText(cmd);
    setCopied(id);
    setTimeout(() => setCopied(""), 2000);
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Hero */}
      <div className="hero-gradient text-center px-6 py-16">
        <div className="inline-flex items-center gap-2 bg-primary/10 border border-primary/20 rounded-full px-4 py-1.5 text-sm font-medium text-primary mb-6">
          🎓 PGDM &amp; PGDM(BM) 25-27 · Maker Lab · Application Test 2
        </div>
        <h1 className="text-5xl font-black tracking-tight mb-4 text-foreground">
          🎙️ VoiceVerse Sprint
        </h1>
        <p className="text-xl text-muted-foreground max-w-2xl mx-auto mb-2">
          Transform documents into AI-powered audio content
        </p>
        <p className="text-sm text-muted-foreground max-w-xl mx-auto">
          RAG Pipeline · LLM Script Generation · Neural Text-to-Speech · Gradio on Hugging Face
        </p>

        <div className="flex justify-center gap-3 mt-8 flex-wrap">
          <span className="badge-pill bg-rose-100 text-rose-700">Python 3.10+</span>
          <span className="badge-pill bg-indigo-100 text-indigo-700">Gradio 4.x</span>
          <span className="badge-pill bg-violet-100 text-violet-700">sentence-transformers</span>
          <span className="badge-pill bg-emerald-100 text-emerald-700">edge-tts</span>
          <span className="badge-pill bg-amber-100 text-amber-700">Together AI / Groq</span>
        </div>
      </div>

      <div className="max-w-5xl mx-auto px-6 py-12 space-y-12">

        {/* Pipeline */}
        <section>
          <h2 className="section-title">🏗️ Pipeline Architecture</h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            {pipeline.map((step, i) => (
              <div key={step.step} className="pipeline-card">
                <div className="pipeline-step">{step.step}</div>
                <div className="text-2xl mb-1">{step.icon}</div>
                <div className="font-semibold text-sm text-foreground">{step.label}</div>
                <div className="text-xs text-muted-foreground mt-1">{step.desc}</div>
                {i < pipeline.length - 1 && (
                  <div className="pipeline-arrow">→</div>
                )}
              </div>
            ))}
          </div>
        </section>

        {/* Files */}
        <section>
          <h2 className="section-title">📁 Project Files</h2>
          <p className="text-muted-foreground mb-5">
            Download the files from the <code className="code-inline">voicverse_hf_app/</code> folder and upload to your Hugging Face Space.
          </p>
          <div className="space-y-3">
            {files.map((file) => (
              <div key={file.name} className="file-card">
                <div className="flex items-start gap-4">
                  <span className="text-2xl">{file.icon}</span>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1 flex-wrap">
                      <code className="font-mono font-bold text-primary">{file.name}</code>
                      <span className={`badge-pill text-white text-xs ${file.badgeColor}`}>
                        {file.badge}
                      </span>
                      <span className="text-xs text-muted-foreground">{file.size}</span>
                    </div>
                    <p className="text-sm text-muted-foreground">{file.description}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Styles */}
        <section>
          <h2 className="section-title">🎨 Output Styles</h2>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {styles.map((s) => (
              <div key={s.name} className={`style-card ${s.wow ? "style-card-wow" : ""}`}>
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-xl">{s.icon}</span>
                  <span className="font-bold text-foreground">{s.name}</span>
                  {s.wow && (
                    <span className="badge-pill bg-rose-500 text-white text-xs ml-auto">WOW ⭐</span>
                  )}
                </div>
                <p className="text-xs text-muted-foreground">
                  <span className="font-medium">Voices:</span> {s.voices}
                </p>
              </div>
            ))}
          </div>
        </section>

        {/* Deployment Steps */}
        <section>
          <h2 className="section-title">🚀 Deploy to Hugging Face Spaces</h2>
          <div className="space-y-4">
            {[
              {
                id: "step1",
                num: "1",
                title: "Create a new Space",
                content: "Go to huggingface.co/spaces → New Space → choose Gradio SDK → Python 3.10",
                cmd: null,
              },
              {
                id: "step2",
                num: "2",
                title: "Upload all 4 files",
                content: "Upload app.py, rag_pipeline.py, script_generator.py, voice_generator.py, and requirements.txt",
                cmd: null,
              },
              {
                id: "step3",
                num: "3",
                title: "Add API key secret (optional)",
                content: 'Space Settings → Secrets → Add secret named TOGETHER_API_KEY or GROQ_API_KEY for better LLM quality',
                cmd: null,
              },
              {
                id: "step4",
                num: "4",
                title: "Clone repo locally (optional)",
                content: "For local testing before deployment:",
                cmd: "git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE\ncd YOUR_SPACE\npip install -r requirements.txt\npython app.py",
              },
            ].map((step) => (
              <div key={step.id} className="deploy-card">
                <div className="flex gap-4">
                  <div className="deploy-num">{step.num}</div>
                  <div className="flex-1">
                    <h3 className="font-bold text-foreground mb-1">{step.title}</h3>
                    <p className="text-sm text-muted-foreground mb-3">{step.content}</p>
                    {step.cmd && (
                      <div className="code-block group">
                        <pre className="text-xs overflow-x-auto">{step.cmd}</pre>
                        <button
                          onClick={() => copyCmd(step.cmd!, step.id)}
                          className="copy-btn"
                        >
                          {copied === step.id ? "✓ Copied!" : "Copy"}
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Rubric coverage */}
        <section>
          <h2 className="section-title">📊 Rubric Coverage</h2>
          <div className="overflow-x-auto rounded-xl border border-border">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-muted">
                  <th className="text-left px-4 py-3 font-semibold text-foreground">Component</th>
                  <th className="text-left px-4 py-3 font-semibold text-foreground">Weight</th>
                  <th className="text-left px-4 py-3 font-semibold text-foreground">Implementation</th>
                </tr>
              </thead>
              <tbody>
                {rubric.map((row, i) => (
                  <tr key={row.component} className={i % 2 === 0 ? "bg-background" : "bg-muted/30"}>
                    <td className="px-4 py-3 font-medium text-foreground">{row.component}</td>
                    <td className="px-4 py-3">
                      <span className="badge-pill bg-primary text-primary-foreground font-bold">
                        {row.weight}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-muted-foreground">{row.coverage}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* Footer */}
        <footer className="text-center py-6 border-t border-border text-sm text-muted-foreground">
          <p>VoiceVerse Sprint · PGDM &amp; PGDM(BM) 25-27 · Maker Lab Assignment</p>
          <p className="mt-1">
            All models open-source: <code className="code-inline">all-MiniLM-L6-v2</code> ·{" "}
            <code className="code-inline">Mixtral-8x7B</code> ·{" "}
            <code className="code-inline">edge-tts</code>
          </p>
        </footer>
      </div>
    </div>
  );
}
