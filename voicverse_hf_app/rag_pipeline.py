"""
RAG Pipeline Module for VoiceVerse
====================================
Document ingestion → chunking (~600 tokens, 100 overlap) →
sentence-transformer embeddings → FAISS index → Top-4 retrieval.
Anti-hallucination: retrieved content is strictly injected into prompts.
"""

import os
import re
import numpy as np
from typing import List, Tuple, Optional


# ── Document Loading ──────────────────────────────────────────────────────────

def load_document(file_path: str) -> str:
    """Load text from PDF, TXT, or DOCX file."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    elif ext == ".pdf":
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except ImportError:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            return "\n".join(page.extract_text() or "" for page in reader.pages)

    elif ext == ".docx":
        from docx import Document
        doc = Document(file_path)
        return "\n".join(para.text for para in doc.paragraphs)

    else:
        raise ValueError(f"Unsupported file type: {ext}. Please upload PDF, TXT, or DOCX.")


# ── Text Chunking ─────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping sentence-aware chunks.
    PRD: ~600 tokens, 100 overlap.
    """
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) < chunk_size:
        return [text] if text else []

    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Overlap: carry last `overlap` chars into next chunk
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + " " + sentence
        else:
            current_chunk += (" " if current_chunk else "") + sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# ── FAISS-Based RAG Pipeline ──────────────────────────────────────────────────

class RAGPipeline:
    """
    RAG pipeline using sentence-transformers + FAISS for fast semantic retrieval.
    Falls back to keyword matching if FAISS/sentence-transformers unavailable.
    """

    def __init__(self):
        self.chunks: List[str] = []
        self.index = None          # FAISS index
        self.embeddings = None     # numpy array (backup)
        self.embed_model = None
        self._load_embed_model()

    def _load_embed_model(self):
        """Lazy-load the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("✅ Embedding model loaded: all-MiniLM-L6-v2")
        except Exception as e:
            print(f"⚠️ sentence-transformers unavailable: {e}")
            self.embed_model = None

    def _build_faiss_index(self, embeddings: np.ndarray):
        """Build or rebuild FAISS flat L2 index."""
        try:
            import faiss
            dim = embeddings.shape[1]
            index = faiss.IndexFlatL2(dim)
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            index.add(embeddings)
            self.index = index
            print(f"✅ FAISS index built with {index.ntotal} vectors (dim={dim})")
        except ImportError:
            print("⚠️ faiss-cpu not available; falling back to numpy cosine similarity")
            self.index = None

    def ingest(self, file_path: str) -> Tuple[int, str]:
        """
        Ingest a document: load → chunk → embed → FAISS index.
        Returns (num_chunks, preview_text).
        """
        raw_text = load_document(file_path)

        if not raw_text or len(raw_text.strip()) < 50:
            raise ValueError("Document appears empty or too short to process.")

        # PRD: ~600 token chunks, 100 overlap
        self.chunks = chunk_text(raw_text, chunk_size=600, overlap=100)

        if not self.chunks:
            raise ValueError("No text chunks could be extracted from the document.")

        if self.embed_model:
            emb = self.embed_model.encode(self.chunks, show_progress_bar=False)
            self.embeddings = emb.astype("float32")
            self._build_faiss_index(self.embeddings.copy())
        else:
            self.embeddings = None
            self.index = None

        preview = raw_text[:500] + "..." if len(raw_text) > 500 else raw_text
        return len(self.chunks), preview

    def retrieve(self, query: str, top_k: int = 4) -> List[str]:
        """
        Retrieve top-k most relevant chunks.
        PRD: Top-4 chunks.
        Falls back to keyword matching if embeddings unavailable.
        """
        if not self.chunks:
            raise ValueError("No document has been ingested yet.")

        if self.embed_model and self.embeddings is not None:
            return self._faiss_retrieve(query, top_k) if self.index else self._cosine_retrieve(query, top_k)
        return self._keyword_retrieve(query, top_k)

    def _faiss_retrieve(self, query: str, top_k: int) -> List[str]:
        """FAISS-based cosine similarity retrieval."""
        import faiss
        query_emb = self.embed_model.encode([query]).astype("float32")
        faiss.normalize_L2(query_emb)
        distances, indices = self.index.search(query_emb, min(top_k, len(self.chunks)))
        return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]

    def _cosine_retrieve(self, query: str, top_k: int) -> List[str]:
        """Numpy cosine similarity fallback."""
        query_emb = self.embed_model.encode([query])
        norm_c = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norm_q = np.linalg.norm(query_emb, axis=1, keepdims=True)
        norm_c = np.where(norm_c == 0, 1e-8, norm_c)
        norm_q = np.where(norm_q == 0, 1e-8, norm_q)
        scores = (query_emb / norm_q @ (self.embeddings / norm_c).T)[0]
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [self.chunks[i] for i in top_idx]

    def _keyword_retrieve(self, query: str, top_k: int) -> List[str]:
        """Simple keyword overlap fallback."""
        query_words = set(query.lower().split())
        scored = sorted(
            self.chunks,
            key=lambda c: len(query_words & set(c.lower().split())),
            reverse=True,
        )
        return scored[:top_k]

    def get_full_context(self, max_chars: int = 4000) -> str:
        context = ""
        for chunk in self.chunks:
            if len(context) + len(chunk) > max_chars:
                break
            context += chunk + "\n\n"
        return context.strip()

    def retrieve_for_style(self, style: str, top_k: int = 4) -> str:
        """Retrieve context tailored to the output style (top-4 per PRD)."""
        style_queries = {
            "podcast": "main ideas key points interesting facts discussion topics",
            "debate": "arguments evidence pros cons perspectives viewpoints",
            "storytelling": "narrative events characters journey transformation",
            "news": "facts recent developments important information summary",
            "lecture": "concepts definitions explanations examples learning objectives",
        }
        query = style_queries.get(style, "main ideas and key points")
        chunks = self.retrieve(query, top_k=top_k)
        return "\n\n".join(chunks)
