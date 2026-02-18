"""
RAG Pipeline Module for VoiceVerse Sprint
Handles document ingestion, chunking, embedding, and retrieval.
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
            # Fallback: pypdf
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

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks for better retrieval coverage.
    Uses sentence-aware splitting to avoid cutting mid-sentence.
    """
    # Clean up whitespace
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) < chunk_size:
        return [text] if text else []

    # Split on sentence boundaries first
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_chunk = ""
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        if current_len + sentence_len > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Keep overlap: take last `overlap` chars
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + " " + sentence
            current_len = len(current_chunk)
        else:
            current_chunk += (" " if current_chunk else "") + sentence
            current_len += sentence_len + 1

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# ── Embedding & Retrieval ─────────────────────────────────────────────────────

class RAGPipeline:
    """
    Simple but effective RAG pipeline using sentence-transformers for embeddings
    and cosine similarity for retrieval.
    """

    def __init__(self):
        self.chunks: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the embedding model (lazy loaded)."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            print("✅ Embedding model loaded: all-MiniLM-L6-v2")
        except Exception as e:
            print(f"⚠️ Could not load sentence-transformers: {e}")
            self.model = None

    def ingest(self, file_path: str) -> Tuple[int, str]:
        """
        Ingest a document: load → chunk → embed.
        Returns (num_chunks, preview_text).
        """
        raw_text = load_document(file_path)

        if not raw_text or len(raw_text.strip()) < 50:
            raise ValueError("Document appears empty or too short to process.")

        self.chunks = chunk_text(raw_text, chunk_size=500, overlap=100)

        if not self.chunks:
            raise ValueError("No text chunks could be extracted from the document.")

        # Embed chunks
        if self.model:
            self.embeddings = self.model.encode(self.chunks, show_progress_bar=False)
        else:
            # Fallback: TF-IDF style keyword matching (no ML)
            self.embeddings = None

        preview = raw_text[:500] + "..." if len(raw_text) > 500 else raw_text
        return len(self.chunks), preview

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve the top-k most relevant chunks for a query.
        Falls back to keyword matching if embeddings unavailable.
        """
        if not self.chunks:
            raise ValueError("No document has been ingested yet. Please upload a document first.")

        if self.model and self.embeddings is not None:
            return self._semantic_retrieve(query, top_k)
        else:
            return self._keyword_retrieve(query, top_k)

    def _semantic_retrieve(self, query: str, top_k: int) -> List[str]:
        """Retrieve using cosine similarity on embeddings."""
        query_emb = self.model.encode([query])

        # Cosine similarity
        norms_chunks = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms_query = np.linalg.norm(query_emb, axis=1, keepdims=True)

        # Avoid division by zero
        norms_chunks = np.where(norms_chunks == 0, 1e-8, norms_chunks)
        norms_query = np.where(norms_query == 0, 1e-8, norms_query)

        normalized_chunks = self.embeddings / norms_chunks
        normalized_query = query_emb / norms_query

        scores = (normalized_query @ normalized_chunks.T)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [self.chunks[i] for i in top_indices]

    def _keyword_retrieve(self, query: str, top_k: int) -> List[str]:
        """Simple keyword-based fallback retrieval."""
        query_words = set(query.lower().split())

        scored = []
        for chunk in self.chunks:
            chunk_words = set(chunk.lower().split())
            score = len(query_words & chunk_words)
            scored.append((score, chunk))

        scored.sort(reverse=True)
        return [chunk for _, chunk in scored[:top_k]]

    def get_full_context(self, max_chars: int = 4000) -> str:
        """Get concatenated chunks up to max_chars for script generation."""
        context = ""
        for chunk in self.chunks:
            if len(context) + len(chunk) > max_chars:
                break
            context += chunk + "\n\n"
        return context.strip()

    def retrieve_for_style(self, style: str, top_k: int = 6) -> str:
        """Retrieve context tailored to the output style."""
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
