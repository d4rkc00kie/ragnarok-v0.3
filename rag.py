"""Simple local RAG with Ollama embeddings + ChromaDB."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional, Tuple

import ollama
from chromadb import PersistentClient
from chromadb.config import Settings

from config import Config


TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".py", ".js", ".ts", ".jsx", ".tsx",
    ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".csv",
    ".html", ".css", ".scss", ".rs", ".go", ".java", ".c", ".cpp",
    ".h", ".hpp", ".sh", ".bash", ".zsh", ".sql", ".log", ".rst",
}


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> List[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, size - overlap)
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def read_file(path: Path) -> Optional[str]:
    try:
        if path.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(path))
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception:
                return None
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


class RAGStore:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cfg.chroma_path.mkdir(parents=True, exist_ok=True)
        self.client = PersistentClient(
            path=str(self.cfg.chroma_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="docs",
            metadata={"hnsw:space": "cosine"},
        )

    def _embed(self, texts: List[str] | str) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        # Prefer modern embed API; fall back for older ollama-python
        try:
            resp = ollama.embed(model=self.cfg.embed_model, input=texts)
        except TypeError:
            if len(texts) == 1:
                resp = ollama.embeddings(model=self.cfg.embed_model, prompt=texts[0])
            else:
                return [
                    ollama.embeddings(model=self.cfg.embed_model, prompt=t)["embedding"]
                    for t in texts
                ]
        if isinstance(resp, dict):
            if "embeddings" in resp:
                return resp["embeddings"]
            if "embedding" in resp:
                return [resp["embedding"]]
        if hasattr(resp, "embeddings"):
            return list(resp.embeddings)
        if hasattr(resp, "embedding"):
            return [list(resp.embedding)]
        return []

    def add_file(self, path: Path) -> int:
        path = path.resolve()
        if not path.is_file():
            return 0
        text = read_file(path)
        if not text or not text.strip():
            return 0
        chunks = chunk_text(text, self.cfg.chunk_size, self.cfg.chunk_overlap)
        if not chunks:
            return 0

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[dict] = []

        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.sha256(f"{path}:{i}:{chunk[:50]}".encode()).hexdigest()[:16]
            ids.append(f"{path.name}-{i}-{chunk_id}")
            documents.append(chunk)
            metadatas.append({"source": str(path), "chunk": i, "filename": path.name})

        embeddings = self._embed(documents)
        # delete existing chunks for this file first
        try:
            existing = self.collection.get(where={"source": str(path)})
            if existing and existing["ids"]:
                self.collection.delete(ids=existing["ids"])
        except Exception:
            pass

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        return len(chunks)

    def add_directory(self, directory: Path, recursive: bool = True) -> Tuple[int, int]:
        directory = directory.resolve()
        if not directory.is_dir():
            return 0, 0
        files_added = 0
        chunks_total = 0
        pattern = "**/*" if recursive else "*"
        for path in directory.glob(pattern):
            if path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS | {".pdf"}:
                n = self.add_file(path)
                if n:
                    files_added += 1
                    chunks_total += n
        return files_added, chunks_total

    def query(self, question: str, top_k: Optional[int] = None) -> List[dict]:
        top_k = top_k or self.cfg.top_k
        if self.collection.count() == 0:
            return []
        emb = self._embed(question)[0]
        results = self.collection.query(
            query_embeddings=[emb],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            hits.append({
                "text": doc,
                "source": meta.get("source", "?"),
                "filename": meta.get("filename", "?"),
                "distance": dist,
            })
        return hits

    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> None:
        try:
            self.client.delete_collection("docs")
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name="docs",
            metadata={"hnsw:space": "cosine"},
        )

    def list_sources(self) -> List[str]:
        if self.collection.count() == 0:
            return []
        data = self.collection.get(include=["metadatas"])
        sources = {m.get("source") for m in data["metadatas"] if m}
        return sorted(s for s in sources if s)
