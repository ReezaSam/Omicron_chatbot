# =============================================================
# EmbeddingModel.py
# Omicron UniBot — Embedding Model Module
# Converts text into 384-dimensional vectors using
# sentence-transformers/all-MiniLM-L6-v2
# =============================================================

from pathlib import Path

from sentence_transformers import SentenceTransformer
from typing import List


# NEW
EMBED_MODEL_NAME = str(Path(__file__).resolve().parent.parent / "models" / "rusl_embedding_model")
EMBEDDING_DIM    = 384

print(f"📌 Loading embedding model: {EMBED_MODEL_NAME}")
_model = SentenceTransformer(EMBED_MODEL_NAME)
print(f"✅ Embedding model loaded (dim={EMBEDDING_DIM})")


def embed_text(text: str) -> List[float]:
    """Encode a single string into a 384-dim vector."""
    return _model.encode(text).tolist()


def embed_batch(texts: List[str]) -> List[List[float]]:
    """Encode a list of strings in one batch (faster)."""
    return _model.encode(texts).tolist()


def to_pgvector(vec: List[float]) -> str:
    """Format float list as pgvector-compatible string."""
    return "[" + ",".join(str(float(x)) for x in vec) + "]"


def embed_for_db(text: str) -> str:
    """Convenience: embed text and return pgvector string."""
    return to_pgvector(embed_text(text))


if __name__ == "__main__":
    sample = "What is the minimum GPA to pass a course?"
    vec = embed_text(sample)
    print(f"\n✅ Test input   : {sample}")
    print(f"✅ Vector length: {len(vec)}")
    print(f"✅ First 5 dims : {vec[:5]}")
    print(f"✅ pgvector fmt : {to_pgvector(vec)[:60]}...")