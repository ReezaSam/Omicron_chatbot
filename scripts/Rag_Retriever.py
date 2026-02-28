# =============================================================
# Rag_Retriever.py
# Omicron UniBot — RAG Retrieval Module (Hybrid: Vector + FTS)
# - Supports department/year/semester query augmentation
# - Optional section filtering by department keywords
# - Hybrid retrieval: pgvector + Postgres Full-Text Search (FTS)
# - Safe fallback to vector-only if FTS column/index not created
# =============================================================

import os
import re
import psycopg2
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

from EmbeddingModel import embed_for_db, EMBEDDING_DIM

# ─────────────────────────────────────────────
# DATABASE CONFIG (ENV SAFE)
# ─────────────────────────────────────────────
DB_NAME     = os.getenv("DB_NAME", "omicron_db")
DB_USER     = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = os.getenv("DB_PORT", "5432")

DOC_ID = int(os.getenv("DOC_ID", "1"))

# ─────────────────────────────────────────────
# RETRIEVAL CONFIG
# ─────────────────────────────────────────────
TOP_K                    = int(os.getenv("RAG_TOP_K", "15"))
DISTANCE_THRESHOLD       = float(os.getenv("RAG_DISTANCE_THRESHOLD", "0.65"))
ADMIN_DISTANCE_THRESHOLD = float(os.getenv("RAG_ADMIN_DISTANCE_THRESHOLD", "0.55"))
MIN_KEYWORD_OVERLAP      = int(os.getenv("RAG_MIN_KEYWORD_OVERLAP", "1"))

# ─────────────────────────────────────────────
# INTENT / FILTER PATTERNS
# ─────────────────────────────────────────────
ACADEMIC_HINTS = {
    "gpa", "grade", "grading", "course", "subject", "semester", "year",
    "credit", "exam", "registration", "results", "repeat", "eligibility",
    "attendance", "programme", "program", "degree", "prerequisite", "corequisite",
    "appeal", "appeals", "suspension", "transfer", "withdraw", "defer",
    "deferment", "medical", "special consideration", "re-evaluation",
    "remark", "disciplinary", "timetable", "syllabus",
    # extra common academic words
    "class", "division", "honours", "honors", "lower", "upper", "minimum", "requirement",
    "library", "librarian", "idc", "fdn"
}

ADMIN_POLICY_PAT = re.compile(
    r"\b(appeal|appeals|academic\s+appeal|"
    r"suspend|suspension|disciplinary|misconduct|"
    r"re[-\s]?evaluation|recheck|remark|"
    r"transfer|change\s+(my\s+)?programme|change\s+(my\s+)?degree|"
    r"withdraw|defer|deferment|special\s+consideration|medical|"
    r"extension|deadline|registration\s+issue)\b",
    re.I
)

STOPWORDS = {
    "the","a","an","is","are","was","were","to","of","in","on",
    "for","and","or","with","by","i","we","you","they","he","she",
    "it","this","that","what","which","when","where","how","can",
    "could","should","do","does","did","me","my","your","our","their"
}

# department -> keywords likely appearing in section titles
DEPT_TO_SECTION_KEYWORDS: Dict[str, List[str]] = {
    "Computing": ["comput", "ict", "it", "software", "cs"],
    "Physical Science": ["physical", "physics", "math", "mathemat", "stat"],
    "Biological Science": ["biolog", "bio", "zool", "botan", "microbio"],
    "Health Promotion": ["health", "promotion", "nursing", "public health"],
    "Chemical Sciences": ["chem", "chemical"]
}

# ─────────────────────────────────────────────
# DATABASE HELPERS
# ─────────────────────────────────────────────

def get_connection():
    if DB_PASSWORD == "":
        print("⚠️ DB_PASSWORD is empty. Set DB_PASSWORD in your .env file.")
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

def save_query(cur, username: str, question: str) -> int:
    cur.execute("""
        INSERT INTO queries (username, query_text, chatbot_response, is_escalated)
        VALUES (%s, %s, NULL, FALSE)
        RETURNING query_id;
    """, (username, question))
    return cur.fetchone()[0]

def update_response(cur, query_id: int, response: str):
    cur.execute("UPDATE queries SET chatbot_response = %s WHERE query_id = %s;", (response, query_id))

def create_escalation(cur, query_id: int):
    cur.execute("UPDATE queries SET is_escalated = TRUE WHERE query_id = %s;", (query_id,))
    cur.execute("""
        INSERT INTO escalations (query_id, status, escalated_at)
        VALUES (%s, 'pending', NOW())
        ON CONFLICT (query_id) DO NOTHING;
    """, (query_id,))

# ─────────────────────────────────────────────
# TEXT UTILS
# ─────────────────────────────────────────────

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().strip())

def _tokenize(text: str) -> List[str]:
    text = _normalize(text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t not in STOPWORDS and len(t) > 2]

def keyword_overlap(question: str, chunk: str) -> int:
    return len(set(_tokenize(question)).intersection(set(_tokenize(chunk))))

# ─────────────────────────────────────────────
# ACADEMIC INTENT CHECKS
# ─────────────────────────────────────────────

def is_academic(question: str) -> bool:
    """
    Soft check: returns True if it looks academic, but do NOT use this to block chat.
    """
    m = _normalize(question)
    if any(k in m for k in ACADEMIC_HINTS):
        return True
    if ADMIN_POLICY_PAT.search(question):
        return True
    if re.search(r"\b(degree|programme|program|faculty|department|semester|exam|gpa|grade|course|subject|timetable)\b", m):
        return True
    return False

def is_admin_policy(question: str) -> bool:
    return bool(ADMIN_POLICY_PAT.search(question))

# ─────────────────────────────────────────────
# QUERY AUGMENTATION + FILTER
# ─────────────────────────────────────────────

def augment_query(question: str,
                  department: Optional[str] = None,
                  year: Optional[str] = None,
                  semester: Optional[str] = None) -> str:
    parts = [question.strip()]
    if department:
        parts.append(f"Department: {department}")
    if year:
        parts.append(f"Year: {year}")
    if semester:
        parts.append(f"Semester: {semester}")
    return " | ".join(parts)

def dept_section_keywords(department: Optional[str]) -> List[str]:
    if not department:
        return []
    if department in DEPT_TO_SECTION_KEYWORDS:
        return DEPT_TO_SECTION_KEYWORDS[department]
    return [department.lower()]

# ─────────────────────────────────────────────
# HYBRID RETRIEVAL (VECTOR + FTS)
# Requires (recommended) DB setup:
#   ALTER TABLE kb_chunks ADD COLUMN IF NOT EXISTS chunk_tsv tsvector;
#   UPDATE kb_chunks SET chunk_tsv = to_tsvector('english', coalesce(chunk_text,''));
#   CREATE INDEX IF NOT EXISTS kb_chunks_tsv_idx ON kb_chunks USING GIN (chunk_tsv);
# ─────────────────────────────────────────────

def _merge_rank(rows_a: List[Tuple[str, float]], rows_b: List[Tuple[str, float]], top_k: int) -> List[Tuple[str, float]]:
    best = {}
    for txt, dist in (rows_a + rows_b):
        d = float(dist)
        if txt not in best or d < best[txt]:
            best[txt] = d
    merged = sorted(best.items(), key=lambda x: x[1])
    return merged[:top_k]

def retrieve_chunks(cur,
                    question: str,
                    department: Optional[str] = None,
                    year: Optional[str] = None,
                    semester: Optional[str] = None,
                    top_k: int = TOP_K) -> List[Tuple[str, float]]:

    augmented = augment_query(question, department, year, semester)
    q_vec = embed_for_db(augmented)
    section_keys = dept_section_keywords(department)

    # ---- vector query (optional section filter)
    def vector_query(filtered: bool) -> List[Tuple[str, float]]:
        if filtered and section_keys:
            conds = " OR ".join(["section ILIKE %s"] * len(section_keys))
            sql = f"""
                SELECT chunk_text,
                       (embedding <=> %s::vector({EMBEDDING_DIM})) AS distance
                FROM kb_chunks
                WHERE doc_id = %s
                  AND ({conds})
                ORDER BY embedding <=> %s::vector({EMBEDDING_DIM})
                LIMIT %s;
            """
            params = [q_vec, DOC_ID] + [f"%{k}%" for k in section_keys] + [q_vec, top_k]
            cur.execute(sql, params)
            return cur.fetchall()

        sql = f"""
            SELECT chunk_text,
                   (embedding <=> %s::vector({EMBEDDING_DIM})) AS distance
            FROM kb_chunks
            WHERE doc_id = %s
            ORDER BY embedding <=> %s::vector({EMBEDDING_DIM})
            LIMIT %s;
        """
        cur.execute(sql, (q_vec, DOC_ID, q_vec, top_k))
        return cur.fetchall()

    # ---- FTS query (ranked by ts_rank); fallback safe if chunk_tsv not present
    def fts_query(filtered: bool) -> List[Tuple[str, float]]:
        if filtered and section_keys:
            conds = " OR ".join(["section ILIKE %s"] * len(section_keys))
            sql = f"""
                SELECT chunk_text,
                       (embedding <=> %s::vector({EMBEDDING_DIM})) AS distance
                FROM kb_chunks
                WHERE doc_id = %s
                  AND ({conds})
                  AND chunk_tsv @@ plainto_tsquery('english', %s)
                ORDER BY ts_rank(chunk_tsv, plainto_tsquery('english', %s)) DESC
                LIMIT %s;
            """
            params = [q_vec, DOC_ID] + [f"%{k}%" for k in section_keys] + [question, question, top_k]
            cur.execute(sql, params)
            return cur.fetchall()

        sql = f"""
            SELECT chunk_text,
                   (embedding <=> %s::vector({EMBEDDING_DIM})) AS distance
            FROM kb_chunks
            WHERE doc_id = %s
              AND chunk_tsv @@ plainto_tsquery('english', %s)
            ORDER BY ts_rank(chunk_tsv, plainto_tsquery('english', %s)) DESC
            LIMIT %s;
        """
        cur.execute(sql, (q_vec, DOC_ID, question, question, top_k))
        return cur.fetchall()

    # 1) try dept-filtered vector
    vec_filtered = vector_query(filtered=True)

    # If good dept results, try dept FTS and merge
    if section_keys and len(vec_filtered) >= max(2, top_k // 2):
        try:
            fts_filtered = fts_query(filtered=True)
            return _merge_rank(vec_filtered, fts_filtered, top_k)
        except Exception:
            return vec_filtered

    # 2) global vector + global FTS merge
    vec_global = vector_query(filtered=False)
    try:
        fts_global = fts_query(filtered=False)
        return _merge_rank(vec_global, fts_global, top_k)
    except Exception:
        return vec_global

# ─────────────────────────────────────────────
# CONFIDENCE + ESCALATION DECISION
# ─────────────────────────────────────────────

def should_escalate(question: str, chunks: List[Tuple[str, float]]) -> Tuple[bool, Optional[float], Optional[int]]:
    if not chunks:
        return True, None, None

    best_text, best_dist = chunks[0][0], float(chunks[0][1])
    overlap = keyword_overlap(question, best_text)
    admin   = is_admin_policy(question)

    if admin and best_dist > ADMIN_DISTANCE_THRESHOLD:
        return True, best_dist, overlap

    if best_dist > DISTANCE_THRESHOLD and overlap < MIN_KEYWORD_OVERLAP:
        return True, best_dist, overlap

    return False, best_dist, overlap

def build_context(chunks: List[Tuple[str, float]]) -> str:
    return "\n\n---\n\n".join(chunk_text.strip() for chunk_text, _ in chunks)