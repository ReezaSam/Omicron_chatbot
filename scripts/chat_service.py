# =============================================================
# chat_service.py (Implementation 2: ML Router + Clarification + Hybrid Retrieval)
# - ML router predicts: answer / clarify / escalate
# - If vague, asks department + year + semester
# - Pass filters into retrieve_chunks() for better retrieval
# - Uses deeper retrieval for fact-heavy questions, but sends only top context chunks to Groq
# =============================================================

import os
import re
from typing import Optional, Dict
from pathlib import Path

import joblib
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# Load .env from project root (works whether this file is in root or /scripts)
HERE = Path(__file__).resolve()
env_path_1 = HERE.parent / ".env"
env_path_2 = HERE.parent.parent / ".env"
load_dotenv(dotenv_path=env_path_1 if env_path_1.exists() else env_path_2)

from Rag_Retriever import (
    get_connection,
    save_query,
    update_response,
    create_escalation,
    retrieve_chunks,
    should_escalate,
    build_context,
)

# ─────────────────────────────────────────────
# GROQ LLM CONFIG (ENV SAFE)
# ─────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
TEMPERATURE  = float(os.getenv("GROQ_TEMPERATURE", "0.3"))
MAX_TOKENS   = int(os.getenv("GROQ_MAX_TOKENS", "350"))

STUDENT_USERNAME = os.getenv("OMICRON_USERNAME", "stu01")

# Retrieval tuning
RAG_RETRIEVE_K = int(os.getenv("RAG_RETRIEVE_K", "15"))  # how deep we retrieve
RAG_CONTEXT_K  = int(os.getenv("RAG_CONTEXT_K", "5"))    # how many chunks we pass to Groq

# Fact-heavy detector (numbers/IDs/where/when/how many)
FACTY_PAT = re.compile(r"(\d{2,4}|/|where|when|how many|gazette|district|located|established)", re.I)

# ─────────────────────────────────────────────
# GREETING DETECTOR  ← NEW FIX
# ─────────────────────────────────────────────
GREETINGS = {
    "hello", "hi", "hey", "good morning", "good afternoon",
    "good evening", "howdy", "greetings", "sup", "what's up",
    "hiya", "hi there", "hello there", "good day"
}

def is_greeting(msg: str) -> bool:
    m = msg.strip().lower()
    if m in GREETINGS:
        return True
    # also catch "hello!" "hi," etc with punctuation
    m_clean = re.sub(r"[^a-z\s]", "", m).strip()
    return m_clean in GREETINGS

# ─────────────────────────────────────────────
# ML ROUTER MODEL
# models/router_classifier.joblib should exist
# ─────────────────────────────────────────────
def _find_router_model_path() -> Path:
    candidates = [
        HERE.parent / "models" / "router_classifier.joblib",
        HERE.parent.parent / "models" / "router_classifier.joblib",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]

ROUTER_MODEL_PATH = _find_router_model_path()
_router_bundle = None
_router_embedder = None

def _load_router():
    global _router_bundle, _router_embedder
    if _router_bundle is None:
        if not ROUTER_MODEL_PATH.exists():
            raise FileNotFoundError(f"Router model not found: {ROUTER_MODEL_PATH}")
        _router_bundle = joblib.load(ROUTER_MODEL_PATH)

    if _router_embedder is None:
        embed_name = _router_bundle.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")
        _router_embedder = SentenceTransformer(embed_name)

def predict_action(question: str) -> str:
    """
    Returns one of: answer / clarify / escalate
    """
    _load_router()
    clf = _router_bundle["classifier"]
    vec = _router_embedder.encode([question])[0]
    pred = clf.predict([vec])[0]
    return str(pred).lower().strip()

# ─────────────────────────────────────────────
# CLARIFICATION (STRUCTURED)
# ─────────────────────────────────────────────
DEPARTMENTS = {
    "computing": "Computing",
    "physical": "Physical Science",
    "physical science": "Physical Science",
    "biological": "Biological Science",
    "biological science": "Biological Science",
    "health": "Health Promotion",
    "health promotion": "Health Promotion",
    "chemical": "Chemical Sciences",
    "chemical sciences": "Chemical Sciences",
    "chemistry": "Chemical Sciences",
    "ict": "Computing",
    "it": "Computing",
    "cs": "Computing",
}

YEAR_PAT = re.compile(r"\b(1st|2nd|3rd|4th)\s*year\b|\byear\s*(1|2|3|4)\b", re.I)
SEM_PAT  = re.compile(r"\bsemester\s*(1|2)\b|\bsem\s*(1|2)\b", re.I)

NEEDS_CONTEXT_PAT = re.compile(
    r"\b(subjects?|courses?|modules?|units?|timetable|schedule|"
    r"what should i study|list.*subjects?|list.*courses?)\b",
    re.I
)
HAS_CONTEXT_PAT = re.compile(
    r"\b(first|second|third|fourth|1st|2nd|3rd|4th|"
    r"year\s*\d|\d\s*year|semester\s*\d|\d\s*semester|"
    r"computing|physical|biological|health|chemical|ict|it|cs)\b",
    re.I
)

# in-memory state per user
clarification_state: Dict[str, Dict] = {}

def detect_department(text: str) -> Optional[str]:
    t = text.lower()
    for k, v in DEPARTMENTS.items():
        if re.search(rf"\b{re.escape(k)}\b", t):
            return v
    return None

def detect_year(text: str) -> Optional[str]:
    m = YEAR_PAT.search(text)
    if not m:
        return None
    g = m.group(0).lower()
    if "1" in g: return "1"
    if "2" in g: return "2"
    if "3" in g: return "3"
    if "4" in g: return "4"
    return None

def detect_semester(text: str) -> Optional[str]:
    m = SEM_PAT.search(text)
    if not m:
        return None
    g = m.group(0)
    if "1" in g: return "1"
    if "2" in g: return "2"
    return None

def needs_clarification(question: str) -> bool:
    if HAS_CONTEXT_PAT.search(question):
        return False
    return bool(NEEDS_CONTEXT_PAT.search(question))

def clarification_prompt(missing_dept: bool, missing_year: bool, missing_sem: bool) -> str:
    lines = ["To give the correct answer, please tell me:"]
    if missing_dept:
        lines.append("1) Your department (Computing / Physical Science / Biological Science / Health Promotion / Chemical Sciences)")
    if missing_year:
        lines.append("2) Your year (1st / 2nd / 3rd / 4th)")
    if missing_sem:
        lines.append("3) Your semester (1 or 2)")
    return "\n".join(lines)

# ─────────────────────────────────────────────
# GROQ ANSWER
# ─────────────────────────────────────────────
def _generate_answer(question: str, context: str) -> str:
    if not GROQ_API_KEY:
        return ""

    client = Groq(api_key=GROQ_API_KEY)

    prompt = (
        "You are Omicron Bot, an academic advising assistant for Rajarata University.\n"
        "Answer ONLY using the provided context.\n"
        "If the answer is not clearly supported by the context, reply exactly: NOT_FOUND_IN_HANDBOOK\n"
        "If it is a yes/no question, answer only Yes/No + one supporting sentence from the context.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}"
    )

    try:
        res = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print("Groq error:", e)
        return ""

# ─────────────────────────────────────────────
# MAIN CHAT FUNCTION
# ─────────────────────────────────────────────
def process_message(username: str, message: str, conn, cur) -> str:
    msg = (message or "").strip()
    if not msg:
        return "Please type your question."

    # ── GREETING CHECK ← NEW FIX ──────────────
    if is_greeting(msg):
        return (
            "Hello! 👋 I'm Omicron Bot, your academic assistant for Rajarata University.\n"
            "I can help you with courses, GPA, exams, registration, and university policies.\n"
            "How can I help you today?"
        )

    # ── NON-ACADEMIC CHECK ────────────────────
    # (greetings already handled above, so this catches other non-academic messages)
    # Goodbye check
    GOODBYES = {"bye", "goodbye", "see you", "see ya", "take care", "cya"}
    if msg.lower().strip() in GOODBYES:
        return "Goodbye! 👋 Feel free to come back anytime you have academic questions!"

    # Thanks check
    THANKS = {"thanks", "thank you", "ok", "okay", "cool", "nice", "great", "perfect"}
    if msg.lower().strip() in THANKS:
        return "You're welcome! Feel free to ask any academic questions anytime. 😊"

    # Emotional message check
    EMOTIONAL_PAT = re.compile(
        r"\b(stress|stressed|anxious|anxiety|worried|worry|sad|depressed|"
        r"overwhelmed|tired|exhausted|nervous|scared|afraid|feel|feeling|"
        r"upset|frustrated|confused|lost|helpless)\b", re.I
    )
    if EMOTIONAL_PAT.search(msg):
        return (
            "I'm sorry to hear you're feeling that way. 💙\n"
            "University life can be tough sometimes. Please consider talking to a "
            "counselor or a trusted person.\n"
            "If you have any academic questions I can help with, I'm here for you!"
        )

    # Handle pending clarification replies
    if username in clarification_state and clarification_state[username].get("waiting_for") == "clarification":
        st = clarification_state[username]

        # try to fill missing
        if st.get("department") is None:
            st["department"] = detect_department(msg)
        if st.get("year") is None:
            st["year"] = detect_year(msg)
        if st.get("semester") is None:
            st["semester"] = detect_semester(msg)

        missing_dept = st.get("department") is None
        missing_year = st.get("year") is None
        missing_sem  = st.get("semester") is None

        # still missing -> ask again
        if missing_dept or missing_year or missing_sem:
            reply = clarification_prompt(missing_dept, missing_year, missing_sem)
            qid = save_query(cur, username, st["original_question"])
            update_response(cur, qid, reply)
            conn.commit()
            return reply

        # all collected -> continue with original question
        original_q = st["original_question"]
        filters = {"department": st["department"], "year": st["year"], "semester": st["semester"]}
        clarification_state.pop(username, None)
        msg = original_q  # continue processing with original question using filters
    else:
        filters = {
            "department": detect_department(msg),
            "year": detect_year(msg),
            "semester": detect_semester(msg)
        }

    # Router predicts action
    try:
        action = predict_action(msg)
    except Exception as e:
        print("Router load/predict error:", e)
        action = "answer"

    # If vague AND router says clarify OR heuristic says clarification needed -> ask
    if (action == "clarify" or needs_clarification(msg)) and not (filters["department"] and filters["year"] and filters["semester"]):
        clarification_state[username] = {
            "waiting_for": "clarification",
            "original_question": msg,
            "department": filters["department"],
            "year": filters["year"],
            "semester": filters["semester"]
        }
        st = clarification_state[username]
        reply = clarification_prompt(
            missing_dept=(st["department"] is None),
            missing_year=(st["year"] is None),
            missing_sem=(st["semester"] is None),
        )
        qid = save_query(cur, username, msg)
        update_response(cur, qid, reply)
        conn.commit()
        return reply

    # Save query
    qid = save_query(cur, username, msg)

    # If router says escalate immediately
    if action == "escalate":
        reply = "This seems like a case that needs an academic advisor. Escalating for a reliable answer."
        create_escalation(cur, qid)
        update_response(cur, qid, reply)
        conn.commit()
        return reply

    # Retrieval (deeper for fact-heavy questions, but pass only top context chunks)
    retrieve_k = RAG_RETRIEVE_K if FACTY_PAT.search(msg) else max(RAG_CONTEXT_K, 5)

    chunks_all = retrieve_chunks(
        cur,
        msg,
        department=filters.get("department"),
        year=filters.get("year"),
        semester=filters.get("semester"),
        top_k=retrieve_k
    )
    chunks = chunks_all[:RAG_CONTEXT_K]

    # Confidence check (uses best chunk)
    escalate, best_dist, overlap = should_escalate(msg, chunks)

    if escalate:
        reply = (
            "I couldn't find a confident answer in the handbook for your question.\n"
            "I have escalated this to an academic advisor who will respond within 5 working days."
        )
        create_escalation(cur, qid)
        update_response(cur, qid, reply)
        conn.commit()
        return reply

    # Generate answer using Groq
    context = build_context(chunks)
    answer = _generate_answer(msg, context)

    if not answer or "NOT_FOUND_IN_HANDBOOK" in answer.upper():
        reply = "I couldn't find this clearly in the handbook. Escalating to an academic advisor for a reliable answer."
        create_escalation(cur, qid)
        update_response(cur, qid, reply)
        conn.commit()
        return reply

    update_response(cur, qid, answer)
    conn.commit()
    return answer


if __name__ == "__main__":
    print("Omicron UniBot — Chat (ML Router + Clarification + Hybrid Retrieval)")
    print("Type 'exit' to quit.\n")

    conn = get_connection()
    cur = conn.cursor()

    try:
        while True:
            user_msg = input("You: ").strip()
            if user_msg.lower() in {"exit", "quit", "q"}:
                print("Bot: Goodbye!")
                break

            bot_reply = process_message(STUDENT_USERNAME, user_msg, conn, cur)
            print("Bot:", bot_reply)

    finally:
        cur.close()
        conn.close()