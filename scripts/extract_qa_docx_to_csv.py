from pathlib import Path
import re
import pandas as pd
from docx import Document

DOCX_PATH = Path(__file__).resolve().parent.parent / "QA_pairs.docx"
OUT_CSV   = Path(__file__).resolve().parent.parent / "data" / "answer_questions_500.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

NUM_LINE_PAT = re.compile(r"^\s*(\d+)\s*[\.\)]\s*(.+?)\s*$")

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def extract_question(body: str):
    body = norm(body)
    if "?" not in body:
        return None
    q = body.split("?", 1)[0].strip()
    if len(q) < 8:
        return None
    return q + "?"

def main():
    if not DOCX_PATH.exists():
        raise FileNotFoundError(f"❌ DOCX not found: {DOCX_PATH}")

    doc = Document(str(DOCX_PATH))
    paras = [norm(p.text) for p in doc.paragraphs if p.text and p.text.strip()]

    # Merge continuation lines: if we see "12. ..." then next paragraph doesn't start with number,
    # we treat it as continuation of the same item.
    merged_items = []  # list of (num, text)
    current_num = None
    current_text = ""

    for line in paras:
        m = NUM_LINE_PAT.match(line)
        if m:
            # save previous
            if current_num is not None:
                merged_items.append((current_num, current_text))
            current_num = int(m.group(1))
            current_text = m.group(2).strip()
        else:
            # continuation
            if current_num is not None:
                current_text = (current_text + " " + line).strip()

    # save last
    if current_num is not None:
        merged_items.append((current_num, current_text))

    questions = []
    for num, body in merged_items:
        q = extract_question(body)
        if q:
            questions.append(q)

    # Fallback: also try any paragraph containing '?'
    for line in paras:
        if "?" in line:
            q2 = extract_question(line)
            if q2:
                questions.append(q2)

    # Deduplicate case-insensitive
    seen = set()
    dedup = []
    for q in questions:
        k = q.lower()
        if k not in seen:
            seen.add(k)
            dedup.append(q)

    df = pd.DataFrame({"question_text": dedup, "label": ["answer"] * len(dedup)})
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    print(f"✅ Extracted {len(df)} questions")
    print(f"✅ Saved: {OUT_CSV}")

if __name__ == "__main__":
    main()