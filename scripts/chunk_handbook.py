import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# =============================
# PATHS
# =============================
root = Path(__file__).resolve().parent.parent
in_path = root / "clean" / "faculty_handbook_cleaned.txt"

out_dir = root / "chunks"
out_dir.mkdir(parents=True, exist_ok=True)

out_path = out_dir / "faculty_handbook_chunks_metadata.jsonl"

# =============================
# CONFIG
# =============================
SOURCE_NAME = "faculty_handbook"

MAX_CHARS = 1600
END_TABLE_BLANKS = 2  # blank lines that end a table block

# =============================
# REGEX
# =============================
PAGE_MARKER_RE = re.compile(r"^\s*===\s*PAGE\s+(\d+)\s*===\s*$", re.I)

TABLE_MARKER_RE = re.compile(r"^\s*Table\s+\d+(\.\d+)+.*$", re.I)
TABLE_HEADER_CUES = [
    ("course code", "course title"),
    ("field of study", "meaning"),
    ("grade", "grade point"),
    ("credit", "course title"),
    ("prerequisite", "corequisite"),
]

LIST_RE = re.compile(
    r"^\s*(?:[-•*]|\(?[ivx]+\)?\.|\(?\d+\)?\.|\d+\)|[a-zA-Z]\.)\s+",
    re.I,
)
URL_RE = re.compile(r"^\s*https?://\S+\s*$", re.I)
MULTI_BLANK_RE = re.compile(r"\n{3,}")

# headings
NUM_HEADING_RE = re.compile(r"^\s*\d+(\.\d+)*\s+\S+")
ALLCAPS_HEADING_RE = re.compile(r"^\s*[A-Z][A-Z0-9 \-/,:&()]{5,}\s*$")


def looks_like_table_header(line: str) -> bool:
    l = line.lower()
    return any(a in l and b in l for a, b in TABLE_HEADER_CUES)


def is_table_start(line: str) -> bool:
    s = line.strip()
    return bool(TABLE_MARKER_RE.match(s)) or looks_like_table_header(s)


def is_list_line(line: str) -> bool:
    return bool(LIST_RE.match(line.strip()))


def is_probable_heading(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if PAGE_MARKER_RE.match(s):
        return False
    if is_table_start(s):
        return False
    if URL_RE.match(s):
        return False

    # numbered heading (any case after)
    if NUM_HEADING_RE.match(s):
        return True

    # ALL CAPS heading
    if ALLCAPS_HEADING_RE.match(s):
        return True

    # title-case-ish short headings
    words = re.findall(r"[A-Za-z]+", s)
    if 2 <= len(words) <= 8 and len(s) <= 60 and not s.endswith((".", ";", ",")):
        titleish = sum(w[0].isupper() for w in words) / len(words)
        if titleish >= 0.6:
            return True

    return False


def is_probable_subheading(line: str) -> bool:
    s = line.strip()
    # subheading if it’s a numbered heading that starts with at least "X.Y ..."
    return bool(re.match(r"^\d+\.\d+(\.\d+)*\s+\S+", s))


# =============================
# REFLOW TEXT ONLY (DO NOT TOUCH TABLES)
# =============================
def restructure_and_reflow(text: str) -> str:
    lines = text.splitlines()
    out: List[str] = []

    in_table = False
    blank_run = 0

    def table_end_logic(s: str):
        nonlocal in_table, blank_run
        if s == "":
            blank_run += 1
        else:
            blank_run = 0
        if blank_run >= END_TABLE_BLANKS:
            in_table = False
            blank_run = 0

    i = 0
    while i < len(lines):
        raw = lines[i].rstrip("\n")
        s = raw.strip()

        # keep page markers as-is (with spacing)
        if PAGE_MARKER_RE.match(s):
            if out and out[-1].strip() != "":
                out.append("")
            out.append(s)
            out.append("")
            i += 1
            continue

        # Start table
        if not in_table and is_table_start(s):
            in_table = True
            blank_run = 0
            out.append(raw.rstrip())
            i += 1
            continue

        # Inside table (RAW)
        if in_table:
            out.append(raw.rstrip())
            table_end_logic(s)
            i += 1
            continue

        # Headings/subheadings
        if is_probable_heading(s):
            out.append(s)
            i += 1
            continue

        if is_probable_subheading(s):
            out.append("")
            out.append(s)
            out.append("")
            i += 1
            continue

        # URL
        if URL_RE.match(s):
            out.append(s)
            i += 1
            continue

        # Blank
        if s == "":
            out.append("")
            i += 1
            continue

        # List line
        if is_list_line(s):
            if out and out[-1].strip() != "":
                out.append("")
            out.append(s)
            i += 1
            continue

        # Reflow paragraph
        para_parts = [s]
        j = i + 1
        while j < len(lines):
            nxt_raw = lines[j].rstrip("\n")
            nxt = nxt_raw.strip()

            if nxt == "":
                break
            if PAGE_MARKER_RE.match(nxt):
                break
            if is_probable_heading(nxt) or is_probable_subheading(nxt):
                break
            if is_table_start(nxt):
                break
            if URL_RE.match(nxt):
                break
            if is_list_line(nxt):
                break

            # hyphen wrap fix
            if para_parts[-1].endswith("-") and nxt and nxt[0].isalnum():
                para_parts[-1] = para_parts[-1][:-1] + nxt
            else:
                para_parts.append(nxt)

            j += 1

        paragraph = re.sub(r"\s+", " ", " ".join(para_parts)).strip()
        out.append(paragraph)
        i = j

    final = "\n".join(out)
    final = MULTI_BLANK_RE.sub("\n\n", final).strip()
    return final


# =============================
# SPLIT INTO SECTIONS
# =============================
def split_into_sections(text: str) -> List[Dict[str, str]]:
    lines = text.splitlines()
    sections: List[Dict[str, str]] = []
    title = "INTRO"
    buf: List[str] = []

    for line in lines:
        if is_probable_heading(line):
            if buf:
                sections.append({"title": title, "content": "\n".join(buf).strip()})
                buf = []
            title = line.strip()
        buf.append(line)

    if buf:
        sections.append({"title": title, "content": "\n".join(buf).strip()})
    return sections


def strip_duplicate_heading(title: str, content: str) -> str:
    lines = content.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip() == "":
            continue
        if ln.strip() == title.strip():
            lines.pop(i)
        break
    return "\n".join(lines).strip()


# =============================
# SPLIT TABLES VS TEXT (RAW TABLES)
# =============================
def split_tables_and_text(section_content: str) -> List[Tuple[str, str]]:
    lines = section_content.splitlines()
    blocks: List[Tuple[str, str]] = []

    text_buf: List[str] = []
    table_buf: List[str] = []
    in_table = False
    blank_run = 0

    def flush_text():
        nonlocal text_buf
        txt = "\n".join(text_buf).strip()
        if txt:
            blocks.append(("text", txt))
        text_buf = []

    def flush_table():
        nonlocal table_buf
        tbl = "\n".join(table_buf).strip("\n").rstrip()
        if tbl.strip():
            blocks.append(("table", tbl))
        table_buf = []

    for line in lines:
        s = line.strip()

        if not in_table and is_table_start(s):
            flush_text()
            in_table = True
            blank_run = 0
            table_buf.append(line.rstrip())
            continue

        if in_table:
            table_buf.append(line.rstrip())
            if s == "":
                blank_run += 1
            else:
                blank_run = 0
            if blank_run >= END_TABLE_BLANKS:
                flush_table()
                in_table = False
            continue

        text_buf.append(line.rstrip())

    if in_table:
        flush_table()
    flush_text()
    return blocks


# =============================
# CHUNK TEXT (WITH OVERLAP)
# =============================
def get_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


def chunk_by_paragraphs_with_overlap(text: str) -> List[str]:
    paras = get_paragraphs(text)
    if not paras:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    last_para: Optional[str] = None  # overlap paragraph

    for p in paras:
        p = p.strip()
        if not p:
            continue

        add_len = len(p) + (2 if current else 0)

        if current and current_len + add_len > MAX_CHARS:
            chunks.append("\n\n".join(current).strip())
            last_para = current[-1]
            current = [last_para] if last_para else []
            current_len = len(last_para) if last_para else 0

        current.append(p)
        current_len += len(p) + 2

    if current:
        chunks.append("\n\n".join(current).strip())

    return chunks


# =============================
# METADATA HELPERS
# =============================
def find_pages_in_text(text: str):
    pages = [int(x) for x in re.findall(r"===\s*PAGE\s+(\d+)\s*===", text, flags=re.I)]
    if not pages:
        return None, None
    return min(pages), max(pages)


# =============================
# MAIN
# =============================
def main():
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    raw = in_path.read_text(encoding="utf-8", errors="ignore")
    cleaned = restructure_and_reflow(raw)
    sections = split_into_sections(cleaned)

    items: List[Dict] = []

    for sec in sections:
        section_title = sec["title"].strip()
        sec_content = strip_duplicate_heading(section_title, sec["content"])
        blocks = split_tables_and_text(sec_content)

        for btype, bcontent in blocks:
            if btype == "table":
                items.append({
                    "type": "table",
                    "section": section_title,
                    "text": bcontent
                })
            else:
                for piece in chunk_by_paragraphs_with_overlap(bcontent):
                    items.append({
                        "type": "text",
                        "section": section_title,
                        "text": piece
                    })

    # WRITE: Standard JSONL (one object per line)
    with out_path.open("w", encoding="utf-8") as f:
        for idx, it in enumerate(items, 1):
            page_start, page_end = find_pages_in_text(it["text"])
            obj = {
                "id": f"chunk_{idx:05d}",
                "text": it["text"],
                "metadata": {
                    "source": SOURCE_NAME,
                    "section": it["section"],
                    "type": it["type"],            # "text" or "table"
                    "chunk_index": idx,
                    "char_len": len(it["text"]),
                    "page_start": page_start,
                    "page_end": page_end,
                }
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"✅ Output saved: {out_path}")
    print(f"✅ Total chunks: {len(items)}")
    print("✅ Tables are stored RAW as separate chunks with metadata.type='table'")


if __name__ == "__main__":
    main()