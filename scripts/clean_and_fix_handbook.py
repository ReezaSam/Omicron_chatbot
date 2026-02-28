import re
from pathlib import Path
from typing import List

# =============================
# PATHS
# =============================
root = Path(__file__).resolve().parent.parent

raw_path = root / "raw" / "faculty_handbook.txt"
out_path = root / "clean" / "faculty_handbook_cleaned.txt"
out_path.parent.mkdir(parents=True, exist_ok=True)

# =============================
# REGEX PATTERNS
# =============================
HEADER_RE = re.compile(r"^\s*Handbook\s+\d{4}/\d{2}\s*$", re.I)
PAGE_NUM_RE = re.compile(r"^\s*\d+\s*$")
TOC_DOTS_RE = re.compile(r"\.{5,}\s*\d+\s*$")
FORMFEED_RE = re.compile(r"\f")

# Keep page markers we add in pdf_to_text.py
PAGE_MARKER_RE = re.compile(r"^\s*===\s*PAGE\s+\d+\s*===\s*$", re.I)

COURSE_CODE_RE = re.compile(r"\b[A-Z]{2,4}\s*\d{4}\b")
GRADE_PAIR_RE = re.compile(
    r"\b(A\+|A-|B\+|B-|C\+|C-|D\+|A|B|C|D|E)\b\s*([0-9]\.[0-9])",
    re.I,
)

# =============================
# BASIC NORMALIZATION
# =============================
def normalize_line(line: str) -> str:
    line = FORMFEED_RE.sub("", line.rstrip("\n"))
    line = (
        line.replace("–", "-")
        .replace("—", "-")
        .replace("̶", "-")
    )
    return line.rstrip()


def is_noise_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False

    # never remove our page markers
    if PAGE_MARKER_RE.match(s):
        return False

    if HEADER_RE.match(s):
        return True
    if PAGE_NUM_RE.match(s):
        return True
    if TOC_DOTS_RE.search(s):
        return True
    return False


# =============================
# TABLE LOGIC
# =============================
def looks_like_table_header(line: str) -> bool:
    l = line.lower()
    return (
        ("course code" in l and "course title" in l)
        or ("field of study" in l and "meaning" in l)
        or ("grade" in l and "grade point" in l)
        or ("credit" in l and "course title" in l)
        or ("prerequisite" in l and "corequisite" in l)
    )


def looks_like_table_row_start(line: str) -> bool:
    if COURSE_CODE_RE.search(line[:40]):
        return True
    if GRADE_PAIR_RE.search(line.replace(" ", "")):
        return True
    return False


def looks_like_continuation(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if looks_like_table_row_start(s):
        return False

    # indentation continuation
    if len(line) - len(line.lstrip()) >= 2:
        return True

    # lowercase continuation or punctuation continuation
    if s[:1].islower() or s[:1] in ",.;:)":
        return True

    # short line without digits often belongs to previous row (course title etc.)
    if not any(ch.isdigit() for ch in s) and len(s) <= 40:
        return True

    return False


def merge_lines(prev: str, curr: str) -> str:
    if prev.rstrip().endswith("-"):
        return prev.rstrip()[:-1] + curr.lstrip()
    return prev.rstrip() + " " + curr.lstrip()


def clean_tables(lines: List[str]) -> List[str]:
    cleaned: List[str] = []
    in_table = False
    blank_count = 0

    for line in lines:
        s = line.strip()

        # Start table
        if re.match(r"^\s*Table\s+\d+(\.\d+)+", s, re.I) or looks_like_table_header(s):
            in_table = True
            blank_count = 0
            cleaned.append(line)
            continue

        # table end by blank run
        if in_table and not s:
            blank_count += 1
        else:
            blank_count = 0

        if in_table and blank_count >= 2:
            in_table = False

        # Safer merge rule: only merge if previous line looks like a row/header/table label
        if in_table and cleaned and looks_like_continuation(line):
            prev = cleaned[-1].strip()
            prev_is_tableish = (
                looks_like_table_row_start(prev)
                or looks_like_table_header(prev)
                or prev.lower().startswith("table")
            )
            if prev_is_tableish:
                cleaned[-1] = merge_lines(cleaned[-1], line)
                continue

        cleaned.append(line)

    return cleaned


# =============================
# FINAL CLEANUP
# =============================
def collapse_blank_lines(lines: List[str], max_blank: int = 1) -> List[str]:
    out: List[str] = []
    blanks = 0
    for l in lines:
        if not l.strip():
            blanks += 1
            if blanks <= max_blank:
                out.append("")
        else:
            blanks = 0
            out.append(l.rstrip())

    while out and out[0] == "":
        out.pop(0)
    while out and out[-1] == "":
        out.pop()

    return out


# =============================
# MAIN
# =============================
def main():
    text = raw_path.read_text(encoding="utf-8", errors="ignore")
    lines = [normalize_line(l) for l in text.splitlines()]

    # remove headers / page numbers / TOC noise
    lines = [l for l in lines if not is_noise_line(l)]

    # clean tables in-place
    lines = clean_tables(lines)

    # normalize spacing
    lines = collapse_blank_lines(lines, max_blank=1)

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✅ Cleaned text saved to:\n{out_path}")


if __name__ == "__main__":
    main()