import pdfplumber
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

pdf_path = BASE_DIR / "raw" / "faculty_handbook.pdf"
txt_path = BASE_DIR / "raw" / "faculty_handbook.txt"

print("Looking for PDF at:")
print(pdf_path)

if not pdf_path.exists():
    raise FileNotFoundError(f"PDF not found at: {pdf_path}")

missing_pages = []

with pdfplumber.open(pdf_path) as pdf:
    with open(txt_path, "w", encoding="utf-8") as f:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(layout=True)

            # Always write a page marker
            f.write(f"\n\n=== PAGE {i} ===\n\n")

            if text:
                f.write(text + "\n")
            else:
                missing_pages.append(i)
                f.write("[NO TEXT EXTRACTED]\n")

if missing_pages:
    print("⚠️ Pages with no extracted text:", missing_pages)

print("✅ PDF converted to TXT successfully")
print(f"Saved to: {txt_path}")