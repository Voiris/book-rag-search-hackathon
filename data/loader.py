from pathlib import Path
from utils.text import normalize_text

def safe_read_text(file_path: Path) -> str:
    for enc in ["utf-8", "cp1251", "utf-8-sig", "latin-1"]:
        try:
            return file_path.read_text(encoding=enc)
        except Exception:
            pass
    return file_path.read_bytes().decode("utf-8", errors="ignore")


def load_books(folder: Path):
    books = []

    for file_path in sorted(folder.glob("*.txt")):
        books.append({
            "book_id": file_path.stem,
            "title": file_path.stem,
            "path": str(file_path),
            "text": normalize_text(safe_read_text(file_path))
        })

    return books
