from pathlib import Path


def safe_read_text(file_path: Path) -> str:
    encodings = ["utf-8", "cp1251", "utf-8-sig", "latin-1"]
    for enc in encodings:
        try:
            return file_path.read_text(encoding=enc)
        except Exception:
            continue
    # Последняя попытка
    with open(file_path, "rb") as f:
        raw = f.read()
    return raw.decode("utf-8", errors="ignore")
