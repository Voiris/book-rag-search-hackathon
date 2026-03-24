import re
from typing import List, Tuple, Dict, Any

from config import CHUNK_SIZE, CHUNK_OVERLAP
from utils.text import normalize_text, clean_for_lexical_search

SPLIT_RULES = (
    (0, lambda c, n: c == "\n" and n == "\n", 1),
    (1, lambda c, n: c == "\n", 0),
    (2, lambda c, n: c in ".!?", 0),
    (3, lambda c, n: c == " ", 0),
)
CHAPTER_PATTERN = re.compile(
    r"(глава\s+[^\n]+|chapter\s+[^\n]+)",
    re.IGNORECASE
)

class TextState:
    def __init__(self):
        self.quote_depth = 0

    @property
    def is_can_split(self):
        return self.quote_depth == 0

    def update_state(self, char):
        if char in ('"', '«'):
            self.quote_depth += 1
        elif char in ('"', '»'):
            self.quote_depth = max(0, self.quote_depth - 1)

def smart_text_split(text: str, max_len: int) -> List[str]:
    if len(text) <= max_len:
        return [text]

    text_state = TextState()
    split_index = max_len - 1
    used_priority = None

    n = len(text)
    limit = min(n - 1, max_len - 1)

    for i in range(limit):
        char = text[i]
        next_char = text[i + 1] if i + 1 < n else None

        text_state.update_state(char)

        if text_state.is_can_split:
            for priority, rule, offset in SPLIT_RULES:
                if rule(char, next_char) and (used_priority is None or priority < used_priority):
                    used_priority = priority
                    split_index = i + offset

    return [
        text[:split_index+1].rstrip(),
        text[split_index+1:].rstrip()
    ]

def split_text_into_chunks(text: str, chunk_size: int = 1400, overlap: int = 250) -> List[Tuple[int, int, str]]:
    """
    Возвращает список чанков: (start_char, end_char, chunk_text)
    """
    text = normalize_text(text)
    chunks = []

    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk_text = text[start:end]

        parts = smart_text_split(chunk_text, chunk_size)
        chunk_text = parts[0]

        actual_end = start + len(chunk_text)
        if chunk_text:
            chunks.append((start, actual_end, chunk_text.strip()))

        if actual_end >= n:
            break

        start = max(actual_end - overlap, start + 1)

    return chunks


def build_chunks(books: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    all_chunks = []

    for book in books:
        text_chunks = split_text_into_chunks(book["text"], CHUNK_SIZE, CHUNK_OVERLAP)

        for idx, (start_char, end_char, chunk_text) in enumerate(text_chunks):
            chapter_match = CHAPTER_PATTERN.search(chunk_text[:300])
            chapter = chapter_match.group(1).strip() if chapter_match else None

            all_chunks.append({
                "chunk_id": f"{book['book_id']}_chunk_{idx}",
                "book_id": book["book_id"],
                "title": book["title"],
                "chunk_index": idx,
                "start_char": start_char,
                "end_char": end_char,
                "chapter": chapter,
                "text": chunk_text,
                "search_text": clean_for_lexical_search(chunk_text)
            })

    return all_chunks
