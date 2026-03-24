import re

# --- compiled patterns ---
MULTI_SPACES = re.compile(r" {2,}")
MULTI_NEWLINES = re.compile(r"\n{3,}")
CLEAN_CHARS = re.compile(r"[^a-zа-я0-9\s.,!?;:\-\"'()]")
MULTI_WHITESPACE = re.compile(r"\s+")
SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')

def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00A0", " ")
    text = text.replace("\t", " ")

    text = MULTI_SPACES.sub(" ", text)
    text = MULTI_NEWLINES.sub("\n\n", text)

    return text.strip()


def clean_for_lexical_search(text: str) -> str:
    text = text.lower().replace("ё", "е")

    text = CLEAN_CHARS.sub(" ", text)

    return re.sub(r"\s+", " ", text).strip()


def split_into_sentences(text: str):
    return SENTENCE_SPLIT.split(normalize_text(text))
