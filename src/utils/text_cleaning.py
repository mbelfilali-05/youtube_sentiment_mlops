import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+|\bu/\w+|\br/\w+")
SPECIAL_RE = re.compile(r"[^a-z0-9\s!?']")
SPACES_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = SPECIAL_RE.sub(" ", text)
    return SPACES_RE.sub(" ", text).strip()