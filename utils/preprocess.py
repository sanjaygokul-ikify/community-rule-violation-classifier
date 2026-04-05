import re


def clean_text(text: str) -> str:
    """Clean Reddit comment text."""
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)       # Remove URLs
    text = re.sub(r"u/\w+", "", text)                  # Remove user mentions
    text = re.sub(r"r/\w+", "", text)                  # Remove subreddit mentions
    text = re.sub(r"[^\w\s'.,!?-]", " ", text)        # Remove special chars
    text = re.sub(r"\s+", " ", text).strip()           # Normalize whitespace
    return text.lower()


def preprocess_batch(texts: list) -> list:
    return [clean_text(t) for t in texts]
