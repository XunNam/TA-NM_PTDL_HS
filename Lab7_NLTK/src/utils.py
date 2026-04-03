from __future__ import annotations

import io
import random
import string
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Callable, Iterable

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DOWNLOAD_DIR = DATA_DIR / "downloaded"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0 Safari/537.36"
    )
}


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_text(path: str | Path, content: str) -> Path:
    """Save UTF-8 text content to a file, creating parent directories as needed."""
    file_path = Path(path)
    ensure_dir(file_path.parent)
    file_path.write_text(content, encoding="utf-8")
    return file_path


def pretty_print_section(title: str) -> str:
    """Return a simple section banner for text reports."""
    line = "=" * 80
    return f"{line}\n{title}\n{line}"


def setup_random_seed(seed: int = 42) -> int:
    """Seed Python's random module and return the chosen seed."""
    random.seed(seed)
    return seed


def capture_stdout(func: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    """Capture stdout from functions that print instead of returning values."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        func(*args, **kwargs)
    return buffer.getvalue().strip()


def safe_request(
    url: str,
    timeout: int = 20,
    headers: dict[str, str] | None = None,
) -> requests.Response:
    """Perform a GET request with default headers and raise for HTTP errors."""
    merged_headers = dict(DEFAULT_HEADERS)
    if headers:
        merged_headers.update(headers)

    response = requests.get(url, timeout=timeout, headers=merged_headers)
    response.raise_for_status()
    return response


def filter_stopwords_and_punctuation(
    tokens: Iterable[str],
    stopword_set: set[str],
) -> list[str]:
    """Lowercase tokens and remove stopwords plus punctuation-only items."""
    cleaned: list[str] = []
    for token in tokens:
        lowered = token.lower()
        if lowered in stopword_set:
            continue
        if lowered in string.punctuation:
            continue
        if not any(character.isalpha() for character in lowered):
            continue
        cleaned.append(lowered)
    return cleaned


def format_ranked_items(items: Iterable[tuple[Any, int]]) -> str:
    """Render ranked items in a readable plain-text format."""
    lines: list[str] = []
    for index, (item, count) in enumerate(items, start=1):
        if isinstance(item, tuple):
            item_text = " ".join(str(part) for part in item)
        else:
            item_text = str(item)
        lines.append(f"{index:>2}. {item_text} -> {count}")
    return "\n".join(lines)

