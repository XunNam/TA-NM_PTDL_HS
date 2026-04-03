from __future__ import annotations

from pathlib import Path

import nltk

from src.nltk_setup import ensure_nltk_resources
from src.utils import DOWNLOAD_DIR, OUTPUT_DIR, pretty_print_section, safe_request, save_text


OUTPUT_PATH = OUTPUT_DIR / "section_06" / "web_text_results.txt"
PRIMARY_URL = "https://www.gutenberg.org/files/2554/2554-0.txt"
FALLBACK_URLS = [
    "https://www.gutenberg.org/cache/epub/2554/pg2554.txt",
    "https://www.gutenberg.org/files/1342/1342-0.txt",
]


def _download_text() -> tuple[str, str, list[str]]:
    attempts: list[str] = []
    for url in [PRIMARY_URL, *FALLBACK_URLS]:
        try:
            response = safe_request(url, timeout=30)
            text = response.content.decode("utf-8-sig", errors="replace")
            if len(text.split()) < 500:
                raise ValueError("Downloaded text is unexpectedly short.")
            return url, text, attempts
        except Exception as exc:
            attempts.append(f"{url} -> {exc}")
    raise RuntimeError("Unable to fetch any remote plain-text source.")


def run(output_path: Path | None = None) -> dict[str, object]:
    """Run Lab 7 section 06: use text downloaded from the web."""
    ensure_nltk_resources(["punkt", "punkt_tab"])
    output_path = output_path or OUTPUT_PATH

    try:
        source_url, raw_text, attempts = _download_text()
        raw_text_path = DOWNLOAD_DIR / "section_06_web_text.txt"
        save_text(raw_text_path, raw_text)

        tokens = nltk.word_tokenize(raw_text)
        remote_text = nltk.Text(tokens)
        lines = [
            pretty_print_section("SECTION 06 - WEB TEXT"),
            f"Source URL: {source_url}",
            f"Fallback used: {'yes' if source_url != PRIMARY_URL else 'no'}",
            f"Token count: {len(tokens)}",
            f"First 12 tokens: {tokens[:12]}",
            f"NLTK Text length: {len(remote_text)}",
            f"Saved raw text: {raw_text_path}",
        ]
        if attempts:
            lines.extend(["", "Failed attempts before success:", *attempts])

        save_text(output_path, "\n".join(lines))
        return {
            "status": "success",
            "output_path": str(output_path),
            "source_url": source_url,
            "fallback_used": source_url != PRIMARY_URL,
            "token_count": len(tokens),
        }
    except Exception as exc:
        lines = [
            pretty_print_section("SECTION 06 - WEB TEXT"),
            "Status: failed",
            str(exc),
        ]
        save_text(output_path, "\n".join(lines))
        return {
            "status": "failed",
            "output_path": str(output_path),
            "source_url": None,
            "fallback_used": False,
            "error": str(exc),
        }


if __name__ == "__main__":
    run()
