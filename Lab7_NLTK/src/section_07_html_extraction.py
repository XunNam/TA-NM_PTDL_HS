from __future__ import annotations

from pathlib import Path

import nltk
from bs4 import BeautifulSoup

from src.nltk_setup import ensure_nltk_resources
from src.utils import DOWNLOAD_DIR, OUTPUT_DIR, pretty_print_section, safe_request, save_text


OUTPUT_PATH = OUTPUT_DIR / "section_07" / "html_extraction_results.txt"
PRIMARY_URL = "http://news.bbc.co.uk/2/hi/health/2284783.stm"
FALLBACK_URLS = [
    "https://www.gutenberg.org/files/2554/2554-h/2554-h.htm",
    "https://www.gutenberg.org/cache/epub/2554/pg2554-images.html",
]


def _download_html() -> tuple[str, str, list[str]]:
    attempts: list[str] = []
    for url in [PRIMARY_URL, *FALLBACK_URLS]:
        try:
            response = safe_request(url, timeout=30)
            html = response.text
            if len(html) < 1000:
                raise ValueError("Downloaded HTML is unexpectedly short.")
            return url, html, attempts
        except Exception as exc:
            attempts.append(f"{url} -> {exc}")
    raise RuntimeError("Unable to fetch any HTML page for extraction.")


def _extract_visible_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    source = soup.body or soup
    return " ".join(source.stripped_strings)


def run(output_path: Path | None = None) -> dict[str, object]:
    """Run Lab 7 section 07: extract text from an HTML page."""
    ensure_nltk_resources(["punkt", "punkt_tab"])
    output_path = output_path or OUTPUT_PATH

    try:
        source_url, raw_html, attempts = _download_html()
        html_path = DOWNLOAD_DIR / "section_07_source_page.html"
        save_text(html_path, raw_html)

        extracted_text = _extract_visible_text(raw_html)
        tokens = nltk.word_tokenize(extracted_text)
        html_text = nltk.Text(tokens)
        preview = " ".join(tokens[:80])

        lines = [
            pretty_print_section("SECTION 07 - HTML EXTRACTION"),
            f"Source URL: {source_url}",
            f"Fallback used: {'yes' if source_url != PRIMARY_URL else 'no'}",
            f"Extracted text length: {len(extracted_text)} characters",
            f"Token count: {len(tokens)}",
            f"NLTK Text length: {len(html_text)}",
            f"Saved HTML: {html_path}",
            "",
            "Preview:",
            preview,
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
            pretty_print_section("SECTION 07 - HTML EXTRACTION"),
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
