from __future__ import annotations

from pathlib import Path

import nltk
from nltk.corpus import gutenberg

from src.nltk_setup import ensure_nltk_resources
from src.utils import OUTPUT_DIR, pretty_print_section, save_text


OUTPUT_PATH = OUTPUT_DIR / "section_01" / "intro_results.txt"


def run(output_path: Path | None = None) -> dict[str, object]:
    """Run Lab 7 section 01: introduction to NLTK and the Gutenberg corpus."""
    ensure_nltk_resources(["gutenberg", "punkt"])
    output_path = output_path or OUTPUT_PATH

    file_ids = gutenberg.fileids()
    macbeth_words = list(gutenberg.words("shakespeare-macbeth.txt"))
    macbeth_sentences = list(gutenberg.sents("shakespeare-macbeth.txt"))

    lines = [
        pretty_print_section("SECTION 01 - INTRODUCTION TO NLTK"),
        f"NLTK version: {nltk.__version__}",
        "",
        "Gutenberg corpus file IDs:",
        *file_ids,
        "",
        "Macbeth summary:",
        f"Total words: {len(macbeth_words)}",
        f"First 10 words: {macbeth_words[:10]}",
        "",
        "First 5 sentences:",
    ]
    lines.extend(
        f"{index}. {' '.join(sentence)}"
        for index, sentence in enumerate(macbeth_sentences[:5], start=1)
    )

    save_text(output_path, "\n".join(lines))
    return {
        "status": "success",
        "output_path": str(output_path),
        "word_count": len(macbeth_words),
        "sentence_count": len(macbeth_sentences),
        "gutenberg_file_count": len(file_ids),
    }


if __name__ == "__main__":
    run()
