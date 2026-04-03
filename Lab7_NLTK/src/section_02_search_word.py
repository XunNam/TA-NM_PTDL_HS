from __future__ import annotations

from pathlib import Path

import nltk
from nltk.corpus import gutenberg

from src.nltk_setup import ensure_nltk_resources
from src.utils import OUTPUT_DIR, capture_stdout, pretty_print_section, save_text


OUTPUT_PATH = OUTPUT_DIR / "section_02" / "search_word_results.txt"
SEARCH_TERM = "Stage"


def run(output_path: Path | None = None) -> dict[str, object]:
    """Run Lab 7 section 02: search for a word with NLTK Text."""
    ensure_nltk_resources(["gutenberg"])
    output_path = output_path or OUTPUT_PATH

    macbeth_words = list(gutenberg.words("shakespeare-macbeth.txt"))
    macbeth_text = nltk.Text(macbeth_words)

    concordance_output = capture_stdout(macbeth_text.concordance, SEARCH_TERM, lines=25)
    common_contexts_output = capture_stdout(macbeth_text.common_contexts, [SEARCH_TERM])
    similar_output = capture_stdout(macbeth_text.similar, SEARCH_TERM, num=20)

    lines = [
        pretty_print_section("SECTION 02 - SEARCHING FOR A WORD"),
        f"Search term: {SEARCH_TERM}",
        f"Occurrences in Macbeth: {macbeth_words.count(SEARCH_TERM)}",
        "",
        "Concordance:",
        concordance_output or "(No concordance output.)",
        "",
        "Common contexts:",
        common_contexts_output or "(No common-context output.)",
        "",
        "Similar words:",
        similar_output or "(No similar-word output.)",
    ]

    save_text(output_path, "\n".join(lines))
    return {
        "status": "success",
        "output_path": str(output_path),
        "search_term": SEARCH_TERM,
        "occurrences": macbeth_words.count(SEARCH_TERM),
    }


if __name__ == "__main__":
    run()
