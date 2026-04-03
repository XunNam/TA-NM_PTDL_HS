from __future__ import annotations

from pathlib import Path

from nltk.corpus import gutenberg

from src.nltk_setup import ensure_nltk_resources
from src.utils import OUTPUT_DIR, pretty_print_section, save_text


OUTPUT_PATH = OUTPUT_DIR / "section_04" / "word_selection_results.txt"


def run(output_path: Path | None = None) -> dict[str, object]:
    """Run Lab 7 section 04: select words that match simple patterns."""
    ensure_nltk_resources(["gutenberg"])
    output_path = output_path or OUTPUT_PATH

    tokens = list(gutenberg.words("shakespeare-macbeth.txt"))
    long_words = sorted({token.lower() for token in tokens if token.isalpha() and len(token) > 12})
    ious_words = sorted(
        {token.lower() for token in tokens if token.isalpha() and "ious" in token.lower()}
    )

    lines = [
        pretty_print_section("SECTION 04 - WORD SELECTION"),
        f"Words longer than 12 characters ({len(long_words)} items):",
        ", ".join(long_words) or "(No match.)",
        "",
        f"Words containing 'ious' ({len(ious_words)} items):",
        ", ".join(ious_words) or "(No match.)",
    ]

    save_text(output_path, "\n".join(lines))
    return {
        "status": "success",
        "output_path": str(output_path),
        "long_word_count": len(long_words),
        "ious_word_count": len(ious_words),
    }


if __name__ == "__main__":
    run()
