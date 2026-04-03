from __future__ import annotations

import string
from pathlib import Path

from nltk import FreqDist
from nltk.corpus import gutenberg, stopwords

from src.nltk_setup import ensure_nltk_resources
from src.utils import OUTPUT_DIR, format_ranked_items, pretty_print_section, save_text


OUTPUT_PATH = OUTPUT_DIR / "section_03" / "frequency_results.txt"


def run(output_path: Path | None = None) -> dict[str, object]:
    """Run Lab 7 section 03: frequency analysis on Macbeth."""
    ensure_nltk_resources(["gutenberg", "stopwords"])
    output_path = output_path or OUTPUT_PATH

    tokens = list(gutenberg.words("shakespeare-macbeth.txt"))
    english_stopwords = set(stopwords.words("english"))

    raw_freq = FreqDist(tokens)
    no_stopwords = [token.lower() for token in tokens if token.lower() not in english_stopwords]
    punctuation_tokens = set(string.punctuation)
    no_stopwords_or_punctuation = [
        token
        for token in no_stopwords
        if token not in punctuation_tokens and any(character.isalpha() for character in token)
    ]

    no_stopwords_freq = FreqDist(no_stopwords)
    clean_freq = FreqDist(no_stopwords_or_punctuation)

    lines = [
        pretty_print_section("SECTION 03 - FREQUENCY ANALYSIS"),
        "Top 10 tokens in raw Macbeth text:",
        format_ranked_items(raw_freq.most_common(10)),
        "",
        "Top 10 tokens after removing English stopwords:",
        format_ranked_items(no_stopwords_freq.most_common(10)),
        "",
        "Top 10 tokens after removing stopwords and punctuation:",
        format_ranked_items(clean_freq.most_common(10)),
    ]

    save_text(output_path, "\n".join(lines))
    return {
        "status": "success",
        "output_path": str(output_path),
        "raw_top_10": raw_freq.most_common(10),
        "clean_top_10": clean_freq.most_common(10),
    }


if __name__ == "__main__":
    run()
