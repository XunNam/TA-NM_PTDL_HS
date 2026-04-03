from __future__ import annotations

from pathlib import Path

import nltk
from nltk import FreqDist
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.corpus import gutenberg, stopwords

from src.nltk_setup import ensure_nltk_resources
from src.utils import (
    OUTPUT_DIR,
    filter_stopwords_and_punctuation,
    format_ranked_items,
    pretty_print_section,
    save_text,
)


OUTPUT_PATH = OUTPUT_DIR / "section_05" / "ngrams_results.txt"


def run(output_path: Path | None = None) -> dict[str, object]:
    """Run Lab 7 section 05: bigrams, trigrams, and optional collocations."""
    ensure_nltk_resources(["gutenberg", "stopwords"])
    output_path = output_path or OUTPUT_PATH

    tokens = list(gutenberg.words("shakespeare-macbeth.txt"))
    english_stopwords = set(stopwords.words("english"))
    cleaned_tokens = filter_stopwords_and_punctuation(tokens, english_stopwords)

    bigram_freq = FreqDist(nltk.bigrams(cleaned_tokens))
    trigram_freq = FreqDist(nltk.trigrams(cleaned_tokens))

    finder = BigramCollocationFinder.from_words(cleaned_tokens)
    finder.apply_freq_filter(2)
    collocations = finder.nbest(BigramAssocMeasures().pmi, 10)

    lines = [
        pretty_print_section("SECTION 05 - NGRAMS AND COLLOCATIONS"),
        f"Cleaned token count: {len(cleaned_tokens)}",
        "",
        "Top 15 bigrams:",
        format_ranked_items(bigram_freq.most_common(15)),
        "",
        "Top 10 trigrams:",
        format_ranked_items(trigram_freq.most_common(10)),
        "",
        "Top 10 collocations by PMI (frequency >= 2):",
    ]
    lines.extend(f"{index:>2}. {' '.join(collocation)}" for index, collocation in enumerate(collocations, start=1))

    save_text(output_path, "\n".join(lines))
    return {
        "status": "success",
        "output_path": str(output_path),
        "cleaned_token_count": len(cleaned_tokens),
        "top_bigram": bigram_freq.most_common(1)[0] if bigram_freq else None,
        "top_trigram": trigram_freq.most_common(1)[0] if trigram_freq else None,
    }


if __name__ == "__main__":
    run()
