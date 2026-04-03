from __future__ import annotations

import random
from pathlib import Path

import nltk
from nltk import FreqDist
from nltk.corpus import movie_reviews

from src.nltk_setup import ensure_nltk_resources
from src.utils import OUTPUT_DIR, capture_stdout, pretty_print_section, save_text, setup_random_seed


OUTPUT_PATH = OUTPUT_DIR / "section_08" / "sentiment_results.txt"
INFORMATIVE_FEATURES_PATH = OUTPUT_DIR / "section_08" / "top_informative_features.txt"
_WORD_FEATURE_LOOKUP: set[str] = set()


def document_features(document: list[str], word_features: list[str]) -> dict[str, bool]:
    """Create sparse boolean word-presence features for a document."""
    global _WORD_FEATURE_LOOKUP
    if len(_WORD_FEATURE_LOOKUP) != len(word_features):
        _WORD_FEATURE_LOOKUP = set(word_features)

    document_words = {word.lower() for word in document}
    return {
        f"contains({word})": True
        for word in document_words
        if word in _WORD_FEATURE_LOOKUP
    }


def run(output_path: Path | None = None) -> dict[str, object]:
    """Run Lab 7 section 08: user sentiment classification with movie reviews."""
    ensure_nltk_resources(["movie_reviews"])
    output_path = output_path or OUTPUT_PATH

    seed = setup_random_seed(42)
    documents = [
        (list(movie_reviews.words(file_id)), category)
        for category in movie_reviews.categories()
        for file_id in movie_reviews.fileids(category)
    ]
    random.shuffle(documents)

    all_words = FreqDist(word.lower() for word in movie_reviews.words())
    word_features = list(all_words)
    featuresets = [
        (document_features(document, word_features), category)
        for document, category in documents
    ]

    train_set = featuresets[:1500]
    test_set = featuresets[1500:2000]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    accuracy = nltk.classify.accuracy(classifier, test_set)

    sample_review_words, sample_actual_label = documents[0]
    sample_prediction = classifier.classify(document_features(sample_review_words, word_features))
    sample_preview = " ".join(sample_review_words[:120])
    informative_output = capture_stdout(classifier.show_most_informative_features, 10)

    lines = [
        pretty_print_section("SECTION 08 - SENTIMENT ANALYSIS"),
        f"Random seed: {seed}",
        f"Total documents: {len(documents)}",
        f"Train set size: {len(train_set)}",
        f"Test set size: {len(test_set)}",
        f"Unique word features tracked: {len(word_features)}",
        "",
        "Sample review preview:",
        sample_preview,
        "",
        f"Actual label: {sample_actual_label}",
        f"Predicted label: {sample_prediction}",
        f"Accuracy: {accuracy:.4f}",
        "",
        "Top 10 informative features:",
        informative_output or "(No informative feature output.)",
    ]

    save_text(output_path, "\n".join(lines))
    save_text(INFORMATIVE_FEATURES_PATH, informative_output or "(No informative feature output.)")

    return {
        "status": "success",
        "output_path": str(output_path),
        "accuracy": accuracy,
        "sample_actual_label": sample_actual_label,
        "sample_prediction": sample_prediction,
        "feature_count": len(word_features),
    }


if __name__ == "__main__":
    run()
