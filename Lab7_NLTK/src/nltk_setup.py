from __future__ import annotations

from collections.abc import Iterable

import nltk


RESOURCE_SPECS = {
    "gutenberg": ("corpora/gutenberg", "corpora/gutenberg.zip"),
    "punkt": ("tokenizers/punkt", "tokenizers/punkt.zip"),
    "punkt_tab": ("tokenizers/punkt_tab", "tokenizers/punkt_tab.zip"),
    "stopwords": ("corpora/stopwords", "corpora/stopwords.zip"),
    "movie_reviews": ("corpora/movie_reviews", "corpora/movie_reviews.zip"),
    "names": ("corpora/names", "corpora/names.zip"),
    "wordnet": ("corpora/wordnet", "corpora/wordnet.zip"),
    "omw-1.4": ("corpora/omw-1.4", "corpora/omw-1.4.zip"),
    "tagsets": ("help/tagsets/upenn_tagset.pickle", "help/tagsets"),
    "averaged_perceptron_tagger": (
        "taggers/averaged_perceptron_tagger",
        "taggers/averaged_perceptron_tagger.zip",
    ),
}


def _resource_is_available(resource_paths: tuple[str, ...]) -> bool:
    for resource_path in resource_paths:
        try:
            nltk.data.find(resource_path)
            return True
        except LookupError:
            continue
    return False


def ensure_nltk_resources(resources: Iterable[str] | None = None) -> list[str]:
    """Ensure the required NLTK resources exist locally without user interaction."""
    selected = list(resources) if resources is not None else list(RESOURCE_SPECS.keys())
    status_lines: list[str] = []

    for package_name in selected:
        if package_name not in RESOURCE_SPECS:
            raise KeyError(f"Unsupported NLTK resource: {package_name}")

        resource_paths = RESOURCE_SPECS[package_name]
        if _resource_is_available(resource_paths):
            status_lines.append(f"[OK] {package_name}")
            continue

        download_ok = nltk.download(package_name, quiet=True)
        if not download_ok:
            raise RuntimeError(f"Failed to download NLTK resource: {package_name}")

        if not _resource_is_available(resource_paths):
            raise RuntimeError(
                f"NLTK resource still unavailable after download: {package_name}"
            )

        status_lines.append(f"[DOWNLOADED] {package_name}")

    return status_lines
