from __future__ import annotations

import re
import random
from collections import Counter
from pathlib import Path
from typing import Callable

import nltk
from nltk.corpus import names, stopwords, wordnet as wn
from nltk.corpus.util import LazyCorpusLoader

from src.nltk_setup import ensure_nltk_resources
from src.utils import OUTPUT_DIR, save_text, setup_random_seed


EXERCISES_OUTPUT_DIR = OUTPUT_DIR / "exercises"


def _save_exercise_result(filename: str, lines: list[str]) -> Path:
    output_path = EXERCISES_OUTPUT_DIR / filename
    save_text(output_path, "\n".join(lines))
    return output_path


def _sample_tokens(tokens: list[str], size: int = 15) -> str:
    return ", ".join(tokens[:size])


def _format_tag_details(tag: str, tagset_map: dict[str, tuple[str, str]]) -> str:
    definition, examples = tagset_map[tag]
    return f"{tag}: {definition}\nExamples: {examples}"


def exercise_01_list_corpora() -> dict[str, object]:
    """Exercise 01: list accessible NLTK corpora and common corpus names.

    Example:
        exercise_01_list_corpora()
    """

    ensure_nltk_resources(["gutenberg", "stopwords", "movie_reviews", "names", "wordnet"])

    discovered = []
    for attribute_name in dir(nltk.corpus):
        if attribute_name.startswith("_"):
            continue
        try:
            attribute = getattr(nltk.corpus, attribute_name)
        except Exception:
            continue
        if isinstance(attribute, LazyCorpusLoader):
            discovered.append(attribute_name)

    common_corpora = [
        "gutenberg",
        "stopwords",
        "movie_reviews",
        "names",
        "wordnet",
        "brown",
        "reuters",
        "inaugural",
        "state_union",
        "webtext",
        "genesis",
        "abc",
    ]

    accessibility_checks: dict[str, Callable[[], object]] = {
        "gutenberg": lambda: nltk.corpus.gutenberg.fileids(),
        "stopwords": lambda: nltk.corpus.stopwords.fileids(),
        "movie_reviews": lambda: nltk.corpus.movie_reviews.fileids(),
        "names": lambda: nltk.corpus.names.fileids(),
        "wordnet": lambda: wn.synsets("computer"),
        "brown": lambda: nltk.corpus.brown.fileids(),
        "reuters": lambda: nltk.corpus.reuters.fileids(),
        "inaugural": lambda: nltk.corpus.inaugural.fileids(),
        "state_union": lambda: nltk.corpus.state_union.fileids(),
        "webtext": lambda: nltk.corpus.webtext.fileids(),
        "genesis": lambda: nltk.corpus.genesis.fileids(),
        "abc": lambda: nltk.corpus.abc.fileids(),
    }

    accessible: list[str] = []
    unavailable: list[str] = []
    for corpus_name in common_corpora:
        try:
            accessibility_checks[corpus_name]()
            accessible.append(corpus_name)
        except Exception:
            unavailable.append(corpus_name)

    lines = [
        "Exercise 01 - List NLTK corpora",
        "Demo call: exercise_01_list_corpora()",
        "",
        f"Discovered LazyCorpusLoader entries: {len(discovered)}",
        f"Preview: {_sample_tokens(sorted(discovered), 40)}",
        "",
        "Accessible common corpora:",
        *accessible,
        "",
        "Unavailable common corpora in this environment:",
        *(unavailable or ["(None in this list.)"]),
    ]
    output_path = _save_exercise_result("ex01_list_corpora.txt", lines)
    return {"status": "success", "output_path": str(output_path), "discovered_count": len(discovered)}


def exercise_02_stopword_languages() -> dict[str, object]:
    """Exercise 02: list the languages available in the NLTK stopwords corpus.

    Example:
        exercise_02_stopword_languages()
    """

    ensure_nltk_resources(["stopwords"])
    languages = stopwords.fileids()

    lines = [
        "Exercise 02 - Stopword languages",
        "Demo call: exercise_02_stopword_languages()",
        "",
        f"Number of languages: {len(languages)}",
        "Languages:",
        *languages,
    ]
    output_path = _save_exercise_result("ex02_stopword_languages.txt", lines)
    return {"status": "success", "output_path": str(output_path), "language_count": len(languages)}


def exercise_03_check_stopwords() -> dict[str, object]:
    """Exercise 03: inspect stopwords for multiple languages.

    Example:
        exercise_03_check_stopwords()
    """

    ensure_nltk_resources(["stopwords"])
    languages = ["english", "french", "german", "spanish"]
    lines = [
        "Exercise 03 - Check multilingual stopwords",
        "Demo call: exercise_03_check_stopwords()",
        "",
    ]

    summary: dict[str, int] = {}
    for language in languages:
        words = stopwords.words(language)
        summary[language] = len(words)
        lines.extend(
            [
                f"Language: {language}",
                f"Stopword count: {len(words)}",
                f"Preview: {_sample_tokens(words, 20)}",
                "",
            ]
        )

    output_path = _save_exercise_result("ex03_check_stopwords.txt", lines)
    return {"status": "success", "output_path": str(output_path), "counts": summary}


def remove_stopwords_from_text(text: str, language: str = "english") -> list[str]:
    """Tokenize text, lowercase it, and remove NLTK stopwords for a language."""
    ensure_nltk_resources(["punkt", "punkt_tab", "stopwords"])
    stopword_set = set(stopwords.words(language))
    return [
        token.lower()
        for token in nltk.word_tokenize(text)
        if token.lower() not in stopword_set and any(character.isalpha() for character in token)
    ]


def exercise_04_remove_stopwords() -> dict[str, object]:
    """Exercise 04: remove stopwords from a sample English text.

    Example:
        exercise_04_remove_stopwords()
    """

    sample_text = (
        "NLTK makes it easier to process text, remove stopwords, and inspect "
        "meaningful words in a sentence."
    )
    filtered_tokens = remove_stopwords_from_text(sample_text)

    lines = [
        "Exercise 04 - Remove stopwords from input text",
        "Demo call: exercise_04_remove_stopwords()",
        "",
        f"Original text: {sample_text}",
        f"Filtered tokens: {filtered_tokens}",
    ]
    output_path = _save_exercise_result("ex04_remove_stopwords.txt", lines)
    return {"status": "success", "output_path": str(output_path), "filtered_count": len(filtered_tokens)}


def exercise_05_custom_stopwords() -> dict[str, object]:
    """Exercise 05: customize the stopword list before filtering text.

    Example:
        exercise_05_custom_stopwords()
    """

    ensure_nltk_resources(["punkt", "punkt_tab", "stopwords"])
    sample_text = (
        "We do not ignore the words no and not because they can change the "
        "meaning of a sentence from positive to negative."
    )
    custom_stopword_set = set(stopwords.words("english"))
    removed_from_stopwords = {"from", "not", "no"}
    custom_stopword_set.difference_update(removed_from_stopwords)

    filtered_tokens = [
        token.lower()
        for token in nltk.word_tokenize(sample_text)
        if token.lower() not in custom_stopword_set and any(character.isalpha() for character in token)
    ]

    lines = [
        "Exercise 05 - Custom stopword filtering",
        "Demo call: exercise_05_custom_stopwords()",
        "",
        "Interpretation used in this project:",
        "Remove selected items from the default English stopword list, then filter the text.",
        f"Removed from stopword list: {sorted(removed_from_stopwords)}",
        f"Original text: {sample_text}",
        f"Filtered tokens: {filtered_tokens}",
    ]
    output_path = _save_exercise_result("ex05_custom_stopwords.txt", lines)
    return {"status": "success", "output_path": str(output_path), "kept_tokens": len(filtered_tokens)}


def exercise_06_wordnet_definition_example() -> dict[str, object]:
    """Exercise 06: list WordNet definitions and example sentences for a word.

    Example:
        exercise_06_wordnet_definition_example()
    """

    ensure_nltk_resources(["wordnet", "omw-1.4"])
    sample_word = "computer"
    synsets = wn.synsets(sample_word)

    lines = [
        "Exercise 06 - WordNet definitions and examples",
        "Demo call: exercise_06_wordnet_definition_example()",
        "",
        "Stable interpretation used in this project:",
        "Use NLTK WordNet instead of scraping external sites.",
        f"Target word: {sample_word}",
        f"Synset count: {len(synsets)}",
        "",
    ]

    for index, synset in enumerate(synsets, start=1):
        examples = synset.examples() or ["(No example sentence available.)"]
        lines.extend(
            [
                f"{index}. Synset: {synset.name()}",
                f"   Definition: {synset.definition()}",
                f"   Examples: {examples}",
            ]
        )

    output_path = _save_exercise_result("ex06_wordnet_definition_example.txt", lines)
    return {"status": "success", "output_path": str(output_path), "synset_count": len(synsets)}


def exercise_07_synonyms_antonyms() -> dict[str, object]:
    """Exercise 07: find synonym and antonym sets for a sample word.

    Example:
        exercise_07_synonyms_antonyms()
    """

    ensure_nltk_resources(["wordnet", "omw-1.4"])
    target_word = "good"
    synonyms: set[str] = set()
    antonyms: set[str] = set()

    for synset in wn.synsets(target_word):
        for lemma in synset.lemmas():
            synonyms.add(lemma.name().replace("_", " "))
            for antonym in lemma.antonyms():
                antonyms.add(antonym.name().replace("_", " "))

    lines = [
        "Exercise 07 - Synonyms and antonyms",
        "Demo call: exercise_07_synonyms_antonyms()",
        "",
        f"Target word: {target_word}",
        f"Synonyms ({len(synonyms)}): {sorted(synonyms)}",
        f"Antonyms ({len(antonyms)}): {sorted(antonyms)}",
    ]
    output_path = _save_exercise_result("ex07_synonyms_antonyms.txt", lines)
    return {
        "status": "success",
        "output_path": str(output_path),
        "synonym_count": len(synonyms),
        "antonym_count": len(antonyms),
    }


def exercise_08_tagsets_regex() -> dict[str, object]:
    """Exercise 08: inspect POS tags and filter tag names with a regex.

    Example:
        exercise_08_tagsets_regex()
    """

    ensure_nltk_resources(["tagsets"])
    tagset_map = nltk.data.load("help/tagsets/upenn_tagset.pickle")
    regex_pattern = r"^(NN|VB|JJ)"
    matching_tags = sorted(tag for tag in tagset_map if re.search(regex_pattern, tag))

    lines = [
        "Exercise 08 - Tagsets and regex filtering",
        "Demo call: exercise_08_tagsets_regex()",
        "",
        f"Total tags in the loaded map: {len(tagset_map)}",
        f"Regex pattern: {regex_pattern}",
        f"Matching tags: {matching_tags}",
        "",
        "Details for NN:",
        _format_tag_details("NN", tagset_map),
        "",
        "Details for VB:",
        _format_tag_details("VB", tagset_map),
        "",
        "Details for JJ:",
        _format_tag_details("JJ", tagset_map),
    ]
    output_path = _save_exercise_result("ex08_tagsets_regex.txt", lines)
    return {"status": "success", "output_path": str(output_path), "matching_tag_count": len(matching_tags)}


def exercise_09_noun_similarity() -> dict[str, object]:
    """Exercise 09: compare similarity between noun synsets.

    Example:
        exercise_09_noun_similarity()
    """

    ensure_nltk_resources(["wordnet", "omw-1.4"])
    comparisons = [
        (wn.synset("car.n.01"), wn.synset("automobile.n.01")),
        (wn.synset("car.n.01"), wn.synset("tree.n.01")),
    ]

    lines = [
        "Exercise 09 - Noun similarity",
        "Demo call: exercise_09_noun_similarity()",
        "",
    ]

    for left_synset, right_synset in comparisons:
        lines.extend(
            [
                f"Pair: {left_synset.name()} vs {right_synset.name()}",
                f"Path similarity: {left_synset.path_similarity(right_synset)}",
                f"Wu-Palmer similarity: {left_synset.wup_similarity(right_synset)}",
                "",
            ]
        )

    output_path = _save_exercise_result("ex09_noun_similarity.txt", lines)
    return {"status": "success", "output_path": str(output_path), "comparison_count": len(comparisons)}


def exercise_10_verb_similarity() -> dict[str, object]:
    """Exercise 10: compare similarity between verb synsets.

    Example:
        exercise_10_verb_similarity()
    """

    ensure_nltk_resources(["wordnet", "omw-1.4"])
    left_synset = wn.synset("run.v.01")
    right_synset = wn.synset("walk.v.01")

    lines = [
        "Exercise 10 - Verb similarity",
        "Demo call: exercise_10_verb_similarity()",
        "",
        f"Pair: {left_synset.name()} vs {right_synset.name()}",
        f"Path similarity: {left_synset.path_similarity(right_synset)}",
        f"Wu-Palmer similarity: {left_synset.wup_similarity(right_synset)}",
    ]
    output_path = _save_exercise_result("ex10_verb_similarity.txt", lines)
    return {"status": "success", "output_path": str(output_path)}


def exercise_11_names_count() -> dict[str, object]:
    """Exercise 11: count male and female names from the names corpus.

    Example:
        exercise_11_names_count()
    """

    ensure_nltk_resources(["names"])
    male_names = names.words("male.txt")
    female_names = names.words("female.txt")

    lines = [
        "Exercise 11 - Names corpus counts",
        "Demo call: exercise_11_names_count()",
        "",
        f"Male name count: {len(male_names)}",
        f"Female name count: {len(female_names)}",
        f"First 10 male names: {male_names[:10]}",
        f"First 10 female names: {female_names[:10]}",
    ]
    output_path = _save_exercise_result("ex11_names_count.txt", lines)
    return {
        "status": "success",
        "output_path": str(output_path),
        "male_count": len(male_names),
        "female_count": len(female_names),
    }


def exercise_12_random_labeled_names() -> dict[str, object]:
    """Exercise 12: shuffle labeled names and print the first 15 items.

    Example:
        exercise_12_random_labeled_names()
    """

    ensure_nltk_resources(["names"])
    seed = setup_random_seed(42)
    labeled_names = ([(name, "male") for name in names.words("male.txt")] + [
        (name, "female") for name in names.words("female.txt")
    ])
    shuffled_names = list(labeled_names)
    random.shuffle(shuffled_names)

    lines = [
        "Exercise 12 - Random labeled names",
        "Demo call: exercise_12_random_labeled_names()",
        "",
        f"Random seed: {seed}",
        "First 15 shuffled (name, label) pairs:",
    ]
    lines.extend(str(item) for item in shuffled_names[:15])

    output_path = _save_exercise_result("ex12_random_labeled_names.txt", lines)
    return {"status": "success", "output_path": str(output_path), "preview_count": 15}


def exercise_13_last_letter_labels() -> dict[str, object]:
    """Exercise 13: build (last_letter, gender_label) pairs from names.

    Example:
        exercise_13_last_letter_labels()
    """

    ensure_nltk_resources(["names"])
    labeled_names = ([(name, "male") for name in names.words("male.txt")] + [
        (name, "female") for name in names.words("female.txt")
    ])
    last_letter_labels = [(name[-1].lower(), label) for name, label in labeled_names if name]

    male_counter = Counter(letter for letter, label in last_letter_labels if label == "male")
    female_counter = Counter(letter for letter, label in last_letter_labels if label == "female")

    lines = [
        "Exercise 13 - Last-letter labels",
        "Demo call: exercise_13_last_letter_labels()",
        "",
        f"Total pairs: {len(last_letter_labels)}",
        f"Preview: {last_letter_labels[:20]}",
        "",
        f"Top male last letters: {male_counter.most_common(10)}",
        f"Top female last letters: {female_counter.most_common(10)}",
    ]
    output_path = _save_exercise_result("ex13_last_letter_labels.txt", lines)
    return {"status": "success", "output_path": str(output_path), "pair_count": len(last_letter_labels)}


EXERCISE_RUNNERS = [
    exercise_01_list_corpora,
    exercise_02_stopword_languages,
    exercise_03_check_stopwords,
    exercise_04_remove_stopwords,
    exercise_05_custom_stopwords,
    exercise_06_wordnet_definition_example,
    exercise_07_synonyms_antonyms,
    exercise_08_tagsets_regex,
    exercise_09_noun_similarity,
    exercise_10_verb_similarity,
    exercise_11_names_count,
    exercise_12_random_labeled_names,
    exercise_13_last_letter_labels,
]


def run_all_exercises() -> list[dict[str, object]]:
    """Run all 13 Lab 7 exercises and return per-exercise results."""
    results: list[dict[str, object]] = []
    for runner in EXERCISE_RUNNERS:
        try:
            result = runner()
        except Exception as exc:
            output_path = _save_exercise_result(
                f"{runner.__name__}.error.txt",
                [
                    f"Exercise runner failed: {runner.__name__}",
                    str(exc),
                ],
            )
            result = {
                "status": "failed",
                "output_path": str(output_path),
                "exercise": runner.__name__,
                "error": str(exc),
            }
        result.setdefault("exercise", runner.__name__)
        results.append(result)
    return results


if __name__ == "__main__":
    run_all_exercises()
