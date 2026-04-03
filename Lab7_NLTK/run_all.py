from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable

from src.exercises import run_all_exercises
from src.nltk_setup import ensure_nltk_resources
from src.section_01_intro import run as run_section_01
from src.section_02_search_word import run as run_section_02
from src.section_03_frequency_analysis import run as run_section_03
from src.section_04_word_selection import run as run_section_04
from src.section_05_ngrams_collocations import run as run_section_05
from src.section_06_web_text import run as run_section_06
from src.section_07_html_extraction import run as run_section_07
from src.section_08_sentiment import run as run_section_08
from src.utils import save_text


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_PATH = PROJECT_ROOT / "report_lab7.md"


def run_with_logging(name: str, runner: Callable[[], dict[str, object]]) -> dict[str, object]:
    print(f"[RUN] {name}")
    try:
        result = runner()
    except Exception as exc:
        result = {"status": "failed", "error": str(exc)}

    status = result.get("status", "success")
    if status == "success":
        print(f"[OK] {name}")
    else:
        print(f"[WARN] {name}: {result.get('error', 'see output file')}")
    return result


def build_report(
    setup_status: list[str],
    section_results: dict[str, dict[str, object]],
    exercise_results: list[dict[str, object]],
) -> str:
    generated_at = datetime.now().isoformat(timespec="seconds")
    fallback_sections = [
        section_name
        for section_name, result in section_results.items()
        if result.get("fallback_used")
    ]
    failed_sections = [
        section_name
        for section_name, result in section_results.items()
        if result.get("status") != "success"
    ]
    failed_exercises = [result["exercise"] for result in exercise_results if result.get("status") != "success"]

    section_08 = section_results["section_08"]
    section_06 = section_results["section_06"]
    section_07 = section_results["section_07"]

    lines = [
        "# Lab 7 Report",
        "",
        f"Generated at: {generated_at}",
        "",
        "## Environment Setup",
        "Automatic NLTK setup was executed at the start of the run.",
        *[f"- {status}" for status in setup_status],
        "",
        "## Section Summary",
        (
            f"- Section 01 matched the Gutenberg introduction task and loaded Macbeth "
            f"with {section_results['section_01'].get('word_count', 'n/a')} words."
        ),
        (
            f"- Section 02 searched the word 'Stage' and found "
            f"{section_results['section_02'].get('occurrences', 'n/a')} occurrences."
        ),
        (
            f"- Section 03 produced raw and cleaned frequency tables; the clean top token was "
            f"{section_results['section_03'].get('clean_top_10', [('n/a', 0)])[0]}."
        ),
        (
            f"- Section 04 selected long words and words containing 'ious'; counts were "
            f"{section_results['section_04'].get('long_word_count', 'n/a')} and "
            f"{section_results['section_04'].get('ious_word_count', 'n/a')}."
        ),
        (
            f"- Section 05 generated bigrams, trigrams, and PMI-based collocations from cleaned Macbeth tokens."
        ),
        (
            f"- Section 06 downloaded remote plain text from {section_06.get('source_url', 'n/a')} "
            f"with fallback used = {section_06.get('fallback_used', False)}."
        ),
        (
            f"- Section 07 extracted visible text from HTML source {section_07.get('source_url', 'n/a')} "
            f"with fallback used = {section_07.get('fallback_used', False)}."
        ),
        (
            f"- Section 08 trained a Naive Bayes sentiment classifier on movie_reviews with "
            f"accuracy = {section_08.get('accuracy', 0.0):.4f}."
        ),
        "",
        "## Exercises",
        f"All 13 exercises were implemented. Failed exercises: {failed_exercises or 'none'}.",
        "The outputs cover corpus listing, stopword handling, WordNet, tagsets, similarity, and names corpus tasks.",
        "",
        "## Assumptions and Fallbacks",
        "- Exercise 05 interprets 'ignore stopwords from the stopword list' as removing selected items from the default stopword list before filtering.",
        "- Exercise 06 uses NLTK WordNet for definitions/examples because it is stable and aligned with the course topic.",
        (
            f"- Sections using old web URLs may rely on public fallback URLs. Sections with fallback in this run: "
            f"{fallback_sections or 'none'}."
        ),
        (
            f"- Sections with runtime failure in this run: {failed_sections or 'none'}."
        ),
    ]

    return "\n".join(lines)


def main() -> None:
    print("[RUN] Ensuring NLTK resources")
    setup_status = ensure_nltk_resources()
    for status_line in setup_status:
        print(f"  {status_line}")

    section_results = {
        "section_01": run_with_logging("Section 01", run_section_01),
        "section_02": run_with_logging("Section 02", run_section_02),
        "section_03": run_with_logging("Section 03", run_section_03),
        "section_04": run_with_logging("Section 04", run_section_04),
        "section_05": run_with_logging("Section 05", run_section_05),
        "section_06": run_with_logging("Section 06", run_section_06),
        "section_07": run_with_logging("Section 07", run_section_07),
        "section_08": run_with_logging("Section 08", run_section_08),
    }

    print("[RUN] Exercises 01-13")
    exercise_results = run_all_exercises()
    success_count = sum(1 for result in exercise_results if result.get("status") == "success")
    print(f"[OK] Exercises completed: {success_count}/{len(exercise_results)}")

    report_text = build_report(setup_status, section_results, exercise_results)
    save_text(REPORT_PATH, report_text)
    print(f"[OK] Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
