# Lab 7 Report

Generated at: 2026-03-27T01:29:02

## Environment Setup
Automatic NLTK setup was executed at the start of the run.
- [OK] gutenberg
- [OK] punkt
- [OK] punkt_tab
- [OK] stopwords
- [OK] movie_reviews
- [OK] names
- [OK] wordnet
- [OK] omw-1.4
- [OK] tagsets
- [OK] averaged_perceptron_tagger

## Section Summary
- Section 01 matched the Gutenberg introduction task and loaded Macbeth with 23140 words.
- Section 02 searched the word 'Stage' and found 3 occurrences.
- Section 03 produced raw and cleaned frequency tables; the clean top token was ('macb', 137).
- Section 04 selected long words and words containing 'ious'; counts were 11 and 18.
- Section 05 generated bigrams, trigrams, and PMI-based collocations from cleaned Macbeth tokens.
- Section 06 downloaded remote plain text from https://www.gutenberg.org/files/2554/2554-0.txt with fallback used = False.
- Section 07 extracted visible text from HTML source http://news.bbc.co.uk/2/hi/health/2284783.stm with fallback used = False.
- Section 08 trained a Naive Bayes sentiment classifier on movie_reviews with accuracy = 0.7120.

## Exercises
All 13 exercises were implemented. Failed exercises: none.
The outputs cover corpus listing, stopword handling, WordNet, tagsets, similarity, and names corpus tasks.

## Assumptions and Fallbacks
- Exercise 05 interprets 'ignore stopwords from the stopword list' as removing selected items from the default stopword list before filtering.
- Exercise 06 uses NLTK WordNet for definitions/examples because it is stable and aligned with the course topic.
- Sections using old web URLs may rely on public fallback URLs. Sections with fallback in this run: none.
- Sections with runtime failure in this run: none.