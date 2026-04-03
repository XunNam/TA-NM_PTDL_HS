# Lab7_NLTK

This project implements Lab 7, "Text Data Analysis with NLTK", for the course "Nhap mon Phan tich du lieu va Hoc sau".

## Project Structure

```text
Lab7_NLTK/
|-- README.md
|-- requirements_lab7.txt
|-- run_all.py
|-- report_lab7.md
|-- data/
|   `-- downloaded/
|-- outputs/
|   |-- section_01/
|   |-- section_02/
|   |-- section_03/
|   |-- section_04/
|   |-- section_05/
|   |-- section_06/
|   |-- section_07/
|   |-- section_08/
|   `-- exercises/
|-- src/
|   |-- __init__.py
|   |-- nltk_setup.py
|   |-- utils.py
|   |-- section_01_intro.py
|   |-- section_02_search_word.py
|   |-- section_03_frequency_analysis.py
|   |-- section_04_word_selection.py
|   |-- section_05_ngrams_collocations.py
|   |-- section_06_web_text.py
|   |-- section_07_html_extraction.py
|   |-- section_08_sentiment.py
|   `-- exercises.py
`-- notebooks/
```

## Environment

Use the Linux Anaconda environment named `PTDLHS`.

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate PTDLHS
which python
python --version
conda env list
```

Expected interpreter:

```text
/home/buitheanh/anaconda3/envs/PTDLHS/bin/python
```

## Install Dependencies

```bash
pip install -r requirements_lab7.txt
```

## Run Everything

```bash
python run_all.py
```

## Scripts Overview

- `src/nltk_setup.py`: downloads required NLTK corpora and models with `nltk.download(..., quiet=True)`.
- `src/utils.py`: shared helpers for directories, file saving, HTTP requests, output formatting, and random seeds.
- `src/section_01_intro.py`: introduces NLTK and the Gutenberg corpus with Macbeth.
- `src/section_02_search_word.py`: uses `nltk.Text` to search the word `Stage`.
- `src/section_03_frequency_analysis.py`: computes frequency tables before and after cleaning.
- `src/section_04_word_selection.py`: selects words by length and substring rules.
- `src/section_05_ngrams_collocations.py`: builds bigrams, trigrams, and collocations.
- `src/section_06_web_text.py`: downloads a remote public text file and tokenizes it with NLTK.
- `src/section_07_html_extraction.py`: downloads HTML, extracts visible text with BeautifulSoup, and tokenizes it.
- `src/section_08_sentiment.py`: trains a Naive Bayes sentiment classifier on `movie_reviews`.
- `src/exercises.py`: implements all 13 exercises as separate functions and writes outputs to `outputs/exercises/`.
- `run_all.py`: runs all sections and exercises, then writes `report_lab7.md`.

## Notes

- Old course URLs may no longer be available. Sections 06 and 07 try the original URLs first and then use public fallback URLs if needed.
- Exercise 05 is interpreted as: remove selected words from the default stopword list, then filter the sample text with the modified stopword set.
- Exercise 06 is implemented with NLTK WordNet for stable definitions and example sentences instead of scraping external websites.
- All outputs are plain text files under `outputs/`, and downloaded web data is stored in `data/downloaded/`.
