"""
Tokenization primitives and string-processing utilities for HPO text.

Re-used from previous work of Feng et al., 2023
@article{Feng2023,
  title = {PhenoBERT: A Combined Deep Learning Method for Automated Recognition of Human Phenotype Ontology},
  volume = {20},
  ISSN = {2374-0043},
  url = {http://dx.doi.org/10.1109/TCBB.2022.3170301},
  DOI = {10.1109/tcbb.2022.3170301},
  number = {2},
  journal = {IEEE/ACM Transactions on Computational Biology and Bioinformatics},
  publisher = {Institute of Electrical and Electronics Engineers (IEEE)},
  author = {Feng,  Yuhao and Qi,  Lei and Tian,  Weidong},
  year = {2023},
  month = mar,
  pages = {1269-1277}
}
"""

import re
import unicodedata
from pathlib import Path

import nltk

# Before the ``from nltk.corpus`` line, and before anything below reads a corpus: ``PhraseItem``
# evaluates ``stopwords.words("english")`` in its **class body**, so a machine whose corpora are in
# somebody else's home directory fails at *import* with a LookupError rather than at first use.
# Importing this puts the repo's own vendored copy on ``nltk.data.path``. See ``hpo_extraction.ontology.nltk_data``.
from hpo_extraction.ontology import nltk_data as _nltk_data  # noqa: F401 - imported for its path side effect
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()

# Resolve resource paths relative to the repository root (three levels up from this file)
from hpo_extraction.paths import REPO_ROOT as _REPO_ROOT
_STOPWORDS_FILE = _REPO_ROOT / "resources" / "util" / "stopwords.txt"
_NUM2WORD_FILE = _REPO_ROOT / "resources" / "util" / "NUM.txt"


def strip_accents(s: str) -> str:
    """*s* without combining accent marks."""
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def processStr(string: str) -> list[str]:
    """Lower-case, accent-free word list of *string*, with hyphens, quotes and tabs turned into spaces."""
    string = re.sub(r"(?<=[A-Z])-(?=[\d])", "", string)
    string = strip_accents(string.lower())
    string = re.sub(r"[-_\"'\\\\\t\r\n\u2018\u2019]", " ", string)
    return string.strip().split()


def isNum(strings: str) -> bool:
    """True when *strings* parses as a number."""
    try:
        float(strings)
        return True
    except ValueError:
        return False


def containNum(strings: str) -> bool:
    """True when *strings* contains a digit."""
    return any(c.isdigit() for c in strings)


def getNum2Word(file_path: Path | str) -> dict[str, str]:
    """``{number: word}`` from a tab-separated file (``resources/util/NUM.txt``)."""
    num2word = {}
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            num2word[parts[0]] = parts[1]
    return num2word


def getStopWords(file_path: Path | str = _STOPWORDS_FILE) -> set[str]:
    """Stop words, one per line of *file_path*."""
    words = set()
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            words.add(line.strip())
    return words


def getSpliters() -> set[str]:
    """Tokens that split a phrase: conjunctions, wh-words, ``to`` and punctuation."""
    return set(
        [word for word, pos in nltk.pos_tag(stopwords.words("english")) if pos in {"CC", "WP", "TO", "WDT"}]
        + [",", ".", ":", ";", "(", ")", "[", "]", "/"]
    )


def getNegativeWords() -> set[str]:
    """Words that mark a negated or normal finding."""
    return {"no", "not", "none", "negative", "non", "never", "few", "lower", "fewer", "less", "barely", "normal"}


class WordItem:
    """Wrapper for an English word with position information."""

    lemma_dict: dict[str, str] = {}

    def __init__(self, text: str, start: int, end: int):
        self.text = text.lower()
        self.start = start
        self.end = end


class PhraseItem:
    """Wrapper for an English phrase with positional and simplification support."""

    Num2Word: dict[str, str] = getNum2Word(_NUM2WORD_FILE)
    StopWords: list[str] = stopwords.words("english")

    def __init__(self, word_items: list[WordItem]):
        self.word_items = word_items
        self.simple_items: list[WordItem] = []
        self.simplify()
        self.locs_set = set(i.start for i in word_items)
        self.start_loc = self.word_items[0].start
        self.end_loc = self.word_items[-1].end
        self.no_flag = False

    def simplify(self):
        """Fill ``simple_items``: numbers written as words, stop words and numerals removed."""
        for word_item in self.word_items:
            if word_item.text in PhraseItem.Num2Word:
                self.simple_items.append(
                    WordItem(PhraseItem.Num2Word[word_item.text], word_item.start, word_item.end)
                )
            elif word_item.text in PhraseItem.StopWords or isNum(word_item.text):
                continue
            else:
                self.simple_items.append(word_item)

    def toString(self) -> str:
        """The phrase's words joined by spaces."""
        return " ".join(i.text for i in self.word_items)

    def toSimpleString(self) -> str:
        """The simplified words joined by spaces."""
        return " ".join(i.text for i in self.simple_items)

    def include(self, phrase_item: "PhraseItem") -> bool:
        """True when one phrase's word positions contain the other's."""
        return self.locs_set.issubset(phrase_item.locs_set) or self.locs_set.issuperset(phrase_item.locs_set)

    def issubset(self, phrase_item: "PhraseItem") -> bool:
        """True when this phrase's word positions lie within *phrase_item*'s."""
        return self.locs_set.issubset(phrase_item.locs_set)

    def set_no_flag(self):
        """Mark the phrase as negated."""
        self.no_flag = True

    def __len__(self) -> int:
        return len(self.word_items)
