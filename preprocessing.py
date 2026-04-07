import re
import functools
import pymorphy3
import nltk
from nltk.corpus import stopwords
import config

nltk.download("stopwords", quiet=True)

_morph = pymorphy3.MorphAnalyzer()

RUSSIAN_STOPWORDS = set(stopwords.words("russian"))

EXTRA_STOPWORDS = {
    "это", "всё", "все", "который", "свой", "такой", "также",
    "вообще", "именно", "просто", "ещё", "уже", "тоже", "вот",
    "раз", "там", "тут", "где", "когда", "если", "так", "чтобы",
}

STOPWORDS = RUSSIAN_STOPWORDS | EXTRA_STOPWORDS

_RE_NON_ALPHA = re.compile(r"[^а-яёa-z\s]")
_RE_SPACES = re.compile(r"\s+")

def clean_text(text):
    text = str(text).lower()
    text = _RE_NON_ALPHA.sub(" ", text)
    text = _RE_SPACES.sub(" ", text).strip()
    return text

def tokenize(text):
    return text.split()

@functools.lru_cache(maxsize=config.CACHE_SIZE)
def _lemmatize_word(word):
    return _morph.parse(word)[0].normal_form

def lemmatize_tokens(tokens):
    return [_lemmatize_word(t) for t in tokens]

def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

def preprocess(text, lemmatize=True):
    text = clean_text(text)
    tokens = tokenize(text)
    if lemmatize:
        tokens = lemmatize_tokens(tokens)
    tokens = remove_stopwords(tokens)
    return " ".join(tokens)

def preprocess_batch(texts, lemmatize=True):
    result = []
    for i, t in enumerate(texts, 1):
        result.append(preprocess(t, lemmatize=lemmatize))
    return result