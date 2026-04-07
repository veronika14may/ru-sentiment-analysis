import re
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import joblib
import config

_POSITIVE_WORDS = {
    "отлично", "хорошо", "супер", "прекрасно", "замечательно",
    "рекомендую", "доволен", "довольна", "понравилось", "быстро",
    "качественно", "удобно", "спасибо", "благодарю", "лучший",
    "великолепно", "шикарно", "идеально", "отличный", "классно",
    "восхитительно", "превосходно", "вкусно", "удобный", "надёжный",
}

_NEGATIVE_WORDS = {
    "плохо", "ужасно", "отстой", "кошмар", "разочарован", "разочарована",
    "не рекомендую", "брак", "сломан", "испорчен", "обман", "мусор",
    "дрянь", "жуть", "кошмарно", "отвратительно", "бракованный",
    "некачественный", "отвратительный", "ужасный", "плохой", "дорого",
}

_NEGATION_RE = re.compile(r"\b(не|нет|никогда|никак|нельзя|ни|без)\b")

class HandcraftedFeatures:
    def transform(self, texts):
        rows = []
        for text in texts:
            raw = str(text).lower()
            words = set(raw.split())
            pos = len(words & _POSITIVE_WORDS)
            neg = len(words & _NEGATIVE_WORDS)
            excl = raw.count("!")
            quest = raw.count("?")
            orig = str(text)
            caps_ratio = sum(1 for c in orig if c.isupper()) / (len(orig) + 1)
            word_list = raw.split()
            text_len = len(word_list)
            avg_wlen = float(np.mean([len(w) for w in word_list])) if word_list else 0.0
            neg_count = len(_NEGATION_RE.findall(raw))
            rows.append([
                pos, neg, pos - neg,
                excl, quest,
                caps_ratio, text_len, avg_wlen, neg_count,
            ])
        return np.array(rows, dtype=np.float32)

    @property
    def feature_names(self):
        return [
            "pos_words", "neg_words", "pos_minus_neg",
            "exclamations", "questions",
            "caps_ratio", "text_len", "avg_word_len", "negations",
        ]

class TFIDFFeatureExtractor:
    def __init__(self):
        self.word_vec = TfidfVectorizer(
            max_features=config.TFIDF_WORD_MAX_FEATURES,
            ngram_range=config.TFIDF_WORD_NGRAM_RANGE,
            min_df=config.TFIDF_WORD_MIN_DF,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.char_vec = TfidfVectorizer(
            max_features=config.TFIDF_CHAR_MAX_FEATURES,
            ngram_range=config.TFIDF_CHAR_NGRAM_RANGE,
            analyzer="char_wb",
            min_df=config.TFIDF_CHAR_MIN_DF,
            sublinear_tf=True,
        )

    def fit_transform(self, texts):
        w = self.word_vec.fit_transform(texts)
        c = self.char_vec.fit_transform(texts)
        return hstack([w, c], format="csr")

    def transform(self, texts):
        w = self.word_vec.transform(texts)
        c = self.char_vec.transform(texts)
        return hstack([w, c], format="csr")

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)

class SVDFeatureExtractor:
    def __init__(self):
        self.word_vec = TfidfVectorizer(
            max_features=config.TFIDF_WORD_MAX_FEATURES,
            ngram_range=(1, 2),
            min_df=config.TFIDF_WORD_MIN_DF,
            sublinear_tf=True,
        )
        self.svd = TruncatedSVD(
            n_components=config.SVD_N_COMPONENTS,
            random_state=config.RANDOM_STATE,
        )
        self.handcrafted = HandcraftedFeatures()

    def fit_transform(self, texts_preprocessed, texts_raw):
        tfidf = self.word_vec.fit_transform(texts_preprocessed)
        svd_feat = self.svd.fit_transform(tfidf)
        hc_feat = self.handcrafted.transform(texts_raw)
        return np.hstack([svd_feat, hc_feat])

    def transform(self, texts_preprocessed, texts_raw):
        tfidf = self.word_vec.transform(texts_preprocessed)
        svd_feat = self.svd.transform(tfidf)
        hc_feat = self.handcrafted.transform(texts_raw)
        return np.hstack([svd_feat, hc_feat])

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)