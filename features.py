"""Text preprocessing + feature building for job postings.

Shared by the notebook (training) and app.py (prediction) so a posting
pasted into the app is cleaned exactly like the training data.
"""
import re
from functools import lru_cache

import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer

try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except LookupError:  # NLTK data not downloaded -> fall back to sklearn's list
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    STOPWORDS = set(ENGLISH_STOP_WORDS)

TEXT_COLUMNS = ["title", "company_profile", "description", "requirements", "benefits"]
META_FEATURES = [
    "has_company_logo", "has_questions", "telecommuting",
    "has_company_profile", "has_salary", "has_requirements", "has_benefits",
    "log_word_count",
]

_stemmer = PorterStemmer()
_stem = lru_cache(maxsize=None)(_stemmer.stem)


def clean_text(text: str) -> str:
    """The NLP preprocessing pipeline, one step per line."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)          # fix glued sentences: "worldwide.Carefully"/"teamOur"
    text = text.lower()                                       # 1. lowercase
    text = re.sub(r"#(url|email|phone)_[^#]*#", " ", text)    # 2. remove anonymised URLs / emails / phones
    text = re.sub(r"<[^>]+>", " ", text)                      # 3. remove HTML tags
    text = text.encode("ascii", "ignore").decode()            # 4. remove emojis + non-English symbols
    text = re.sub(r"[^a-z\s]", " ", text)                     # 5. remove punctuation + numbers
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]  # 6. tokenize + drop stopwords
    return " ".join(_stem(w) for w in words)                  # 7. stemming: "managing" -> "manag"


def make_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw job-posting columns -> one cleaned text column + numeric meta features."""
    df = raw.copy()
    for col in TEXT_COLUMNS + ["salary_range"]:
        if col not in df:
            df[col] = ""
    df[TEXT_COLUMNS + ["salary_range"]] = df[TEXT_COLUMNS + ["salary_range"]].fillna("")

    out = pd.DataFrame(index=df.index)
    raw_text = df[TEXT_COLUMNS].agg(" ".join, axis=1)
    out["clean_text"] = raw_text.apply(clean_text)

    # Missing (or present) information is itself a signal: scammers skip the company profile and logo.
    for col in ["has_company_logo", "has_questions", "telecommuting"]:
        out[col] = df[col].fillna(0).astype(int) if col in df else 0
    out["has_company_profile"] = (df["company_profile"].str.strip() != "").astype(int)
    out["has_salary"] = (df["salary_range"].str.strip() != "").astype(int)
    out["has_requirements"] = (df["requirements"].str.strip() != "").astype(int)
    out["has_benefits"] = (df["benefits"].str.strip() != "").astype(int)
    out["log_word_count"] = np.log1p(raw_text.str.split().str.len())
    return out
