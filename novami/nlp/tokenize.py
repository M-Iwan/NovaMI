from typing import Optional, Pattern

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer


def tokenize_document(document: str, lemmatize: bool = True, stem: bool = False, chars: Optional[str] = "alpha",
                      re_pattern: Optional[Pattern[str]] = None):
    """
    Parameters
    ----------
    document : str
        Input text to tokenize.
    lemmatize : bool, default=True
        Whether to lemmatize tokens.
    stem : bool, default=False
        Whether to stem tokens.
    chars : {'alpha', 'alnum', 'ascii'} or None, default='alpha'
        Character filter to apply to tokens. If None, no built-in character filtering is applied.
    re_pattern : re.Pattern, optional
        Compiled regex pattern used to keep matching tokens only (using re.search).

    Returns
    -------
    list of str
        Processed tokens or generated n-grams.
    """
    stop_words = set(stopwords.words("english"))

    tokens = [
        token.lower()
        for sent in sent_tokenize(document)
        for token in word_tokenize(sent)
    ]

    tokens = [token for token in tokens if token not in stop_words]

    if chars is not None:
        if chars == "alpha":
            tokens = [token for token in tokens if token.isalpha()]
        elif chars == "alnum":
            tokens = [token for token in tokens if token.isalnum()]
        elif chars == "ascii":
            tokens = [token for token in tokens if token.isascii()]
        else:
            raise ValueError("chars must be one of {'alpha', 'alnum', 'ascii'} or None")

    if re_pattern is not None:
        tokens = [token for token in tokens if re_pattern.search(token)]

    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

    if stem:
        stemmer = SnowballStemmer(language="english")
        tokens = [stemmer.stem(token) for token in tokens]

    return tokens