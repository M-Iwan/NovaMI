from typing import List

from nltk import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer


def tokenize_document(document: str, lemmatize: bool = True, stem: bool = False, n_grams: List = None):

    stop_words = set(stopwords.words('english'))

    tokens = [word_tokenize(sent) for sent in sent_tokenize(document)]
    tokens = [token.lower() for sublist in tokens for token in sublist]
    tokens = [token for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in stop_words]

    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

    if stem:  # check an alternative stemmer
        stemmer = SnowballStemmer(language='english')
        tokens = [stemmer.stem(token) for token in tokens]

    if n_grams is None:
        return tokens

    n_gram_tokens = []
    for n in n_grams:
        n_gram_tokens.extend(ngrams(tokens, n))

    n_gram_tokens = [' '.join(tks) for tks in n_gram_tokens]

    return n_gram_tokens
