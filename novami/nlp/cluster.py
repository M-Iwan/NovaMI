from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
import pyLDAvis
import pyLDAvis.gensim


def lda_cluster(tokenized_documents, no_below: int = 3, no_above: float = 0.7, num_topics: int = 64,
                chunksize: int = 1024, passes: int = 16, random_state: int = 42):

    dictionary = corpora.Dictionary(tokenized_documents)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)

    corpus = [dictionary.doc2bow(document) for document in tokenized_documents]

    lda_model = LdaModel(corpus = corpus, id2word=dictionary, num_topics=num_topics, chunksize=chunksize,
                         passes=passes, alpha='auto', per_word_topics=True, random_state=random_state)

    coherence_model = CoherenceModel(model=lda_model, texts=tokenized_documents, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()

    topics = lda_model.print_topics(num_topics=num_topics, num_words=16)
    lda_display = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary, sort_topics=False)

    results = {'dictionary': dictionary,
               'corpus': corpus,
               'lda_model': lda_model,
               'coherence_model': coherence_model,
               'coherence_score': coherence_score,
               'topics': topics,
               'display': lda_display}

    return results