from rank_bm25 import BM25Okapi

class BM25Helper:
    def __init__(self, corpus_texts):
        tokenized_corpus = [text.split() for text in corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.corpus_texts = corpus_texts

    def get_scores(self, query):
        return self.bm25.get_scores(query.split())
