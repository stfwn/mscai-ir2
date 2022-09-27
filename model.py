from sentence_transformers import SentenceTransformer, util
import numpy as np
import os

class Model:

    def __init__(self, cache_path='cache/'):
        self.model = SentenceTransformer('sentence-transformers/msmarco-distilbert-dot-v5')
        cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), cache_path)
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        self.cache_path = cache_path


    def encode_passages(self, passages):
        #np.ndarry shape(num_passages, 768)
        return self.model.encode(passages)

    def encode_query(self, query):
        #np.ndarry shape(768,)
        return self.model.encode(query)

    def score(self, passages, query_emb, passage_emb, mode='dot'):
        #list: [ tuple: ( str: doc, float: score) ]
        scores = []
        if mode == 'dot':
            scores = util.dot_score(query_emb, passage_emb)[0].cpu().tolist()
        doc_score_pairs = list(zip(passages, scores))
        return sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

    def cache_embedding(self, emb, id):
        # using np.save and np.load will be very efficient for np.ndarrays
        np.save(self.cache_path+id, emb)