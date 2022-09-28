import faiss
import os
import numpy as np
from sentence_transformers import SentenceTransformer

class FaissIndexer():

    def __init__(self):
        self.embeddings = None
        self.dimension = None
        self.index = None

    def load_embeddings(self, location):
        # Load embeddings from folder
        embeddings = []
        files = os.listdir(location)
        files.sort()
        for file in files:
            with open(location + file, 'rb') as f:
                embeddings.append(np.load(f))
        self.embeddings = np.array(embeddings)
        self.dimension = self.embeddings.shape[1]

    def set_embeddings(self, embeddings):
        # Sets embeddings without loading them from folder
        self.embeddings = embeddings
        self.dimension = self.embeddings.shape[1]

    def create_index(self, index_type):
        if index_type == 'L2':
            self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(self.embeddings)

    def search(self, querys, k=5):
        if len(querys.shape) == 1:
            querys = np.array([querys])
        I, D = self.index.search(querys, k)
        return I, D

if __name__ == '__main__':
        faiss_index = FaissIndexer()
        faiss_index.load_embeddings(os.path.dirname(os.path.abspath(__file__)) + '/test_embeddings/')
        faiss_index.create_index('L2')
        encoder = SentenceTransformer('bert-base-nli-mean-tokens')
        encoded_sentences = encoder.encode(["Schneider 's mugging is relentless and his constant need to suddenly transpose himself into another character undermines the story 's continuity and progression ."])#, "One of the best movies i've ever seen", 'Robert de Niro'])
        D, I  = faiss_index.search(encoded_sentences, k=4)

        sentence_folder_name = os.path.dirname(os.path.abspath(__file__)) + '/test_sentences/'
        for indexes in I:
            for sentence_index in indexes:
                print(sentence_index)
                with open(f'{sentence_folder_name}sentence_{sentence_index}.txt', 'r') as f:
                    print(f.read())
            print('\n')
            print('-----------------')
            print('\n')
