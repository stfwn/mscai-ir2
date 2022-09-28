import os
from typing import Callable
from faiss_indexer import FaissIndexer

from model import Model

from data import MSMarcoDocs
import config


# THIJS: keeping this out of main; maybe consider making this it's
#        own file when we make it smarter

# Chunk docs into passages
# TODO: Consider making this smarter.
def chunk_doc_into_passages(doc: dict):
    try:
        words = doc["body"].split(" ")
    except AttributeError:
        # Some bodies are None, so they have no .split
        words = []
    doc["passages"] = [
        " ".join(words[i : i + config.tokens_per_passage])
        for i in range(0, len(words), config.tokens_per_passage)
    ]
    return doc

def main():
    print("=" * 10)
    print("Running in stage:", config.stage)
    print("=" * 10)

    # (1) Initialize MS Marco
    ms_marco_docs = MSMarcoDocs()
    queries = ms_marco_docs.get_queries()
    docs = ms_marco_docs.get_docs()

    if config.stage == "DEV":
        debug = docs.train_test_split(train_size=10000, test_size=1, shuffle=True)[
            "train"
        ].map(
            chunk_doc_into_passages,
            num_proc=os.cpu_count(),
            keep_in_memory=True
        )


    # (2) Initialize model to compute passage embeddings
    model = Model(cache_path='cache/passage_embeddings/example/')

    example_doc = [
                    "Around 9 Million people live in London", 
                    "London is known for its financial district",
                    "Bangkok is the capital of Thailand",
                    "The new macbook M2 line is the best of the best"
                    ]

    example_embeddings = model.encode_passages(example_doc)
    model.cache_embedding(example_embeddings, 'doc_0')


    # (3) Build FAISS index
    faiss_index = FaissIndexer()
    faiss_index.set_embeddings(example_embeddings)
    faiss_index.create_index("L2")

    # Some example cases using the example docs
    example_sentence = 'What is the capital of Thailand'
    example_query = model.encode_query(example_sentence)
    D, I = faiss_index.search(example_query, k=1)
    print(example_sentence)
    print(example_doc[I[0][0]])
    print('\n')

    example_sentence = "I live in the capital of England"
    example_query = model.encode_query(example_sentence)
    D, I = faiss_index.search(example_query, k=4)
    ranking = [example_doc[I[0][i]] for i in I[0]]
    print(example_sentence)
    print(ranking)

    # (4) Compute score on dev set
    raise NotImplementedError


if __name__ == "__main__":
    main()
