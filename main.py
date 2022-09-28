import os

from sentence_transformers import SentenceTransformer, util

import config
from data import MSMarcoDocs
import encoding
from model import Model
import preprocessing


def main():
    print("=" * 10)
    print("Running in stage:", config.stage)
    print("=" * 10)

    # Initialize MS Marco
    ms_marco_docs = MSMarcoDocs()
    queries = ms_marco_docs.get_queries()
    docs = ms_marco_docs.get_docs()

    if config.stage == "DEV":
        docs = docs.train_test_split(train_size=10000, test_size=1, shuffle=True)[
            "train"
        ]

    docs = docs.map(
        preprocessing.doc_to_passages,
        num_proc=os.cpu_count(),
        fn_kwargs={
            "passage_size": config.passage_size,
            "tokenization_method": config.tokenization_method,
            "prepend_title_to_passage": config.prepend_title_to_passage,
        },
        keep_in_memory=config.use_cache,
    )

    # Initialize model to compute passage embeddings
    model = SentenceTransformer("sentence-transformers/msmarco-distilbert-dot-v5")

    # Compute document embeddings
    docs = docs.map(
        encoding.encode_doc,
        fn_kwargs={
            "encode_fn": model.encode,
            "method": "mean_passage_embeddings",
        },
        keep_in_memory=config.use_cache,
    )

    docs.add_faiss_index(column="embedding")
    docs.save_faiss_index("embeddings", "./data/ms-marco/document-embeddings.faiss")

    ##

    # example_doc = [
    #    "Around 9 Million people live in London",
    #    "London is known for its financial district",
    #    "Bangkok is the capital of Thailand",
    #    "The new macbook M2 line is the best of the best",
    # ]

    # example_embeddings = model.encode_passages(example_doc)
    # model.cache_embedding(example_embeddings, "doc_0")

    ## (3) Build FAISS index
    # faiss_index = FaissIndexer()
    # faiss_index.set_embeddings(example_embeddings)
    # faiss_index.create_index("L2")

    ## Some example cases using the example docs
    # example_sentence = "What is the capital of Thailand"
    # example_query = model.encode_query(example_sentence)
    # D, I = faiss_index.search(example_query, k=1)
    # print(example_sentence)
    # print(example_doc[I[0][0]])
    # print("\n")

    # example_sentence = "I live in the capital of England"
    # example_query = model.encode_query(example_sentence)
    # D, I = faiss_index.search(example_query, k=4)
    # ranking = [example_doc[I[0][i]] for i in I[0]]
    # print(example_sentence)
    # print(ranking)

    # (4) Compute score on dev set
    raise NotImplementedError


if __name__ == "__main__":
    main()
