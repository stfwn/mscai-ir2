from typing import Callable
import os

from sentence_transformers import SentenceTransformer, util
from datasets import Dataset

import config
from data import MSMarcoDocs
import encoding
import preprocessing


def main():
    print("=" * 10)
    print("Running in stage:", config.stage)
    print("=" * 10)

    # Initialize MS Marco
    ms_marco_docs = MSMarcoDocs()
    docs = ms_marco_docs.get_docs()
    queries = ms_marco_docs.get_queries()

    if config.stage == "DEV":
        docs = docs.train_test_split(train_size=10, test_size=1, shuffle=False)["train"]
        queries["train"] = queries["train"].train_test_split(
            train_size=50, test_size=1, shuffle=False
        )["train"]
        queries["dev"] = queries["dev"].train_test_split(
            train_size=50, test_size=1, shuffle=False
        )["train"]

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

    try:
        docs.load_faiss_index("embedding", "./data/ms-marco/document-embeddings.faiss")
    except:
        docs.add_faiss_index(column="embedding")
        docs.save_faiss_index("embedding", "./data/ms-marco/document-embeddings.faiss")

    def rank(query: dict, docs: Dataset, encode_fn: Callable) -> dict:
        scores, retrieved_docs = docs.get_nearest_examples(
            "embedding", model.encode(query["text"]), k=config.ranking_size
        )
        query["ranking"] = {"scores": scores, "doc_id": retrieved_docs["doc_id"]}
        return query

    queries = queries.map(
        rank,
        fn_kwargs={"encode_fn": model.encode, "docs": docs},
        keep_in_memory=config.use_cache,
    )

    breakpoint()


if __name__ == "__main__":
    main()
