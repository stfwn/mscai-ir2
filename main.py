from typing import Callable
import os
import json

from datasets import Dataset
from sentence_transformers import SentenceTransformer, util
import torch

import config
from data import MSMarcoDocs, TREC2019, TREC2020
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

    # Initialize model to compute passage embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/msmarco-distilbert-dot-v5").to(
        device
    )

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
        fn_kwargs={
            "model": model,
            "passage_size": config.passage_size,
            "tokenization_method": config.tokenization_method,
            "prepend_title_to_passage": config.prepend_title_to_passage,
        },
        keep_in_memory=config.keep_in_memory,
    )

    # Compute passage embeddings
    docs = docs.map(
        encoding.encode_passages,
        fn_kwargs={"encode_fn": model.encode},
        keep_in_memory=config.keep_in_memory,
    )

    # Compute document embeddings
    docs = docs.map(
        encoding.encode_doc,
        fn_kwargs={
            "method": "mean_passage_embeddings",
        },
        keep_in_memory=config.keep_in_memory,
    )

    json.dump(docs[:], open('./data/ms-marco/document-embeddings-dataset.json', 'w'))

    try:
        docs.load_faiss_index("embedding", "./data/ms-marco/document-embeddings.faiss")
    except:
        docs.add_faiss_index(column="embedding")
        docs.save_faiss_index("embedding", "./data/ms-marco/document-embeddings.faiss")


if __name__ == "__main__":
    main()
