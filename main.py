from typing import Callable
import os

from datasets import Dataset
from sentence_transformers import SentenceTransformer, util
import torch

import config
from data import MSMarcoDocs, TREC2019, TREC2020
import encoding
import preprocessing
from evaluation import evaluate


def main():
    print("=" * 10)
    print("Running in stage:", config.stage)
    print("=" * 10)

    # Initialize MS Marco
    ms_marco_docs = MSMarcoDocs()
    docs = ms_marco_docs.get_docs()
    queries = ms_marco_docs.get_queries()

    if config.stage == "EVAL":
        # Initialize TREC
        trec19 = TREC2019()
        trec20 = TREC2020()
        queries19 = trec19.get_queries()
        queries20 = trec20.get_queries()

    if config.stage == "DEV":
        docs = docs.train_test_split(train_size=10, test_size=1, shuffle=False)["train"]
        queries["train"] = queries["train"].train_test_split(
            train_size=50, test_size=1, shuffle=False
        )["train"]
        queries["dev"] = queries["dev"].train_test_split(
            train_size=50, test_size=1, shuffle=False
        )["train"]

    if config.stage == "PROD":
        docs = docs.map(
            preprocessing.doc_to_passages,
            fn_kwargs={
                "passage_size": config.passage_size,
                "tokenization_method": config.tokenization_method,
                "prepend_title_to_passage": config.prepend_title_to_passage,
            },
            keep_in_memory=config.keep_in_memory,
        )

    # Initialize model to compute passage embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/msmarco-distilbert-dot-v5").to(
        device
    )

    if config.stage in ["DEV", "PROD"]:
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
    elif config.stage == "EVAL":
        # load embeddings
        pass

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
        keep_in_memory=config.keep_in_memory,
    )

    print(evaluate(queries["train"], qrels_path="data/ms-marco/msmarco-doctrain-qrels.tsv"))
    print(evaluate(queries["dev"], qrels_path="data/ms-marco/msmarco-docdev-qrels.tsv"))


if __name__ == "__main__":
    main()
