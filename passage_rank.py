from typing import Callable
from argparse import ArgumentParser
import os
import json
import time

from datasets import Dataset
from datasets import disable_caching
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import faiss

import config
from data import MSMarcoDocs, TREC2019, TREC2020
import encoding
import preprocessing


def rank(query: dict, docs: Dataset, model: SentenceTransformer) -> dict:
    scores, retrieved_docs = docs.get_nearest_examples(
        "passage_embedding", model.encode(query["text"]), k=config.ranking_size
    )
    query["ranking"] = dict(zip(retrieved_docs["passage_id"], scores))
    return query

def to_trec(query_id: str, ranking: dict, name: str) -> str:
    result = ""
    for i, (doc_id, score) in enumerate(ranking.items()):
        result += f"{query_id}\t0\t{doc_id}\t{i+1}\t{score}\t{name}\n"
    return result


def main():
    disable_caching()
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device {device}")
    model = SentenceTransformer("sentence-transformers/msmarco-distilbert-dot-v5").to(device)

    # load data
    passages = Dataset.load_from_disk('./data/ms-marco/passage-embeddings/passage_size=512+prepend_title_to_passage=True+tokenization_method=model+flattened/')

    # build faiss index   
    passages.add_faiss_index(column="passage_embedding", metric_type=faiss.METRIC_INNER_PRODUCT)
    passages.save_faiss_index("passage_embedding", "./data/ms-marco/passage-embeddings/passage-embeddings.faiss")
    print('Saved faiss index to disk')

    # Load queries and qrels
    trec19 = TREC2019()
    print("init done")
    queries_trec19 = trec19.get_queries()["test"]
    print("queries loaded")
    qrels_trec19 = "./data/trec/2019qrels-docs.tsv"
    print('qrels set')
    # Rank
    name = "passages-trec19-ranking"
    print('writing to {}'.format(name))
    results_max = []
    a = time.time()
    for i, query in enumerate(queries_trec19):
        b = time.time()
        if i % 10 == 0:
            print(f"Done {i}/{len(queries_trec19)} queries in {b-a} sec")
        ranking = rank(query, passages, model)["ranking"]
        results_max.append((query["query_id"], ranking, name+'max'))

    result_file = "./data/results/" + name + "max.tsv"
    with open(result_file, "w") as f:
        for query_id, ranking, name in results_max:
            f.write(to_trec(query_id, ranking, name))


    # Load queries and qrels
    trec20 = TREC2020()
    queries_trec20 = trec20.get_queries()["test"]
    qrels_trec20 = "./data/trec/2020qrels-docs.tsv"
    # Rank
    name = "passages-trec20-ranking"
    print('writing to {}'.format(name))
    results_max = []
    results_mean = []
    results_sum = []
    a = time.time()
    for i, query in enumerate(queries_trec20):
        b = time.time()
        if i % 10 == 0:
            print(f"Done {i}/{len(queries_trec20)} queries in {b-a} sec")
        results_max.append((query["query_id"], ranking, name+'max'))

    result_file = "./data/results/" + name + "max.tsv"
    with open(result_file, "w") as f:
        for query_id, ranking, name in results_max:
            f.write(to_trec(query_id, ranking, name))


    # Load queries (ds) and qrels (trec)
    ms_marco_docs = MSMarcoDocs()
    queries_msmarco = ms_marco_docs.get_queries()["dev"]
    qrels_msmarco = "./data/ms-marco/msmarco-docdev-qrels.tsv"
    # Rank
    name = "passages-ms-marco-ranking"
    print('writing to {}'.format(name))
    results_max = []
    a = time.time()
    for i, query in enumerate(queries_msmarco):
        b = time.time()
        if i % 100 == 0:
            print(f"Done {i}/{len(queries_msmarco)} queries in {b-a} sec")
        ranking = rank(query, passages, model)["ranking"]
        results_max.append((query["query_id"], ranking, name+'max'))

    result_file = "./data/results/" + name + "max.tsv"
    with open(result_file, "w") as f:
        for query_id, ranking, name in results_max:
            f.write(to_trec(query_id, ranking, name))


if __name__ == "__main__":
    main()





