from typing import Callable
from argparse import ArgumentParser
import os
import json
from datasets import disable_caching

from datasets import Dataset, Features, Value
from sentence_transformers import SentenceTransformer
import torch
# from ir_measures import *
# import ir_measures
import numpy as np
import faiss

import config
from data import MSMarcoDocs, TREC2019, TREC2020

# METRICS = [RR@10, nDCG@10, R@1000, nDCG, RR, AP, P @ 1, P @ 5, P @ 20, P @ 100, R @ 100]

"""
Evaluation
==================================================
Input:
    - Path to .json file containing Hugginface Dataset
    - Path to .faiss file containing FAISS index
Output:
    - Dict of format:
        output = {measure: value}
"""


def rank(query: dict, docs: Dataset, model: SentenceTransformer) -> dict:
    scores, retrieved_docs = docs.get_nearest_examples(
        "embeddings", model.encode(query["text"]), k=config.ranking_size
    )
    query["ranking"] = dict(zip(retrieved_docs["doc_id"], scores))
    return query


def to_trec(query_id: str, ranking: dict, name: str) -> str:
    result = ""
    for i, (doc_id, score) in enumerate(ranking.items()):
        result += f"{query_id}\t0\t{doc_id}\t{i+1}\t{score}\t{name}\n"
    return result

def batch_rank(queries, docs):
    # Change query embeddings to np array, otherwise the get_nearest_examples fails
    query_embeddings = np.array(queries['embedding'], dtype=np.float32)
    
    scores, retrieved_docs = docs.get_nearest_examples_batch(
        'passage_embedding', query_embeddings, k=config.ranking_size
    )
    parsed_scores = [dict(zip(ret_docs["doc_id"], score)) for score, ret_docs in zip(scores, retrieved_docs)]
    queries = queries.add_column('ranking', parsed_scores)
    return queries


def main(args):
    disable_caching()

    # Load index
    docs = Dataset.load_from_disk(args.dataset_file)
    docs.add_faiss_index(column="passage_embedding", metric_type=faiss.METRIC_INNER_PRODUCT)

    print('Ranking MSMarco')

    # Initialize MS-MARCO
    queries_msmarco = Dataset.load_from_disk('./data/embeddings/queries-msmarco-dev')

    # Encode queries (dict --> trec) and get rankings (dict)
    name = "passage-ms-marco-ranking"
    results = batch_rank(queries_msmarco, docs)
    result_file = "./data/results/" + name + ".tsv"
    with open(result_file, "w") as f:
        for query in results:
            f.write(to_trec(query['query_id'], query['ranking'], name))

    print('Ranking Trec')

    queries_trec19 = Dataset.load_from_disk('./data/embeddings/queries-trec19')
    
    name = "passage-trec19-ranking"
    results = batch_rank(queries_trec19, docs)
    result_file = "./data/results/" + name + ".tsv"
    with open(result_file, "w") as f:
        for query in results:
            f.write(to_trec(query['query_id'], query['ranking'], name))
    
    # Load queries (ds) and qrels (trec)
    queries_trec20 = Dataset.load_from_disk('./data/embeddings/queries-trec20')

    name = "passage-trec20-ranking"
    results = batch_rank(queries_trec20, docs)
    result_file = "./data/results/" + name + ".tsv"
    with open(result_file, "w") as f:
        for query in results:
            f.write(to_trec(query['query_id'], query['ranking'], name))
        


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument(
        "-d",
        "--dataset-file",
        type=str,
        help="Path to Huggingface Dataset (.arrow) of documents to rank.",
    )
    argparser.add_argument(
        '--disable-caching', default=False, type=bool
    )
    args = argparser.parse_args()
    main(args)