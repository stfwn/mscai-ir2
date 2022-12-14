from typing import Callable
from argparse import ArgumentParser
import os
import json

from datasets import Dataset
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np

from data import MSMarcoDocs, TREC2019, TREC2020
import encoding
import preprocessing

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
ranking_size = 1000

def rank(query: dict, docs: Dataset, model: SentenceTransformer) -> dict:
    scores, retrieved_docs = docs.get_nearest_examples(
        "embedding", model.encode(query["text"]), k=ranking_size
    )
    query["ranking"] = dict(zip(retrieved_docs["doc_id"], scores))
    return query


def to_trec(query_id: str, ranking: dict, name: str) -> str:
    result = ""
    for i, (doc_id, score) in enumerate(ranking.items()):
        result += f"{query_id}\t0\t{doc_id}\t{i+1}\t{score}\t{name}\n"
    return result

def main(args):

    # Load index
    docs = Dataset.load_from_disk(args.dataset_file)
    docs.load_faiss_index("embedding", args.index_file)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/msmarco-distilbert-dot-v5").to(
        device
    )
    print(f"Using device: {model.device}")

    ms_marco = True
    if ms_marco:
        # Initialize MS-MARCO
        ms_marco_docs = MSMarcoDocs()

        # Load queries (ds) and qrels (trec)
        queries_msmarco = ms_marco_docs.get_queries()["dev"]
        qrels_msmarco = "./data/ms-marco/msmarco-docdev-qrels.tsv"

        # Encode queries (dict --> trec) and get rankings (dict)
        name = args.dataset_file.split("/")[3] + "-ms-marco-ranking"
        results = []
        for i, query in enumerate(queries_msmarco):
            if i % 100 == 0:
                print(f"Done {i}/{len(queries_msmarco)} queries.")
            ranking = rank(query, docs, model)["ranking"]
            results.append((query["query_id"], ranking, name))

        result_file = "./data/results/" + name + ".tsv"
        with open(result_file, "w") as f:
            for query_id, ranking, name in results:
                f.write(to_trec(query_id, ranking, name))

    trec = True
    if trec:
        # Initialize TREC 2019
        trec19 = TREC2019()

        queries_trec19 = trec19.get_queries()["test"]
        qrels_trec19 = "./data/trec/2019qrels-docs.tsv"

        # Encode queries (ds) and get rankings (dict)
        name = args.dataset_file.split("/")[3] + "-trec19-ranking"
        result = "./data/results/" + name + ".tsv"
        with open(result, "w") as f:
            for i, query in enumerate(queries_trec19):
                if i % 100 == 0:
                    print(f"Done {i}/{len(queries_trec19)} queries.")
                ranking = rank(query, docs, model)["ranking"]
                f.write(to_trec(query["query_id"], ranking, name))


        # Initialize TREC 2020
        trec20 = TREC2020()

        # Load queries (ds) and qrels (trec)
        queries_trec20 = trec20.get_queries()["test"]
        qrels_trec20 = "./data/trec/2020qrels-docs.tsv"

        # Encode queries (ds) and get rankings (dict)
        name = args.dataset_file.split('/')[3] + "-trec20-ranking"
        result = './data/results/'+name+'.tsv'
        with open(result, 'w') as f:
            for i, query in enumerate(queries_trec20):
                if i%100 == 0:
                    print(f"Done {i}/{len(queries_trec20)} queries.")
                ranking = rank(query, docs, model)['ranking']
                f.write(to_trec(query['query_id'], ranking, name))


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument(
        "-d",
        "--dataset-file",
        type=str,
        help="Path to Huggingface Dataset (.arrow) of documents to rank.",
    )
    argparser.add_argument(
        "-f",
        "--index-file",
        type=str,
        help="Path to FAISS index (.faiss) of documents to rank.",
    )
    args = argparser.parse_args()
    main(args)
