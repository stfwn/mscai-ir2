from typing import Callable
from argparse import ArgumentParser
import os
import json

from datasets import Dataset
from sentence_transformers import SentenceTransformer, util
import torch
from ir_measures import *
import ir_measures
import numpy as np

import config
from data import MSMarcoDocs, TREC2019, TREC2020
import encoding
import preprocessing

METRICS = [RR@10, nDCG@10, R@1000, nDCG, RR, AP, P @ 1, P @ 5, P @ 20, P @ 100, R @ 100]

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
        "embedding", model.encode(query["text"]), k=config.ranking_size
    )
    query["ranking"] = dict(zip(retrieved_docs["doc_id"], scores))
    return query


def to_trec(query_id: str, ranking: dict, name: str) -> str:
    result = ""
    for i, (doc_id, score) in enumerate(ranking.items()):
        result += f"{query_id}\t0\t{doc_id}\t{i+1}\t{score}\t{name}\n"
    return result


def evaluate(run, qrels_path: str, metrics: list) -> dict:
    if isinstance(run, str):
        run = ir_measures.read_trec_run(run)
    qrels = ir_measures.read_trec_qrels(qrels_path)
    return ir_measures.calc_aggregate(metrics, qrels, run)


def main(args):

    # Load index
    docs = Dataset.load_from_disk(args.dataset_file)
    # docs = Dataset.from_dict(json.load(open(args.dataset_file, 'r')))
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
        # queries_msmarco = ms_marco_docs.get_queries()["dev"]
        queries_msmarco = Dataset.load_from_disk('./data/ms-marco/query-embeddings/dev/')
        qrels_msmarco = "./data/ms-marco/msmarco-docdev-qrels.tsv"

        # Encode queries (dict --> trec) and get rankings (dict)
        name = args.dataset_file.split("/")[3] + "-ms-marco-ranking"
        results = []
        # run_msmarco = {}
        for i, query in enumerate(queries_msmarco):
            if i % 100 == 0:
                print(f"Done {i}/{len(queries_msmarco)} queries.")
            # query_emb = model.encode(query["text"])
            # print(type(query_emb), query_emb.shape, type(query_emb[0]), query_emb[0])
            print(np.array(query["embedding"]).shape, type(np.array(query["embedding"])[0]), np.array(query["embedding"])[0])
            scores, retrieved_docs = docs.get_nearest_examples("embedding", np.array(query["embedding"]), k=config.ranking_size)
            ranking = dict(zip(retrieved_docs["doc_id"], scores))
            results.append((query["query_id"], ranking, name))
            # run_msmarco.update({query['query_id'] : ranking})

        result_file = "./data/results/" + name + ".tsv"
        with open(result_file, "w") as f:
            for query_id, ranking, name in results:
                f.write(to_trec(query_id, ranking, name))
                
        # Evaluate (either from dict or from TREC formatted file)
        # results_from_dict = evaluate(run_msmarco, qrels_msmarco, METRICS)
        results_from_file = evaluate(result_file, qrels_msmarco, METRICS)
        print(f"results from {args.dataset_file} on {qrels_msmarco}")
        # print(results_from_dict)
        print(results_from_file)

    trec = True
    if trec:
        # Initialize TREC 2019
        trec19 = TREC2019()

        queries_trec19 = trec19.get_queries()["test"]
        qrels_trec19 = "./data/trec/2019qrels-docs.tsv"

        # Encode queries (ds) and get rankings (dict)
        name = args.dataset_file.split("/")[3] + "-trec19-ranking"
        result = "./data/results/" + name + ".tsv"
        # run_trec19 = {}
        with open(result, "w") as f:
            for i, query in enumerate(queries_trec19):
                if i % 100 == 0:
                    print(f"Done {i}/{len(queries_trec19)} queries.")
                ranking = rank(query, docs, model)["ranking"]
                # run_trec19.update({query['query_id'] : ranking})
                f.write(to_trec(query["query_id"], ranking, name))

        # Evaluate (either from dict or from TREC formatted file)
        # results_from_dict = evaluate(run_trec19, qrels_trec19, METRICS)
        results_from_file = evaluate(result, qrels_trec19, METRICS)
        # print(results_from_dict)
        print(f"results from {args.dataset_file} on {qrels_trec19}")
        print(results_from_file)

        # Initialize TREC 2020
        trec20 = TREC2020()

        # Load queries (ds) and qrels (trec)
        queries_trec20 = trec20.get_queries()["test"]
        qrels_trec20 = "./data/trec/2020qrels-docs.tsv"

        # Encode queries (ds) and get rankings (dict)
        name = args.dataset_file.split('/')[3] + "-trec20-ranking"
        result = './data/results/'+name+'.tsv'
        # run_trec20 = {}
        with open(result, 'w') as f:
            for i, query in enumerate(queries_trec20):
                if i%100 == 0:
                    print(f"Done {i}/{len(queries_trec20)} queries.")
                ranking = rank(query, docs, model)['ranking']
                # run_trec19.update({query['query_id'] : ranking})
                f.write(to_trec(query['query_id'], ranking, name))

        # Evaluate (either from dict or from TREC formatted file)
        # results_from_dict = evaluate(run_trec20, qrels_trec20, METRICS)
        results_from_file = evaluate(result, qrels_trec20, METRICS)
        # print(results_from_dict)
        print(f"results from {args.dataset_file} on {qrels_trec20}")
        print(results_from_file)


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
