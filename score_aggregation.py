from typing import Callable
from argparse import ArgumentParser
import os
import json

from datasets import Dataset
from sentence_transformers import SentenceTransformer, util
import torch
from ir_measures import *
import ir_measures

import config
from data import MSMarcoDocs, TREC2019, TREC2020
import encoding
import preprocessing

from tqdm import tqdm


METRICS = [nDCG, RR, AP, P @ 1, P @ 5, P @ 20, P @ 100, R @ 100]

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

def main():

	# load data
    docs = Dataset.load_from_disk('./data/ms-marco/passage-embeddings/passage_size=512+prepend_title_to_passage=True+tokenization_method=model/')
    print(docs[0]['passages'][0]['passage_id'])
 #    passages = docs['passages']
	# # build faiss index
 #    try:
 #    	passages.load_faiss_index("embedding", "./data/ms-marco/passage-embeddings/passage-embeddings.faiss")
 #    except:
	#     passages.add_faiss_index(column="embedding")
	#     passages.save_faiss_index("embedding", "./data/ms-marco/passage-embeddings/passage-embeddings.faiss")

 #    # # Load queries (ds) and qrels (trec)
 #    # ms_marco_docs = MSMarcoDocs()
 #    # # queries_msmarco = ms_marco_docs.get_queries()["dev"]
 #    # queries_msmarco = Dataset.load_from_disk('./data/ms-marco/query-embeddings/dev/')
 #    # qrels_msmarco = "./data/ms-marco/msmarco-docdev-qrels.tsv"

 #    # Load queries and qrels
 #    trec19 = TREC2019()
 #    queries_trec19 = trec19.get_queries()["test"]
 #    qrels_trec19 = "./data/trec/2019qrels-docs.tsv"
 #    # Rank
 #    name = args.dataset_file.split("/")[3] + "-trec19-ranking"
 #    result = "./data/results/" + name + ".tsv"
 #    with open(result, "w") as f:
 #        for i, query in enumerate(queries_trec19):
 #            if i % 100 == 0:
 #                print(f"Done {i}/{len(queries_trec19)} queries.")
 #            ranking = rank(query, docs, model)["ranking"]
 #            f.write(to_trec(query["query_id"], ranking, name))

    # Rerank mean

    # Rerank sum

if __name__ == "__main__":
	main()