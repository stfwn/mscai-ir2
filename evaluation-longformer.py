from argparse import ArgumentParser
from datasets import disable_caching
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np
from sklearn.preprocessing import Normalizer
import config
import pickle
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


def rank(query: dict, docs: Dataset) -> dict:
    scores, retrieved_docs = docs.get_nearest_examples(
        "norm_embeddings", np.array(query["norm_embedding"], dtype=np.float32), k=2000
    )
    query["ranking"] = dict(zip(retrieved_docs["doc_id"], scores))
    return query

def to_trec(query_id: str, ranking: dict, name: str) -> str:
    result = ""
    for i, (doc_id, score) in enumerate(ranking.items()):
        result += f"{query_id}\t0\t{doc_id}\t{i+1}\t{score}\t{name}\n"
    return result

def normalize_embs(query):
    normer = Normalizer()
    query.update({'norm_embedding': normer.fit_transform(np.array(query['embedding']).reshape(1, -1))})
    return query

def main(args):
    if args.disable_caching:
        disable_caching()

    docs = Dataset.load_from_disk(args.dataset_file)
    docs.remove_columns('embedding')
    docs.add_faiss_index('norm_embeddings')

    print('Ranking Trec')

    queries_trec19 = Dataset.load_from_disk('longformer/embeddings/queries-trec19')
    queries_trec19 = queries_trec19.map(normalize_embs)
 
    queries_trec19.remove_columns('embedding')

    name = "longformer-trec19-ranking"
    result_file = "./data/results/" + name + ".tsv"
    with open(result_file, "w") as f:
        for query in queries_trec19:
            query = rank(query, docs)
            f.write(to_trec(query['query_id'], query['ranking'], name))
    
    queries_trec20 = Dataset.load_from_disk('longformer/embeddings/queries-trec20')
    queries_trec20 = queries_trec20.map(normalize_embs)
    queries_trec20.remove_columns('embedding')

    name = "longformer-trec20-ranking"
    result_file = "./data/results/" + name + ".tsv"
    with open(result_file, "w") as f:
        for query in queries_trec19:
            query = rank(query, docs)
            f.write(to_trec(query['query_id'], query['ranking'], name))
    
    print('Ranking MSMarco')

    # Initialize MS-MARCO
    queries_msmarco = Dataset.load_from_disk('longformer/embeddings/queries-msmarco-dev')
    queries_msmarco = queries_msmarco.map(normalize_embs)
    queries_msmarco.remove_columns('embedding')

    name = "longformer-ms-marco-ranking"

    result_file = "./data/results/" + name + ".tsv"
    with open(result_file, "w") as f:
        for query in queries_trec19:
            query = rank(query, docs)
            f.write(to_trec(query['query_id'], query['ranking'], name))


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument(
        "-d",
        "--dataset-file",
        type=str,
        default='longformer/combined-embeddings-normalized',
        help="Path to Huggingface Dataset (.arrow) of embedded, normalized docs to rank.",
    )

    argparser.add_argument(
        '--disable-caching', default=False, type=bool)
    args = argparser.parse_args()
    main(args)