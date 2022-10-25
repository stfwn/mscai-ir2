from typing import Callable
from argparse import ArgumentParser
import os
import json
import time

from datasets import Dataset
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

# def get_embeddings(d_id: str, passages: Dataset) -> list:
#     i = passages['passage_id'].index(d_id+'_0')
#     embs = [passages['passage_embedding'][i]]
#     i+=1
#     while passages['passage_id'][i].split('_')[0] == d_id:   
#         embs.append(passages['passage_embedding'][i])
#         i+=1
#         print(f"found passage {i}")
#     return embs


def main():

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device {device}")
    model = SentenceTransformer("sentence-transformers/msmarco-distilbert-dot-v5").to(device)

    # load data
    passages = Dataset.load_from_disk('./data/ms-marco/passage-embeddings/passage_size=512+prepend_title_to_passage=True+tokenization_method=model+flattened/')

    # build faiss index
    try:
        passages.load_faiss_index("passage_embedding", "./data/ms-marco/passage-embeddings/passage-embeddings.faiss")
        print('loaded faiss from disk')
    except:        
        passages.add_faiss_index(column="passage_embedding", metric_type=faiss.METRIC_INNER_PRODUCT)
        passages.save_faiss_index("passage_embedding", "./data/ms-marco/passage-embeddings/passage-embeddings.faiss")
        print('Saved faiss index to disk')
        
    # Load queries and qrels
    trec19 = TREC2019()
    queries_trec19 = trec19.get_queries()["test"]
    qrels_trec19 = "./data/trec/2019qrels-docs.tsv"
    print('loaded queries and qrels')
    # Rank
    name = "passages-trec19-ranking"
    print('writing to {}'.format(name))
    results_max = []
    results_mean = []
    results_sum = []
    a = time.time()
    for i, query in enumerate(queries_trec19):
        b = time.time()
        if i % 10 == 0:
            print(f"Done {i}/{len(queries_trec19)} queries in {b-a} sec")
        ranking = rank(query, passages, model)["ranking"]
        # Rerank
        # scores = {}
        # for p_id in ranking.keys():
        #     d_id = p_id.split('_')[0]
        #     doc_scores = np.array([np.dot(query_embedding, passage_embedding) for passage_embedding in get_embeddings(d_id, passages)])
        #     scores.update({d_id: (doc_scores.sum(), doc_scores.mean())})

        # rerank_sum = {k: v[0] for k, v in sorted(scores.items(), key=lambda item: item[1][0], reverse=True)}
        # rerank_mean = {k: v[1] for k, v in sorted(scores.items(), key=lambda item: item[1][1], reverse=True)}

        results_max.append((query["query_id"], ranking, name+'max'))
        # results_mean.append((query["query_id"], rerank_mean, name+'mean'))
        # results_sum.append((query["query_id"], rerank_sum, name+'sum'))

    result_file = "./data/results/" + name + "max.tsv"
    with open(result_file, "w") as f:
        for query_id, ranking, name in results_max:
            f.write(to_trec(query_id, ranking, name))

    result_file = "./data/results/" + name + "mean.tsv"
    with open(result_file, "w") as f:
        for query_id, ranking, name in results_mean:
            f.write(to_trec(query_id, ranking, name))

    result_file = "./data/results/" + name + "sum.tsv"
    with open(result_file, "w") as f:
        for query_id, ranking, name in results_sum:
            f.write(to_trec(query_id, ranking, name))


    Load queries and qrels
    trec20 = TREC2020()
    queries_trec20 = trec20.get_queries()["test"]
    qrels_trec20 = "./data/trec/2020qrels-docs.tsv"
    # Rank
    name = "passages-trec20-ranking"
    results_max = []
    results_mean = []
    results_sum = []
    a = time.time()
    for i, query in enumerate(queries_trec20):
        b = time.time()
        if i % 10 == 0:
            print(f"Done {i}/{len(queries_trec20)} queries in {b-a} sec")
        ranking = rank(query, passages, model)["ranking"]
        # Rerank
        # scores = {}
        # for p_id in ranking.keys():
        #     d_id = p_id.split('_')[0]
        #     doc_scores = np.array([np.dot(query_embedding, passage_embedding) for passage_embedding in get_embeddings(d_id, passages)])
        #     scores.update({d_id: (doc_scores.sum(), doc_scores.mean())})

        # rerank_sum = {k: v[0] for k, v in sorted(scores.items(), key=lambda item: item[1][0], reverse=True)}
        # rerank_mean = {k: v[1] for k, v in sorted(scores.items(), key=lambda item: item[1][1], reverse=True)}

        results_max.append((query["query_id"], ranking, name+'max'))
        # results_mean.append((query["query_id"], rerank_mean, name+'mean'))
        # results_sum.append((query["query_id"], rerank_sum, name+'sum'))

    result_file = "./data/results/" + name + "max.tsv"
    with open(result_file, "w") as f:
        for query_id, ranking, name in results_max:
            f.write(to_trec(query_id, ranking, name))

    # result_file = "./data/results/" + name + "mean.tsv"
    # with open(result_file, "w") as f:
    #     for query_id, ranking, name in results_mean:
    #         f.write(to_trec(query_id, ranking, name))

    # result_file = "./data/results/" + name + "sum.tsv"
    # with open(result_file, "w") as f:
    #     for query_id, ranking, name in results_sum:
    #         f.write(to_trec(query_id, ranking, name))


    # Load queries (ds) and qrels (trec)
    ms_marco_docs = MSMarcoDocs()
    queries_msmarco = ms_marco_docs.get_queries()["dev"]
    # queries_msmarco = Dataset.load_from_disk('./data/ms-marco/query-embeddings/dev/')
    qrels_msmarco = "./data/ms-marco/msmarco-docdev-qrels.tsv"
    # Rank
    name = "passages-ms-marco-ranking"
    results_max = []
    # results_mean = []
    # results_sum = []
    a = time.time()
    for i, query in enumerate(queries_msmarco):
        b = time.time()
        if i % 100 == 0:
            print(f"Done {i}/{len(queries_msmarco)} queries in {b-a} sec")
        ranking = rank(query, passages, model)["ranking"]
        # Rerank
        # scores = {}
        # for p_id in ranking.keys():
        #     d_id = p_id.split('_')[0]
        #     doc_scores = np.array([np.dot(query_embedding, passage_embedding) for passage_embedding in get_embeddings(d_id, passages)])
        #     scores.update({d_id: (doc_scores.sum(), doc_scores.mean())})

        # rerank_sum = {k: v[0] for k, v in sorted(scores.items(), key=lambda item: item[1][0], reverse=True)}
        # rerank_mean = {k: v[1] for k, v in sorted(scores.items(), key=lambda item: item[1][1], reverse=True)}

        results_max.append((query["query_id"], ranking, name+'max'))
        # results_mean.append((query["query_id"], rerank_mean, name+'mean'))
        # results_sum.append((query["query_id"], rerank_sum, name+'sum'))

    result_file = "./data/results/" + name + "max.tsv"
    with open(result_file, "w") as f:
        for query_id, ranking, name in results_max:
            f.write(to_trec(query_id, ranking, name))


if __name__ == "__main__":
    main()





