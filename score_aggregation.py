from typing import Callable
from argparse import ArgumentParser
import os
import json
import time

from datasets import Dataset
from sentence_transformers import SentenceTransformer, util
import torch

import config
from data import MSMarcoDocs, TREC2019, TREC2020
import encoding
import preprocessing

import gc


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

def main():

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/msmarco-distilbert-dot-v5").to(device)

    # load data
    try:
        passages = load_from_disk('./data/ms-marco/passage-embeddings/passage_size=512+prepend_title_to_passage=True+tokenization_method=model+flattened/')
    except:
        print("===>STARTING SHARTING")
        num_shards = 10
        for k in range(num_shards):
            # load only shard
            docs = Dataset.from_dict(
                    Dataset.load_from_disk(
                        './data/ms-marco/passage-embeddings/passage_size=512+prepend_title_to_passage=True+tokenization_method=model/'
                    )[int((k/num_shards)*3201821):int((k+1/num_shards)*3201821)]
                )
            # flatten data
            ps_data = {
                'passage_id': [],
                'passage_embedding': []
            }
            a = time.time()
            for i, doc in enumerate(docs):
                if i % 100 == 0:
                    b = time.time()
                    print(f"Flattening doc {i} out of {len(docs)}; time {b-a}")
                for passage in doc['passages']:
                    ps_data['passage_id'].append(passage['passage_id'])
                    ps_data['passage_embedding'].append(passage['passage_embedding'])

            passages = Dataset.from_dict(ps_data)
            passages.save_to_disk('./data/ms-marco/passage-embeddings/passage_size=512+prepend_title_to_passage=True+tokenization_method=model+flattened_{}/'.format(k))
            print('Saved flattened passage dataset to disk')

            del docs
            del ps_data
            del passages
            gc.collect()

    # build faiss index
    try:
        passages.load_faiss_index("passage_embedding", "./data/ms-marco/passage-embeddings/passage-embeddings.faiss")
    except:
        passages.add_faiss_index(column="passage_embedding")
        passages.save_faiss_index("passage_embedding", "./data/ms-marco/passage-embeddings/passage-embeddings.faiss")
        print('Saved faiss index to disk')

 #    # # Load queries (ds) and qrels (trec)
 #    # ms_marco_docs = MSMarcoDocs()
 #    # # queries_msmarco = ms_marco_docs.get_queries()["dev"]
 #    # queries_msmarco = Dataset.load_from_disk('./data/ms-marco/query-embeddings/dev/')
 #    # qrels_msmarco = "./data/ms-marco/msmarco-docdev-qrels.tsv"

    # Load queries and qrels
    trec19 = TREC2019()
    queries_trec19 = trec19.get_queries()["test"]
    qrels_trec19 = "./data/trec/2019qrels-docs.tsv"
    # Rank
    name = "passages-trec19-ranking"
    results_max = []
    results_mean = []
    results_sum = []
    a = time.time()
    for i, query in enumerate(queries_trec19):
        b = time.time()
        if i % 10 == 0:
            print(f"Done {i}/{len(queries_trec19)} queries in {b-a} sec")
        ranking = rank(query, docs, model)["ranking"]
        # Rerank
        scores = {}
        for p_id in ranking.keys():
            d_id = p_id.split('_')[0]
            i = ds['doc_id'].index(d_id)
            doc_scores = np.array([np.dot(query_embedding, passage_embedding) for passage_embedding in ds['passages'][i]['passage_embedding']])
            scores.update({d_id: (doc_scores.sum(), doc_scores.mean())})

        rerank_sum = {k: v[0] for k, v in sorted(scores.items(), key=lambda item: item[1][0], reverse=True)}
        rerank_mean = {k: v[1] for k, v in sorted(scores.items(), key=lambda item: item[1][1], reverse=True)}

        results_max.append((query["query_id"], ranking, name+'max'))
        results_mean.append((query["query_id"], rerank_mean, name+'mean'))
        results_sum.append((query["query_id"], rerank_sum, name+'sum'))

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



if __name__ == "__main__":
    main()





