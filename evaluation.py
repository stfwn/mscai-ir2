import ir_measures
from ir_measures import *

"""
Evaluation
==================================================
Input:
	- Huggingface Dataset containing queries:
		queries["ranking"] = [{"doc_id": [doc_ids], "scores": [scores]}]
		queries["query_id"] = ["id_1, ..., id_n"]
	- Relevance labels in TREC format
		i.e. PATH to labels: "/data/ms-marco/msmarco-docdev-qrels.tsv"
	- List of measures to calculate
		default = [RR, AP, P@k, R@100] (k = [1, 5, 20, 50, 100])

Output:
	- Dict of format:
		output = {measure: value}

"""

def input_to_run_format(queries):
	result = {}
	for i, q_id in enumerate(queries["query_id"]):
		ranking = queries["ranking"][i]
		tmp = dict(zip(ranking["doc_id"], ranking["scores"]))
		result.update({q_id : tmp})
	return result

def evaluate(
	queries,
	qrels_path="/data/ms-marco/msmarco-docdev-qrels.tsv",
	metrics=[RR, AP, P@1, P@5, P@20, P@100, R@100]
	):
	run = input_to_run_format(queries)
	qrels = ir_measures.read_trec_qrels(qrels_path)
	return ir_measures.calc_aggregate(metrics, qrels, run)
