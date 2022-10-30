import ir_measures
from ir_measures import *
import random

from argparse import ArgumentParser
import pandas as pd

METRICS = [RR@10, RR@100, nDCG@10, R@1000, AP@100]#, RR, AP, P@1, P@5, P@20, P@100, R@100, nDCG]

def evaluate(run, qrels_path: str, metrics: list) -> dict:
    if isinstance(run, str):
        run = ir_measures.read_trec_run(run)
    qrels = ir_measures.read_trec_qrels(qrels_path)
    return ir_measures.calc_aggregate(metrics, qrels, run)

def main(args):

    # Load qrels (trec)
    qrels = args.qrels
    ranking = args.ranking

    results = evaluate(ranking, qrels, METRICS)
    for metric, score in results.items():
        print("{}: {:.4f}".format(metric,score))
    print(results)


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument(
        "-q",
        "--qrels",
        type=str,
        help="Path to qrels",
    )
    argparser.add_argument(
        "-r",
        "--ranking",
        type=str,
        help="Path to ranking",
    )
    argparser.add_argument(
        "-m",
        "--maxP",
        type=bool,
        help="Whether maxP is true",
    )
    args = argparser.parse_args()
    main(args)
