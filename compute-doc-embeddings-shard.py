from argparse import ArgumentParser
from pathlib import Path
from typing import Callable
import os

import datasets
from datasets import Dataset
from sentence_transformers import SentenceTransformer, util
import torch

import config
from data import MSMarcoDocs
import encoding
import preprocessing


def main(args):
    print("=" * 20)
    print("Number of shards:", args.n_shards)
    print("Computing shard:", args.shard_index)
    print("=" * 20)

    print("Loading docs shard")
    ms_marco_docs = MSMarcoDocs()
    docs = (
        ms_marco_docs.get_docs()
        .sort("doc_id")
        .shard(
            num_shards=args.n_shards,
            index=args.shard_index,
            contiguous=True,
        )
    )

    print("Splitting docs into passages")
    docs = docs.map(
        preprocessing.doc_to_passages,
        fn_kwargs={
            "passage_size": args.passage_size,
            "tokenization_method": args.tokenization_method,
            "prepend_title_to_passage": args.prepend_title_to_passage,
        },
    )

    print("Initializing model")
    # Initialize model to compute passage embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/msmarco-distilbert-dot-v5").to(
        device
    )

    print("Computing passage embeddings")
    docs = docs.map(
        encoding.encode_passages,
        fn_kwargs={"encode_fn": model.encode},
        writer_batch_size=10000,
    )

    print("Computing document embeddings")
    docs = docs.map(
        encoding.encode_doc,
        fn_kwargs={
            "method": "mean_passage_embeddings",
        },
        writer_batch_size=10000,
    )

    print("Saving dataset with doc and passage embeddings to disk")
    dataset_dir = Path("./data/ms-marco/") / "+".join(
        sorted([f"{k}={v}" for k, v in vars(args).items()])
    )
    print(f"Dir:", dataset_dir)
    docs.save_to_disk(dataset_dir)


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("-n", "--n-shards", type=int, help="Total number of shards.")
    argparser.add_argument(
        "-i",
        "--shard-index",
        type=int,
        help="Which shard to compute.",
    )

    argparser.add_argument("--passage-size", type=int, default=400)
    argparser.add_argument("--tokenization-method", type=str, default="spaces")
    argparser.add_argument(
        "--prepend-title-to-passage", action="store_true", default=True
    )

    args = argparser.parse_args()
    main(args)
