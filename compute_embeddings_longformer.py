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
    print("=" * 10)
    print("Number of shards:", args.n_shards)
    print("Computing shard:", args.shard_index)
    print("=" * 10)

    print("==> Initializing model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.model_path).to(
        device
    )

    print("==> Loading docs shard")
    ms_marco_docs = MSMarcoDocs()
    docs = (
        ms_marco_docs.get_docs()
        .sort("doc_id")
        .shard(
            num_shards=args.n_shards,
            index=args.shard_index,
            contiguous=True,
        )
        .filter(lambda d: d["body"] != "" and d["body"] is not None)
    )
    print("==> Preprocessing docs")
    docs = docs.map(
        preprocessing.doc_to_longformer_input,
        fn_kwargs={
            "passage_size": args.passage_size,
        },
    )

    print("==> Computing doc embeddings")
    embeddings = model.encode([doc['body'] for doc in docs], batch_size=args.batch_size, device=device, show_progress_bar=True)
    for doc, embedding in zip(docs, embeddings):
        doc['embedding'] = embedding 
        del doc['body']
    
    print("==> Saving dataset with passage embeddings to disk")
    dataset_dir = Path("./longformer/embeddings/") / "+".join(
        sorted([f"{k}={v}" for k, v in vars(args).items()])
    )
    print(f"== Dir:", dataset_dir)
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

    argparser.add_argument("--passage-size", type=int, default=1000)
    argparser.add_argument(
        "--tokenization-method",
        type=str,
        choices=["model", "spaces"],
        default="model",
    )
    argparser.add_argument(
        "--prepend-title-to-passage", action="store_true", default=True
    )
    argparser.add_argument(
        "--model_path", default='allenai/longformer-base-4096'
    )
    argparser.add_argument(
        "--batch_size", default=32, type=int
    )

    args = argparser.parse_args()
    main(args)
