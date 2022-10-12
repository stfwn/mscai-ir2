from argparse import ArgumentParser
from typing import Literal

import datasets
import faiss
import torch


def main(args):
    args_dict_without_agg = vars(args).copy()
    del args_dict_without_agg["aggregation_method"]

    print("==> Loading passage embeddings dataset from disk")
    ds = datasets.load_from_disk(
        "./data/ms-marco/passage-embeddings/"
        + "+".join(sorted([f"{k}={v}" for k, v in args_dict_without_agg.items()]))
    )

    print("==> Computing doc encodings")
    ds = ds.map(encode_doc, fn_kwargs={"method": args.aggregation_method})

    new_dataset_dir = "./data/ms-marco/doc-embeddings/" + "+".join(
        sorted([f"{k}={v}" for k, v in vars(args).items()])
    )
    print("==> Saving dataset to:", new_dataset_dir)
    ds.save_to_disk(new_dataset_dir)

    print("==> Computing FAISS index")
    ds.add_faiss_index(
        column="doc_embedding",
        metric_type=faiss.METRIC_INNER_PRODUCT,
    )

    print("==> Saving FAISS index to the same folder")
    ds.save_faiss_index("doc_embedding", new_dataset_dir + "/doc-embedding-index.faiss")


def encode_doc(doc: dict, method: Literal["mean", "sum", "first", "max"]) -> dict:
    passage_embeddings = torch.tensor(
        [
            p["passage_embedding"]
            for p in sorted(doc["passages"], key=lambda p: p["passage_id"])
        ]
    )
    if method == "mean":
        doc["doc_embedding"] = passage_embeddings.mean(dim=0).numpy()
    elif method == "sum":
        doc["doc_embedding"] = passage_embeddings.sum(dim=0).numpy()
    elif method == "max":
        doc["doc_embedding"] = passage_embeddings.max(dim=0)[0].numpy()
    elif method == "first":
        doc["doc_embedding"] = passage_embeddings[0].numpy()
    del doc["passages"]
    return doc


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--passage-size", type=int, default=512)
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
        "-m",
        "--aggregation-method",
        type=str,
        choices=["mean", "sum", "first", "max"],
    )
    args = argparser.parse_args()
    main(args)
