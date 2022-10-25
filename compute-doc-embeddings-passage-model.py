from argparse import ArgumentParser
from collections import defaultdict
import os
from pathlib import Path
from typing import Callable

import datasets
import faiss
import torch
from torch import nn

from tqdm import tqdm

import models


def main(args):
    print("==> Initializing model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PassageModelWrapper(models.PassageTransformer())
    model.load_state_dict(
        torch.load(
            "./models/passage-transformer-v1/checkpoint-400000/pytorch_model.bin",
            map_location=device,
        )
    )
    model = model.model.to(device)
    model.eval()

    print("==> Loading passage embeddings")
    ds = datasets.load_from_disk(
        Path(
            "./data/ms-marco/passage-embeddings/passage_size=512+prepend_title_to_passage=True+tokenization_method=model"
        )
    )

    ds = ds.map(
        encode,
        fn_kwargs={"encode_fn": model, "device": device},
        batched=True,
        batch_size=64,
    )
    new_dataset_dir = f"./data/ms-marco/doc-embeddings/passage-transformer-v1"
    print("==> Saving dataset to:", new_dataset_dir)
    ds.save_to_disk(new_dataset_dir)

    print("==> Computing FAISS index")
    ds.add_faiss_index(
        column="doc_embedding",
        metric_type=faiss.METRIC_INNER_PRODUCT,
    )
    ds.save_faiss_index("doc_embedding", new_dataset_dir + "/doc-embedding-index.faiss")


def encode(batch, encode_fn: Callable, device: str):
    docs_passages = [
        torch.vstack(
            [torch.tensor(p["passage_embedding"], device=device) for p in doc_passages]
        )
        for doc_passages in batch["passages"]
    ]
    with torch.no_grad():
        doc_embeddings = encode_fn(docs_passages).cpu().numpy()
    batch["doc_embedding"] = list(doc_embeddings)
    del batch["passages"]
    del batch["url"]
    del batch["title"]
    return batch


class PassageModelWrapper(nn.Module):
    """This would be handled in the collate function but the HuggingFace
    Trainer class removes columns from the training set if the column names
    don't match with any model args."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, query_embedding, doc_passage_embeddings):
        return self.model(doc_passage_embeddings)


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--n-shards", type=int)
    argparser.add_argument("--shard", type=int)
    args = argparser.parse_args()
    main(args)
