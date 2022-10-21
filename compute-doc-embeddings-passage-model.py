from argparse import ArgumentParser
from collections import defaultdict
import os
from pathlib import Path

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

    print("==> Loading flattened passage embeddings")
    ds = datasets.load_from_disk(
        Path(
            "./data/ms-marco/passage-embeddings/passage_size=512+prepend_title_to_passage=True+tokenization_method=model+flattened"
        )
    ).sort("passage_id")

    # Hack needed because the dataset was flattened previously.
    idxs_per_doc = defaultdict(list)

    def find_idxs_per_doc(passage_id, idxs):
        for pid, i in zip(passage_id, idxs):
            idxs_per_doc[pid.split("_")[0]].append(i)

    ds.map(
        find_idxs_per_doc,
        with_indices=True,
        batched=True,
        batch_size=10000,
        input_columns="passage_id",
    )
    # Hard to do this in a batched way now because of the flattened structure.
    docs = {"doc_id": [], "doc_embedding": []}
    n_docs = len(idxs_per_doc.keys())
    raise NotImplementedError("There might be docs missing.")
    if int(n_docs / args.n_shards) != n_docs / args.n_shards:
        raise ValueError("Number of docs not cleanly divisible by number of shards")
    from_ = int((args.shard * n_docs / args.n_shards) * n_docs)
    to = int(((args.shard + 1) * n_docs / args.to_doc_idx) * n_docs + 1)
    for doc_id, indices in tqdm(list(idxs_per_doc.items())[from_:to]):
        passage_embeddings = torch.vstack(
            [torch.tensor(e) for e in ds.select(indices)["passage_embedding"]]
        )
        with torch.no_grad():
            docs["doc_embedding"].append(model([passage_embeddings]).squeeze().numpy())
            docs["doc_id"].append(doc_id)
    ds = datasets.Dataset.from_dict(docs)
    new_dataset_dir = f"./data/ms-marco/doc-embeddings/passage-transformer-v1-shard-{args.shard}-of-{args.n_shards}"
    print("==> Saving dataset to:", new_dataset_dir)
    ds.save_to_disk(new_dataset_dir)
    exit(0)

    print("==> Computing FAISS index")
    ds.add_faiss_index(
        column="doc_embedding",
        metric_type=faiss.METRIC_INNER_PRODUCT,
    )
    ds.save_faiss_index("doc_embedding", new_dataset_dir + "/doc-embedding-index.faiss")


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
