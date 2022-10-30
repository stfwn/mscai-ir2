from pathlib import Path
from typing import Callable

from sentence_transformers import SentenceTransformer
import torch

from data import MSMarcoDocs


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/msmarco-distilbert-dot-v5").to(
        device
    )
    ds = MSMarcoDocs().get_queries()
    ds = ds.map(encode_query, fn_kwargs={"encode_fn": model.encode})
    ds.save_to_disk(Path("./data/ms-marco/query-embeddings/"))


def encode_query(q, encode_fn: Callable):
    q["embedding"] = encode_fn(q["text"])
    del q["text"]
    return q


if __name__ == "__main__":
    main()
