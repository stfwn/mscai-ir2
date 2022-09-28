from typing import Callable, Literal

import torch
import numpy as np


def encode_passages(doc: dict, encode_fn: Callable):
    """
    Args:
        - doc: Dict with at least key 'passages' with a list of {'passage_id': str, 'passage_body': str}
        - encode_fn: Takes a str and returns an embedding
    Returns:
        doc dict with a 'passage_embedding' key-value pair in each doc['passages']
    """
    passages = []
    for i, passage in enumerate(doc["passages"]):
        passage["passage_embedding"] = encode_fn(passage["passage_body"])
        passages.append(passage)
    doc["passages"] = passages
    return doc


def encode_doc(
    doc: dict,
    encode_fn: Callable,
    method: Literal["mean_passage_embeddings", "max_passage_embeddings"],
):
    """
    Args:
        - doc: Dict with at least a 'passages' key with a list of {'passage_id': str, 'passage_body': str, 'passage_embedding': tensor}
        - encode_fn: Takes a str and returns an embedding.
        - method:
            - mean_passage_embeddings: doc
    """

    if method.endswith("passage_embeddings"):
        passage_embeddings = torch.tensor(
            [p["passage_embedding"] for p in doc["passages"]]
        )
        if method.startswith("mean"):
            doc["embedding"] = passage_embeddings.mean(dim=0)
        elif method.startswith("max"):
            doc["embedding"] = passage_embeddings.max(dim=0)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return doc
