from typing import Callable, Literal
from sentence_transformers import SentenceTransformer, util

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
    passage_bodies = [p["passage_body"] for p in doc["passages"]]
    passage_embeddings = encode_fn(passage_bodies)
    for i, passage_embedding in enumerate(passage_embeddings):
        doc["passages"][i]["passage_embedding"] = passage_embedding
        del doc["passages"][i]["passage_body"]
    return doc


def encode_doc(
    doc: dict,
    method: Literal["mean_passage_embeddings", "max_passage_embeddings"],
):
    """
    Args:
        - doc: Dict with at least a 'passages' key with a list of {'passage_id': str, 'passage_body': str, 'passage_embedding': tensor}
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

def encode_doc_longformer(
    doc: dict,
    model,
):
    doc['passages'] = model.encode(doc['passages']) 
    return doc['passages']