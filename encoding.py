from typing import Callable, Literal

import torch
import numpy as np


def encode_doc(
    doc: dict,
    encode_fn: Callable,
    method: Literal["mean_passage_embeddings", "max_passage_embeddings"],
):
    """
    Args:
        - doc: Dict with at least a 'passages' key with a list of {'passage_id': str, 'passage_body': str}
        - encode_fn: Takes a str and returns an embedding.
        - method:
            - mean_passage_embeddings: doc
    """

    if method.endswith("passage_embeddings"):
        passage_encodings = torch.tensor(
            [encode_fn(p["passage_body"]) for p in doc["passages"]]
        )
        if method.startswith("mean"):
            doc["embedding"] = passage_encodings.mean(dim=0)
        elif method.startswith("max"):
            doc["embedding"] = passage_encodings.max(dim=0)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return doc
