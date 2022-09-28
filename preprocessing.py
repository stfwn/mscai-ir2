from enum import Enum, auto
from typing import Literal


def doc_to_passages(
    doc: dict,
    passage_size: int,
    tokenization_method: Literal["spaces"],
    prepend_title_to_passage: bool,
):
    """Takes a doc dict with at least a 'body' key that contains its text and returns that dict with a 'passage' key added to it."""
    if tokenization_method == "spaces":
        try:
            tokens = doc["body"].split(" ")
        except AttributeError:
            # Some bodies are None, so they have no .split
            tokens = []

        if prepend_title_to_passage:
            doc["passages"] = [
                (doc["title"] + " " if doc["title"] else "")
                + " ".join(tokens[i : i + passage_size])
                for i in range(0, len(tokens), passage_size)
            ]
        else:
            doc["passages"] = [
                " ".join(tokens[i : i + passage_size])
                for i in range(0, len(tokens), passage_size)
            ]
    else:
        raise NotImplementedError(f"Unknown tokenization_method: {tokenization_method}")

    doc["passages"] = [
        {
            "passage_id": doc["doc_id"] + "_" + str(i),
            "passage_body": passage,
        }
        for i, passage in enumerate(doc["passages"])
    ]
    return doc
