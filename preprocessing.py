from typing import Any, Callable, Literal
import config
from sentence_transformers.SentenceTransformer import SentenceTransformer


def doc_to_passages(
    doc: dict,
    model: SentenceTransformer,
    passage_size: int,
    tokenization_method: Literal["spaces", "model"],
    prepend_title_to_passage: bool,
):
    """Takes a doc dict with at least a 'body' key that contains its text and
    returns that dict with a 'passage' key added to it. Not batched because we
    don't have to do it often."""
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
    elif tokenization_method == "model":
        # Chop off [CLS] at the start and [SEP] at the end
        body_input_ids = model.tokenizer(doc["body"])["input_ids"][1:-1]

        if prepend_title_to_passage:
            # Chop off [CLS] at the start and [SEP] at the end
            if doc["title"]:
                MAX_TITLE_LEN = 50
                title_input_ids = model.tokenizer(doc["title"])["input_ids"][1:-1][
                    :MAX_TITLE_LEN
                ]
            else:
                title_input_ids = []
            # Take 20 margin
            passage_chunk_len = passage_size - len(title_input_ids) - 20
            if passage_chunk_len <= 0:
                raise ValueError(
                    "No room for passage text in passage chunk len. Increase passage size arg."
                )
            doc["passages"] = [
                model.tokenizer.decode(
                    title_input_ids[1:-1] + body_input_ids[i : i + passage_chunk_len]
                )
                for i in range(0, len(body_input_ids), passage_chunk_len)
            ]
        else:
            # Take 20 margin
            passage_chunk_len = passage_size - 20
            doc["passages"] = [
                model.tokenizer.decode(
                    title_input_ids[1:-1] + body_input_ids[i : i + passage_chunk_len]
                )
                for i in range(0, len(body_input_ids), passage_chunk_len)
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
    del doc["body"]
    return doc

def doc_to_longformer_input(
    doc: dict,
    passage_size: int,
):
    MAX_TITLE_LEN = 50
    if doc["title"]:
        doc['title'] = doc['title'][:MAX_TITLE_LEN]
    else: doc['title'] = ''
    
    if doc['body']:
        doc['body'] = doc['title'] + doc['body'] if doc['title'] else doc['body']
        if len(doc['body']) > passage_size:
            doc['body'] = doc['body'][:passage_size]
    
    return doc
    