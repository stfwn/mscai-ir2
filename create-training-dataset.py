from pathlib import Path

import datasets
import pandas as pd
from tqdm import tqdm

import data


def main():
    ms_marco = data.MSMarcoDocs()
    passage_embeddings = datasets.load_from_disk(
        Path(
            "./data/ms-marco/passage-embeddings/passage_size=512+prepend_title_to_passage=True+tokenization_method=model"
        )
    ).sort("doc_id")

    # Build LUT from doc_id to dataset index
    doc_id_to_idx = {}
    for i, doc_id in enumerate(passage_embeddings["doc_id"]):
        doc_id_to_idx[doc_id] = i

    relevance_scores = ms_marco.get_relevance_scores()["train"].set_index("query_id")
    query_embeddings = datasets.load_from_disk(
        Path("./data/ms-marco/query-embeddings/")
    ).sort("query_id")["train"]

    training_ds = query_embeddings.map(
        assemble_datapoint,
        fn_kwargs={
            "passage_embeddings": passage_embeddings,
            "relevance_scores": relevance_scores,
            "doc_id_to_idx": doc_id_to_idx,
        },
        batched=True,
        batch_size=1000,
        remove_columns=["query_id", "embedding"],
    )
    training_ds.save_to_disk(Path("./data/ms-marco/embedding-training-set"))


def assemble_datapoint(
    queries: dict,
    passage_embeddings: datasets.DatasetDict,
    relevance_scores: pd.DataFrame,
    doc_id_to_idx: dict,  # Indexes passage_embeddings
):
    datapoints = {"query_embedding": [], "doc_passage_embeddings": []}
    not_found_count = 0
    for query_id, query_embedding in zip(
        tqdm(queries["query_id"]), queries["embedding"]
    ):
        try:
            relevant_doc_id = relevance_scores.loc[query_id].doc_id
            relevant_doc = passage_embeddings[doc_id_to_idx[relevant_doc_id]]
            datapoints["query_embedding"].append(query_embedding)
            datapoints["doc_passage_embeddings"].append(
                [p["passage_embedding"] for p in relevant_doc["passages"]]
            )
        except KeyError:
            not_found_count += 1
    print("not found count in batch:", not_found_count)
    return datapoints


if __name__ == "__main__":
    main()
