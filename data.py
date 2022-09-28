import os

import datasets
import pandas as pd


class MSMarcoDocs:
    """Singleton to namespace methods related to MS Marco Docs.

    Improvements would be to move hardcoded file paths to the arguments of an
    __init__ method and use it as an object.
    """

    def get_queries(self) -> datasets.DatasetDict:
        queries = datasets.load_dataset(
            "csv",
            data_files={
                "train": "./data/ms-marco/msmarco-doctrain-queries.tsv",
                "dev": "./data/ms-marco/msmarco-docdev-queries.tsv",
            },
            name="ms-marco-docs-queries",
            encoding="utf8",
            delimiter="\t",
            column_names=["query_id", "text"],
            features=datasets.Features(
                {
                    "query_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                }
            ),
        )
        return queries

    def get_docs(self) -> datasets.DatasetDict:
        return datasets.load_dataset(
            "csv",
            data_files="./data/ms-marco/msmarco-docs.tsv",
            name="ms-marco-docs",
            encoding="utf8",
            delimiter="\t",
            column_names=["doc_id", "url", "title", "body"],
            split="train",
            features=datasets.Features(
                {
                    "doc_id": datasets.Value("string"),
                    "url": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "body": datasets.Value("string"),
                }
            ),
        )

    def get_relevance_scores(self) -> pd.DataFrame:
        scores = {
            "train": pd.read_csv(
                "./data/ms-marco/msmarco-doctrain-qrels.tsv",
                delimiter=" ",
                names=["query_id", "_", "doc_id", "relevant"],
            ),
            "dev": pd.read_csv(
                "./data/ms-marco/msmarco-docdev-qrels.tsv",
                delimiter=" ",
                names=["query_id", "_", "doc_id", "relevant"],
            ),
        }
        for k in scores:
            scores[k].query_id = scores[k].query_id.astype(str)
        return scores
