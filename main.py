import os

from sentence_transformers import SentenceTransformer, util

import config
from data import MSMarcoDocs
import encoding
import preprocessing


def main():
    print("=" * 10)
    print("Running in stage:", config.stage)
    print("=" * 10)

    # Initialize MS Marco
    ms_marco_docs = MSMarcoDocs()
    queries = ms_marco_docs.get_queries()
    docs = ms_marco_docs.get_docs()

    if config.stage == "DEV":
        docs = docs.train_test_split(train_size=10, test_size=1, shuffle=True)["train"]

    docs = docs.map(
        preprocessing.doc_to_passages,
        num_proc=os.cpu_count(),
        fn_kwargs={
            "passage_size": config.passage_size,
            "tokenization_method": config.tokenization_method,
            "prepend_title_to_passage": config.prepend_title_to_passage,
        },
        keep_in_memory=config.use_cache,
    )

    # Initialize model to compute passage embeddings
    model = SentenceTransformer("sentence-transformers/msmarco-distilbert-dot-v5")

    # Compute document embeddings
    docs = docs.map(
        encoding.encode_doc,
        fn_kwargs={
            "encode_fn": model.encode,
            "method": "mean_passage_embeddings",
        },
        keep_in_memory=config.use_cache,
    )

    try:
        docs.load_faiss_index("embedding", "./data/ms-marco/document-embeddings.faiss")
    except:
        docs.add_faiss_index(column="embedding")
        docs.save_faiss_index("embedding", "./data/ms-marco/document-embeddings.faiss")

    # Score a query
    scores, retrieved_docs = docs.get_nearest_examples(
        "embedding", model.encode("Here's an example query"), k=10
    )


if __name__ == "__main__":
    main()
