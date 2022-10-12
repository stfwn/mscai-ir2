from pathlib import Path

import faiss
import datasets

base_dir = Path("./data/ms-marco/doc-embeddings/")

for sub_dir in [
    "aggregation_method=first+passage_size=512+prepend_title_to_passage=True+tokenization_method=model",
    "aggregation_method=max+passage_size=512+prepend_title_to_passage=True+tokenization_method=model",
    "aggregation_method=mean+passage_size=512+prepend_title_to_passage=True+tokenization_method=model",
    "aggregation_method=sum+passage_size=512+prepend_title_to_passage=True+tokenization_method=model",
]:
    ds_dir = base_dir / sub_dir
    faiss_path = ds_dir / "doc-embedding-index.faiss"
    print("==> Loading dataset from", ds_dir)
    ds = datasets.load_from_disk(ds_dir)
    print("Computing FAISS index")
    print("Will save faiss index to", faiss_path)
    ds.add_faiss_index(
        column="doc_embedding",
        metric_type=faiss.METRIC_INNER_PRODUCT,
    )
    ds.save_faiss_index("doc_embedding", ds_dir / "doc-embedding-index.faiss")
