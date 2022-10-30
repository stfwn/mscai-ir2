from argparse import ArgumentParser
from pathlib import Path
import datasets
import os
from sklearn.preprocessing import Normalizer
import numpy as np

def normalize_embs(doc):
    normer = Normalizer()
    doc.update({'norm_embedding': normer.fit_transform(np.array(doc['embedding']).reshape(1, -1))})
    return doc


def main(args):
    dsets = []
    for i in range(args.n_shards):
        dataset_dir = Path(f'longformer/embeddings/passage_size-{args.passage_size}-shard_index-{i}') 
        dsets.append(datasets.load_from_disk(dataset_dir))
    ds = datasets.concatenate_datasets(dsets)

    ds = ds.map(normalize_embs)
    ds.remove_columns('embedding')
    
    print("Saving to disk")
    ds.save_to_disk(
        f'longformer/combined-embeddings-normalized'
    )

if __name__ =='__main__':
    argparser = ArgumentParser()
    argparser.add_argument("-n", "--n-shards", type=int, help="Total number of shards.")

    argparser.add_argument("--passage-size", type=int, default=1000)
    argparser.add_argument(
        "--tokenization-method",
        type=str,
        choices=["model", "spaces"],
        default="model",
    )
    argparser.add_argument(
        "--prepend-title-to-passage", action="store_true", default=True
    )

    args = argparser.parse_args()
    main(args)