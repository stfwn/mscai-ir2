from argparse import ArgumentParser
from genericpath import exists
from pathlib import Path
import datasets
import os
import faiss

def main(args):
    dsets = []
    args_dict = vars(args)
    for i in range(args.n_shards):
        print("Loading shard:", i)
        args_dict["shard_index"] = i
        if args.normalized:
            dataset_dir = Path(f'longformer/embeddings/passage_size-{args.passage_size}-shard_index-{args.shard_index}-normalized')
        else:
            dataset_dir = Path(f'longformer/embeddings/passage_size-{args.passage_size}-shard_index-{args.shard_index}')
      
        dsets.append(datasets.load_from_disk(dataset_dir))

    print("Merging the shards..")
    ds = datasets.concatenate_datasets(dsets)

    print("Creating FAISS index")
    ds.add_faiss_index(column='embedding', metric_type=faiss.METRIC_INNER_PRODUCT)

    if args.normalized:
        save_dir = 'longformer/combined-embeddings-normalized/'
    else:
        save_dir = 'longformer/combined-embeddings/'
    
    os.makedirs(save_dir, exist_ok=True)
    ds.save_faiss_index('embedding', f'{save_dir}/index.faiss')
    ds.drop_index('embedding')

    print("Saving to disk")
    ds.save_to_disk(
        save_dir
    )


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("-n", "--n-shards", type=int, help="Total number of shards.")
    argparser.add_argument("--passage-size", type=int, default=512)
    argparser.add_argument(
        "--tokenization-method",
        type=str,
        choices=["model", "spaces"],
        default="model",
    )
    argparser.add_argument(
        "--prepend-title-to-passage", action="store_true", default=True
    )
    argparser.add_argument(
        '--normalized', default=True, type=bool
    )
    args = argparser.parse_args()
    main(args)
