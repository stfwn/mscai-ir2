from argparse import ArgumentParser
from pathlib import Path

import datasets


def main(args):
    dsets = []
    args_dict = vars(args)
    for i in range(args.n_shards):
        print("Loading shard:", i)
        args_dict["shard_index"] = i
        dataset_dir = Path("./data/ms-marco/passage-embeddings/") / "+".join(
            sorted([f"{k}={v}" for k, v in args_dict.items()])
        )
        dsets.append(datasets.load_from_disk(dataset_dir))

    print("Merging the shards..")
    ds = datasets.concatenate_datasets(dsets)
    print("Saving to disk")
    ds.save_to_disk(
        "./data/ms-marco/passage-embeddings/passage_size=512+prepend_title_to_passage=True+tokenization_method=model"
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
    args = argparser.parse_args()
    main(args)
