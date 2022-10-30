from argparse import ArgumentParser

import datasets
from datasets import Dataset

def main(args):
    passages = Dataset.load_from_disk(
                './data/ms-marco/passage-embeddings/passage_size=512+prepend_title_to_passage=True+tokenization_method=model+flattened_0/'
            )
    for k in range(1,args.n_shards):
        tmp = Dataset.load_from_disk(
                './data/ms-marco/passage-embeddings/passage_size=512+prepend_title_to_passage=True+tokenization_method=model+flattened_{}/'
                .format(k)
            )
        passages = datasets.concatenate_datasets([passages, tmp])

    passages.save_to_disk(
            './data/ms-marco/passage-embeddings/passage_size=512+prepend_title_to_passage=True+tokenization_method=model+flattened/'
        )


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument(
        "-n",
        "--n-shards",
        type=int,
        help="Number of Shards Total",
    )
    args = argparser.parse_args()
    main(args)