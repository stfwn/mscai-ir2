from typing import Callable
from argparse import ArgumentParser
import os
import json
import time
import sys

from datasets import Dataset
from sentence_transformers import SentenceTransformer, util
import torch

import config
from data import MSMarcoDocs, TREC2019, TREC2020
import encoding
import preprocessing


def main(args):

    num_shards = args.n_shards
    k = args.fold_k

    START = int((k/num_shards)*3201821)
    END = int(((k+1)/num_shards)*3201821)

    print("===>STARTING SHARTING")
    docs = Dataset.load_from_disk('./data/ms-marco/passage-embeddings/passage_size=512+prepend_title_to_passage=True+tokenization_method=model/')
    print('loaded data')

    # flatten data
    ps_data = {
        'passage_id': [],
        'passage_embedding': []
    }
    a = time.time()
    for i in range(3201821):
        if i < START:
            pass
        else:
            doc = docs[i]
            if i % 100 == 0:
                b = time.time()
                print(f"Flattening doc {i} out of {len(docs)}; time {b-a}")
            for passage in doc['passages']:
                ps_data['passage_id'].append(passage['passage_id'])
                ps_data['passage_embedding'].append(passage['passage_embedding'])
            if i >= END:
                passages = Dataset.from_dict(ps_data)
                passages.save_to_disk('./data/ms-marco/passage-embeddings/passage_size=512+prepend_title_to_passage=True+tokenization_method=model+flattened_{}/'.format(k))
                print('Saved flattened passage dataset to disk')
                break



if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument(
        "-n",
        "--n-shards",
        type=int,
        help="Number of Shards Total",
    )
    argparser.add_argument(
        "-k",
        "--fold-k",
        type=int,
        help="Which shard to compute",
    )
    args = argparser.parse_args()
    main(args)
