from typing import Callable
from argparse import ArgumentParser
import os
import json
import time

from datasets import Dataset
from sentence_transformers import SentenceTransformer, util
import torch

import config
from data import MSMarcoDocs, TREC2019, TREC2020
import encoding
import preprocessing

def main():

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device {device}")

    # load data
    try:
        passages = load_from_disk('./data/ms-marco/passage-embeddings/passage_size=512+prepend_title_to_passage=True+tokenization_method=model+flattened/')
    except:
        print("===>STARTING SHARTING")
        num_shards = 30
        for k in range(1,num_shards):
            print(f"SHARD {k}/{num_shards}")
            # load only shard
            docs = Dataset.from_dict(
                    Dataset.load_from_disk(
                        './data/ms-marco/passage-embeddings/passage_size=512+prepend_title_to_passage=True+tokenization_method=model/'
                    )[int((k/num_shards)*3201821):int((k+1/num_shards)*3201821)]
                )
            # flatten data
            ps_data = {
                'passage_id': [],
                'passage_embedding': []
            }
            a = time.time()
            for i, doc in enumerate(docs):
                if i % 100 == 0:
                    b = time.time()
                    print(f"Flattening doc {i} out of {len(docs)}; time {b-a}")
                for passage in doc['passages']:
                    ps_data['passage_id'].append(passage['passage_id'])
                    ps_data['passage_embedding'].append(passage['passage_embedding'])

            passages = Dataset.from_dict(ps_data)
            passages.save_to_disk('./data/ms-marco/passage-embeddings/passage_size=512+prepend_title_to_passage=True+tokenization_method=model+flattened_{}/'.format(k))
            print('Saved flattened passage dataset to disk')



if __name__ == "__main__":
    main()





