from doctest import DocTestSuite
import os
from venv import create
from datasets import Dataset
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import random
import config
from data import MSMarcoDocs
from transformers import LongformerTokenizerFast 
from torch.utils.data.dataloader import default_collate
import pandas as pd
import numpy as np
import pickle

# Create a custom MSMARCO dataset that returns triplets (query, positive, negative)
class MSMARCODatasetLongFormer(Dataset):
    def __init__(self, indexes, corpus, queries):
        
        self.indexes = indexes
        self.queries = queries
        # self.query_ids = list(self.indexes['query_id'].keys())
        self.corpus = corpus
        # Neg_counter keeps track of which of the 50 negatives to use for all of the queries
        self.neg_counter = {i: 1 for i in range(len(queries))}

    def __getitem__(self, item):
        query_id = self.indexes['query_id'][item]
        query_text = self.queries[query_id]
       
        pos_id = self.indexes['pos'][item]
        pos_text = self.corpus[pos_id]

        # Use the counter to keep track of which negative we've had already
        neg_iteration = f'neg_{self.neg_counter[item]}'
        self.neg_counter[item] += 1 if self.neg_counter[item] < 50 else 0 # Reset after all 50 negatives are used
        neg_id = self.indexes[neg_iteration][item]
        neg_text = self.corpus[neg_id]

        return InputExample(texts=[query_text, pos_text, neg_text])

    def __len__(self):
        return len(self.indexes['query_id'])

def create_dataset(tokenization_method='spaces', prepend_title_to_doc=True, num_negs=50, mode='save',
                    data_dir='longformer/data', save_indicator=''):

    if mode == 'load':
        data_dir += save_indicator
        with open(f'{data_dir}/docs_prepped.pkl', 'rb') as f:
            docs_prepped = pickle.load(f)
        with open(f'{data_dir}/train_queries.pkl', 'rb') as f:
            train_queries = pickle.load(f)
        with open(f'{data_dir}/dev_queries.pkl', 'rb') as f:
            dev_queries = pickle.load(f)
        with open(f'{data_dir}/train_indexes.pkl', 'rb') as f:
            train_indexes = pickle.load(f)
        with open(f'{data_dir}/dev_indexes.pkl', 'rb') as f:
            dev_indexes = pickle.load(f)
        return docs_prepped, train_indexes, dev_indexes, train_queries, dev_queries
    
    #TODO: remove passages longer than max input length
    # Initialize MS Marco
    ms_marco_docs = MSMarcoDocs()
    docs = ms_marco_docs.get_docs()
    queries = ms_marco_docs.get_queries()
    relevance_scores = ms_marco_docs.get_relevance_scores()
    print('STAGE = ' + config.stage)
    if config.stage == "DEV":

        queries["train"] = queries["train"].train_test_split(
            train_size=25, test_size=1, shuffle=False
        )["train"]
        queries["dev"] = queries["dev"].train_test_split(
            train_size=25, test_size=1, shuffle=False
        )["train"]

        corresponding_docs = []
        for query in queries['train']:
            corresponding_docs.append(relevance_scores['train'].loc[
                                                    relevance_scores['train']['query_id'] == query['query_id']
                                                                    ]['doc_id'].values[0])
        for query in queries['dev']:
            corresponding_docs.append(relevance_scores['dev'].loc[
                                                    relevance_scores['dev']['query_id'] == query['query_id']
                                                                    ]['doc_id'].values[0])

        print('Filtering docs')
        docs = docs.filter(lambda x: x['doc_id'] in corresponding_docs)
        relevance_scores['train'] = relevance_scores['train'][relevance_scores['train']['doc_id'].apply(lambda x: x in corresponding_docs)]
        relevance_scores['dev'] = relevance_scores['dev'][relevance_scores['dev']['doc_id'].apply(lambda x: x in corresponding_docs)]

    
    # Prepare docs, append title to body. Docs longer than max input lenght get truncated when passed to the model so
    # no need to check for that here. Docs_prepped = {doc_id: title + doc}
    print("Preparing docs")
    docs_prepped = {} 
    remove_ids = []
    for doc in tqdm(docs):

        if tokenization_method == "spaces":
            doc_id = doc['doc_id']

            if prepend_title_to_doc:
                if doc['body'] and doc['title']:
                    docs_prepped[doc_id] = doc['title'] + doc['body'] if doc['title'] else doc['body']
                    if len(docs_prepped[doc_id]) > config.passage_size_longformer:
                        docs_prepped[doc_id] = docs_prepped[doc_id][:config.passage_size_longformer]
                
                # If doc has no body or title, remove the queries using that doc from the training data
                # Store the doc ids to remove for now so we can remove all rows at once later
                else:
                    remove_ids.append(doc_id)
            else:
                docs_prepped[doc_id] = doc['body']
        #TODO: Implement model tokenization. Faster during training and we probably won't
        # be updating the weights of the tokenizer anyway

        else:
            raise NotImplementedError(f"Unknown tokenization_method: {tokenization_method}")

    # Remove docs without body or title
    relevance_scores['train'] = relevance_scores['train'][~relevance_scores['train']['doc_id'].isin(remove_ids)]
    relevance_scores['dev'] = relevance_scores['dev'][~relevance_scores['dev']['doc_id'].isin(remove_ids)]

    # Link the queries to documents. 
    # Queries are stored in a dict: {query_id: [qid_1, qid_2, ... qid_n],
    #                                           pos: [did_1, did_2, ... did_n]
    #                                           neg_1: [did_1 ... did_n], neg_2: [did_1 ... did_n] ... neg_50: [did_1 ... did_n]
    #                               }
    # The pos and neg keys contain doc_ids for relevant and irrelevant docs.
    train_indexes = {"query_id": list(relevance_scores['train']['query_id']),
        "pos": list(relevance_scores['train']['doc_id'])
    }
    dev_indexes = {"query_id": list(relevance_scores['dev']['query_id']),
        "pos": list(relevance_scores['dev']['doc_id'])
    }

    for i in tqdm(range(1, num_negs+1)):
        train_indexes[f'neg_{i}'] = list(np.roll(relevance_scores['train']['doc_id'], i))
        dev_indexes[f'neg_{i}'] = list(np.roll(relevance_scores['dev']['doc_id'], i))
    
    train_queries = {row['query_id']: row['text'] for row in queries['train']}
    dev_queries = {row['query_id']: row['text'] for row in queries['dev']}

    if mode == 'save':
        data_dir += save_indicator
        print('Saving docs')
        os.makedirs(data_dir, exist_ok=True)
        with open(f'{data_dir}/docs_prepped.pkl', 'wb') as f:
            pickle.dump(docs_prepped, f)
        with open(f'{data_dir}/train_queries.pkl', 'wb') as f:
            pickle.dump(train_queries, f)
        with open(f'{data_dir}/dev_queries.pkl', 'wb') as f:
            pickle.dump(dev_queries, f)
        with open(f'{data_dir}/train_indexes.pkl', 'wb') as f:
            pickle.dump(train_indexes, f)
        with open(f'{data_dir}/dev_indexes.pkl', 'wb') as f:
            pickle.dump(dev_indexes, f)
    return docs_prepped, train_indexes, dev_indexes, train_queries, dev_queries

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_batch_size", default=64, type=int)
    parser.add_argument("--model_name", default='allenai/longformer-base-4096')
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--warmup_steps", default=1000, type=int)
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("--num_negs_per_system", default=5, type=int)
    parser.add_argument("--use_pre_trained_model", default=False, action="store_true")
    parser.add_argument("--use_all_queries", default=False, action="store_true")
    parser.add_argument("--model_save_path", default='longformer/')
    parser.add_argument("--mode", default='load')
    parser.add_argument("--data_dir", default='longformer/data')
    parser.add_argument("--save_indicator", default='')
    parser.add_argument("--device", default='cuda')
    args = parser.parse_args()

    docs_prepped, train_indexes, dev_indexes, train_queries, dev_queries = create_dataset(mode=args.mode,
                                                                                data_dir=args.data_dir, 
                                                                                save_indicator=args.save_indicator)
    #TODO: send data to device?

    train_dataset = MSMARCODatasetLongFormer(train_indexes, docs_prepped, train_queries)

    # For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size)
    model = SentenceTransformer(args.model_name, device=args.device)

    # See https://www.sbert.net/examples/training/ms_marco/README.html
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    #TODO: check for good params for fine tuning
    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
            epochs=args.epochs,
            warmup_steps=args.warmup_steps,
            use_amp=True,
            checkpoint_path=args.model_save_path,
            checkpoint_save_steps=len(train_dataloader),
            optimizer_params = {'lr': args.lr}
            )

    # Save the model
    model.save(args.model_save_path)