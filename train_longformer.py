from venv import create
from datasets import Dataset
from sentence_transformers import SentenceTransformer, LoggingHandler, util, models, evaluation, losses, InputExample
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import random
import config
from data import MSMarcoDocs
from transformers import LongformerTokenizerFast 
from torch.utils.data.dataloader import default_collate

# Create a custom MSMARCO dataset that returns triplets (query, positive, negative)
# on-the-fly based on the information from the mined-hard-negatives jsonl file.
class MSMARCODatasetLongFormer(Dataset):
    def __init__(self, queries, corpus):

        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus

    def __getitem__(self, item):
        query = self.queries[self.queries_ids[item]]
        query_text = query['query']

        pos_id = query['pos'].pop(0)    #Pop positive and add at end
        pos_text = self.corpus[pos_id]
        query['pos'].append(pos_id)

        neg_id = query['neg'].pop(0)    #Pop negative and add at end
        neg_text = self.corpus[neg_id]
        query['neg'].append(neg_id)
        return InputExample(texts=[query_text, pos_text, neg_text])

    def __len__(self):
        return len(self.queries)

def create_dataset(tokenization_method='spaces', prepend_title_to_doc=True):
    #TODO: remove passages longer than max input length
    # Initialize MS Marco
    ms_marco_docs = MSMarcoDocs()
    docs = ms_marco_docs.get_docs()
    queries = ms_marco_docs.get_queries()
    relevance_scores = ms_marco_docs.get_relevance_scores()
    print('STAGE = ' + config.stage)
    if config.stage == "DEV":

        queries["train"] = queries["train"].train_test_split(
            train_size=10, test_size=1, shuffle=False
        )["train"]
        queries["dev"] = queries["dev"].train_test_split(
            train_size=10, test_size=1, shuffle=False
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

    # Prepare docs, append title to body. Docs longer than max input lenght get truncated when passed to the model so
    # no need to check for that here. Docs_prepped = {doc_id: title + doc}
    print("Preparing docs")
    docs_prepped = {} 
    for doc in tqdm(docs):

        if tokenization_method == "spaces":
            doc_id = doc['doc_id']

            if prepend_title_to_doc:
                if doc['body'] and doc['title']:
                    docs_prepped[doc_id] = doc['title'] + doc['body'] if doc['title'] else doc['body']
                    if len(docs_prepped[doc_id]) > config.passage_size_longformer:
                        docs_prepped[doc_id] = docs_prepped[doc_id][:config.passage_size_longformer]
            else:
                docs_prepped[doc_id] = doc['body']
        #TODO: Implement model tokenization. Faster during training and we probably won't
        # be updating the weights of the tokenizer anyway
        
        else:
            raise NotImplementedError(f"Unknown tokenization_method: {tokenization_method}")

    # Link the queries to documents. Queries are stored in a nested dict: {query_id: {query: [...], pos, [...], neg [...]}}
    # The pos and neg keys contain doc_ids for relevant and irrelevant docs.
    print("Preparing train queries")
    queries_train = {}
    for i, query in tqdm(enumerate(queries['train'])):
        id = query['query_id']
        relevant_doc_ids = relevance_scores['train'].loc[relevance_scores['train']['query_id'] == id]['doc_id']
        neg_ids = set(docs_prepped.keys())
        neg_ids = list(neg_ids - set(relevant_doc_ids))
        # Sample 50 random docs as negatives
        queries_train[id] = {'query': queries['train']['text'][i],
                             'pos': list(relevant_doc_ids),
                             'neg': random.choices(neg_ids, k=50)}

    print('Preparing dev queries')
    queries_dev = {}
    for i, query in tqdm(enumerate(queries['dev'])):
        id = query['query_id']
        relevant_doc_ids = relevance_scores['dev'].loc[relevance_scores['dev']['query_id'] == id]['doc_id']
        neg_ids = set(docs_prepped.keys())
        neg_ids = list(neg_ids - set(relevant_doc_ids))
        # Sample 50 random docs as negatives
        queries_train[id] = {'query': queries['dev']['text'][i],
                             'pos': list(relevant_doc_ids),
                             'neg': random.choices(neg_ids, k=50)}

    return docs_prepped, queries_train, queries_dev

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_batch_size", default=64, type=int)
    parser.add_argument("--model_name", default='allenai/longformer-base-4096')
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--warmup_steps", default=1000, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--num_negs_per_system", default=5, type=int)
    parser.add_argument("--use_pre_trained_model", default=False, action="store_true")
    parser.add_argument("--use_all_queries", default=False, action="store_true")
    parser.add_argument("--model_save_path", default='longformer/')
    parser.add_argument("--device", default='cuda')
    args = parser.parse_args()

    docs_prepped, queries_train, queries_dev = create_dataset()
    #TODO: send data to device?

    train_dataset = MSMARCODatasetLongFormer(queries_train, docs_prepped)

    # For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size)

    model = SentenceTransformer(args.model_name, device=args.device)

    #TODO: check for good params for fine tuning
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
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