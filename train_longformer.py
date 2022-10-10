from venv import create
from datasets import Dataset
from sentence_transformers import SentenceTransformer, LoggingHandler, util, models, evaluation, losses, InputExample
from torch.utils.data import DataLoader
import argparse
import tqdm
import random
import config
from data import MSMarcoDocs

# We create a custom MSMARCO dataset that returns triplets (query, positive, negative)
# on-the-fly based on the information from the mined-hard-negatives jsonl file.
class MSMARCODatasetLongFormer(Dataset):
    def __init__(self, queries, corpus):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus

        # for qid in self.queries:
        #     self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
        #     self.queries[qid]['neg'] = list(self.queries[qid]['neg'])
        #     random.shuffle(self.queries[qid]['neg'])

    def __getitem__(self, item):
        query = self.queries[self.queries_ids[item]]
        query_text = query['query']
        print('\n')
        print(item)
        print('\n')

        # print(self.corpus)
        # print(type(self.corpus))
        print(self.corpus.keys())
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

    if config.stage == "DEV":
        docs = docs.train_test_split(train_size=50, test_size=1, shuffle=False)["train"]
        queries["train"] = queries["train"].train_test_split(
            train_size=50, test_size=1, shuffle=False
        )["train"]
        queries["dev"] = queries["dev"].train_test_split(
            train_size=50, test_size=1, shuffle=False
        )["train"]
        

    print("Preparing docs")
    # Prepare docs, append title to body. Docs longer than max input lenght get truncated when passed to the model so
    # no need to check for that here. Docs_prepped = {doc_id: title + doc}
    docs_prepped = {}
    for doc in tqdm.tqdm(docs):
        # id = doc['doc_id']
        if tokenization_method == "spaces":
            try:
                tokens = doc["body"].split(" ")
            except AttributeError:
                # Some bodies are None, so they have no .split
                tokens = []
            if prepend_title_to_doc:
                docs_prepped[doc['doc_id']] = [
                    (doc['title'] + ' ' if doc['title'] else '')
                    + " ".join(tokens) 
                ]
        #TODO: Implement model tokenization. Faster during training and we probably won't
        # be updating the weights of the tokenizer anyway
        else:
            raise NotImplementedError(f"Unknown tokenization_method: {tokenization_method}")
        
        # docs_prepped[id] = docs_prepped

    ids_train = [query['query_id'] for query in queries['train']]
    ids_dev = [query['query_id'] for query in queries['dev']]

    print("Preparing train queries")
    queries_train = {}
    for i, id in tqdm.tqdm(enumerate(ids_train)):
        relevant_doc_ids = relevance_scores['train'].loc[relevance_scores['train']['query_id'] == id]['doc_id']
        neg_ids = list(ids_train)
        neg_ids.remove(id)
        # Sample 50 random docs as negatives
        queries_train[id] = {'query': queries['train']['text'][i],
                             'pos': list(relevant_doc_ids),
                             'neg': random.choices(neg_ids, k=50)}
    print('Preparing dev queries')
    
    queries_dev = {}
    for i, id in tqdm.tqdm(enumerate(ids_dev)):
        relevant_doc_ids = relevance_scores['dev'].loc[relevance_scores['dev']['query_id'] == id]['doc_id']
        neg_ids = list(ids_dev)
        neg_ids.remove(id)
        # Sample 50 random docs as negatives
        queries_train[id] = {'query': queries['dev']['text'][i],
                             'pos': list(relevant_doc_ids),
                             'neg': random.choices(neg_ids, k=50)}

    return docs_prepped, queries_train, queries_dev

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_batch_size", default=64, type=int)
    parser.add_argument("--max_seq_length", default=300, type=int)
    parser.add_argument("--model_name", default='allenai/longformer-base-4096')
    parser.add_argument("--max_passages", default=0, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--pooling", default="mean")
    parser.add_argument("--negs_to_use", default=None, help="From which systems should negatives be used? Multiple systems seperated by comma. None = all")
    parser.add_argument("--warmup_steps", default=1000, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--num_negs_per_system", default=5, type=int)
    parser.add_argument("--use_pre_trained_model", default=False, action="store_true")
    parser.add_argument("--use_all_queries", default=False, action="store_true")
    parser.add_argument("--ce_score_margin", default=3.0, type=float)
    parser.add_argument("--model_save_path", default='longformer/')
    args = parser.parse_args()

    print(args)

    docs_prepped, queries_train, queries_dev = create_dataset()
    train_dataset = MSMARCODatasetLongFormer(queries_train, docs_prepped)
    print(docs_prepped)
    # For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size)
    model = SentenceTransformer(args.model_name)

    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
            epochs=args.epochs,
            warmup_steps=args.warmup_steps,
            use_amp=True,
            checkpoint_path=args.model_save_path,
            checkpoint_save_steps=len(train_dataloader),
            optimizer_params = {'lr': args.lr},
            )

    # Save the model
    model.save(args.model_save_path)