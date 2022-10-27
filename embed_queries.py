from re import T
from data import MSMarcoDocs, TREC2019, TREC2020
from sentence_transformers import SentenceTransformer
import torch
from datasets import Dataset, concatenate_datasets
from argparse import ArgumentParser
from pathlib import Path


def embed_queries(args):

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.path_to_model).to(
        device
    )
    print(f"Using device: {model.device}")

    # Load data
    
    ms_marco_docs = MSMarcoDocs()
    trec19 = TREC2019()
    trec20 = TREC2020()
    queries_msmarco = ms_marco_docs.get_queries()["dev"]
    queries_t19 = trec19.get_queries()['test']
    queries_t20 = trec20.get_queries()['test']

    print("==> Computing query embeddings")
    embeddings_msmarco = model.encode([query['text'] for query in queries_msmarco], batch_size=args.batch_size, device=device, show_progress_bar=True)
    embeddings_t19 = model.encode([query['text'] for query in queries_t19], batch_size=args.batch_size, device=device, show_progress_bar=True)
    embeddings_t20 = model.encode([query['text'] for query in queries_t20], batch_size=args.batch_size, device=device, show_progress_bar=True)
    
    # Add doc embeddings to dataset and remove the body column
    embeddings = Dataset.from_dict({"embedding": embeddings_msmarco})
    queries_msmarco = concatenate_datasets([queries_msmarco, embeddings], axis=1)
    embeddings_t19 = Dataset.from_dict({"embedding": embeddings_t19})
    queries_t19 = concatenate_datasets([queries_t19, embeddings_t19], axis=1)
    embeddings_t20 = Dataset.from_dict({"embedding": embeddings_t20})
    queries_t20 = concatenate_datasets([queries_t20, embeddings_t20], axis=1)

    print("==> Saving dataset with query embeddings to disk")
    print(f"== Dir:", args.save_dir)
    queries_msmarco.save_to_disk(Path(f'{args.save_dir}/queries-msmarco-dev'))
    queries_t19.save_to_disk(Path(f'{args.save_dir}/queries-trec19'))
    queries_t20.save_to_disk(Path(f'{args.save_dir}/queries-trec20'))


if __name__ == "__main__":
    
    argparser = ArgumentParser()
    argparser.add_argument(
        "-d",
        "--dataset-file",
        type=str
    )
    
    argparser.add_argument(
        '--path-to-model', default='longformer/final_model/'
    )
    argparser.add_argument(
        '--disable-caching', default=False, type=bool)
    argparser.add_argument('--batch-size', default=16, type=int)
    argparser.add_argument('--save-dir', default='longformer/embeddings')
    args = argparser.parse_args()
    embed_queries(args)