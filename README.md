## Initialize environment and obtain the raw data

```bash
# Set up virtual env
python -m venv env
source env/bin/activate
pip install -r requirements.txt

# Download the data from Microsoft
sh get-data.sh
```

## Computing Passage Representation Dataset

1. Follow the steps for initializing your environment and obtaining the raw data.
2. Compute embeddings for all passages in shards

```bash
n=20
for i in {0..n}
do
    python compute-passage-embeddings-shard.py -n $n -i $i
done
```

3. Merge the shards

```bash
n=20
python merge-passage-embeddings-shards.py -n $n
```

## Training passage model

1. Follow the steps for initializing your environment and obtaining the raw data.
2. Follow the steps for computing the passage representation dataset
3. Create the training dataset using `python create-training-dataset.py`
4. Train the passage model using `python train-passage-model.py --version 1`
5. Rank the resulting doc embeddings using `rank.py` with the right
   arguments: 
   ```bash
   python rank.py
          -d data/ms-marco/doc-embeddings/passage-transformer-v1
          -f data/ms-marco/doc-embeddings/passage-transformer-v1index.faiss
   ```
6. Evaluate the rankings with `evaluate.py` using `-q (path to qrels)` and `-r (path to ranking)`
   
## Computing Score MaxP
1. Follow the steps for initializing your environment and obtaining the raw data.
2. Follow the steps for computing the passage representation dataset
3. Flatten the Passage Representation dataset with:
```bash
n=20
for k in {0..n}
do
    python flatten_Pdataset.py -n $n -k $k
done
```

4. Merge the shards with:
```bash
python stich_Pdatasets.py -n 20
```

5. Compute the rankings with `passage_rank.py`
6. Evaluate and compute the positional bias histograms using `eval_score_max.ipynb`

## Computing Embedding MaxP, MeanP, SumP and FirstP
1. Follow the steps for initializing your environment and obtaining the raw data.
2. Compute the embedding pooling dataset using `compute-passage-pooling-doc-embedding-dataset.py` with `-m` in `[mean, max, sum, first]`
3. Rank resulting doc embeddings using `rank.py` with `-d (path to the dataset)` and `-f (path to the faiss index file)`
4. Evaluate the rankings with `evaluate.py` using `-q (path to qrels)` and `-r (path to ranking)`


## Train longformer and compute rankings
1. Follow the steps for initializing your environment and obtaining the raw data.
2. Fine-tune the longformer using `train-longformer.py` with your preferred batch size `--train-batch-size (batch size)`, the path to the data files `--data-dir (path to the dataset)` and `--mode create-dataset`
3. Use the fine-tuned model to create document embeddings with `compute-embeddings-longformer.py` and the parameters `--train-batch-size (batch size)` `--model-path longformer/final_model`, `--n-shards num-shards` and `--shard-index [0 to num-shards -1]`. `--n-shards can` be used to distribute the inference over multiple compute nodes, set it to the number of nodes available and set `--shard-index` to the index of one specific node. When using one compute node set these parameters to 1 and 0 respectively. 
4. Combine the sharded datasets (when applicable) and normalize embeddings with `normalize-combine-embeddings.py`, using `--n-shards` with the same value as in step 3.
5. Embed the queries with the fine-tuned model using `embed-queries.py`.
6. Run evaluation using `evaluation-longformer.py`. This will save rankings in `data/results/longformer-....tsv`, for all 3 query sets.
7. Evaluate the rankings with `evaluate.py` using `-q (path to qrels)` and `-r (path to ranking)`