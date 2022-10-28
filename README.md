https://www.overleaf.com/2354611475wtkmfxsqzhgt

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
5. Evaluate the resulting doc embeddings using `evaluation.py` with the right
   arguments: TODO: THIJS, doc embedding dataset dir: `data/ms-marco/doc-embeddings/passage-transformer-v1`).
