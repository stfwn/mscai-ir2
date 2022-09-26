#! /bin/bash

mkdir -p data/ms-marco/ && cd data/ms-marco

# Docs
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs-lookup.tsv.gz

# Train
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-queries.tsv.gz
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-qrels.tsv.gz
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-top100.gz

# Dev
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docdev-queries.tsv.gz
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docdev-qrels.tsv.gz
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docdev-top100.gz

gzip -dkv msmarco*

mv msmarco-doctrain-top100{,.tsv}
mv msmarco-docdev-top100{,.tsv}
