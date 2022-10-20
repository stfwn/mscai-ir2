from pathlib import Path

import datasets
import torch
import torch.nn.functional as F
from torch import nn
from transformers import Trainer, TrainingArguments

import models


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.PassageTransformer().to(device)

    ds = datasets.load_from_disk(Path("./data/ms-marco/embedding-training-set"))
    trainer = PassageModelTrainer(
        model=PassageModelWrapper(model),
        data_collator=collate_fn,
        train_dataset=ds,
        args=TrainingArguments(
            per_device_train_batch_size=64,
            output_dir="./models/passage-transformer-v1",
            report_to=["tensorboard", "wandb"],
            save_strategy="steps",
            save_steps=500,
            fp16=True,
        ),
    )
    trainer.train()


class PassageModelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        doc_embeddings = model(
            query_embedding=None,  # This arg is just here to make HuggingFace happy.
            doc_passage_embeddings=inputs["doc_passage_embeddings"],
        )
        loss = loss_fn(inputs["query_embeddings"], doc_embeddings)
        return (loss, doc_embeddings) if return_outputs else loss


class PassageModelWrapper(nn.Module):
    """This would be handled in the collate function but the HuggingFace
    Trainer class removes columns from the training set if the column names
    don't match with any model args."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, query_embedding, doc_passage_embeddings):
        return self.model(doc_passage_embeddings)


def collate_fn(batch):
    return {
        "query_embeddings": [
            torch.tensor(sample["query_embedding"]) for sample in batch
        ],
        "doc_passage_embeddings": [
            torch.tensor(sample["doc_passage_embeddings"]) for sample in batch
        ],
    }


def loss_fn(query_embeddings, doc_embeddings):
    if not isinstance(query_embeddings, torch.Tensor):
        query_embeddings = torch.vstack(query_embeddings)
    if not isinstance(doc_embeddings, torch.Tensor):
        doc_embeddings = torch.vstack(doc_embeddings)

    doc_embeddings_norm = F.normalize(doc_embeddings)
    query_embeddings_norm = F.normalize(query_embeddings)
    cos_sim_matrix = query_embeddings_norm @ doc_embeddings_norm.transpose(0, 1)
    labels = torch.arange(len(query_embeddings))
    return F.cross_entropy(cos_sim_matrix, labels, reduction="mean")


if __name__ == "__main__":
    main()
