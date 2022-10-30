from argparse import ArgumentParser
import os
from pathlib import Path

import datasets
import torch
import torch.nn.functional as F
from torch import nn
from transformers import Trainer, TrainingArguments

import models


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PassageModelWrapper(models.PassageTransformer()).to(device)

    ds = datasets.load_from_disk(Path("./data/ms-marco/embedding-training-set"))
    print("Version:", args.version)
    output_dir = f"./models/passage-transformer-v{args.version}"
    # This commented out stuff loads the latest checkpoint
    # but the trainer's state (like the learning rate etc) will not be loaded
    #
    # latest_version_dir = sorted(
    #     Path(output_dir).glob("checkpoint-*"),
    #     key=lambda x: int(x.name.split("-")[-1]),
    # )[-1]
    # model.load_state_dict(torch.load(latest_version_dir / "pytorch_model.bin"))
    # print("Starting from checkpoint:", latest_version_dir)
    trainer = PassageModelTrainer(
        model=model,
        data_collator=collate_fn,
        train_dataset=ds,
        args=TrainingArguments(
            per_device_train_batch_size=64,
            output_dir=output_dir,
            report_to=["tensorboard", "wandb"],
            save_strategy="steps",
            save_steps=10000,
            fp16=True,
            num_train_epochs=100,
            dataloader_num_workers=os.cpu_count(),
            save_total_limit=10,
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

    dot_sim_matrix = query_embeddings @ doc_embeddings.transpose(0, 1)
    labels = torch.arange(len(query_embeddings), device=dot_sim_matrix.device)
    return F.cross_entropy(dot_sim_matrix, labels, reduction="mean")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-v",
        "--version",
        type=int,
        required=True,
        help="Simple versioning system. If you don't know what to pick: check in ./models and use one that's not there",
    )
    args = parser.parse_args()
    main(args)
