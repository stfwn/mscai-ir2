from typing import List

import torch
from torch import Tensor
from torch.nn import Module, TransformerEncoder, TransformerEncoderLayer


class PassageTransformer(Module):
    def __init__(
        self,
        d_model=768,
        dim_feedforward=1024,
        nhead=8,
        num_layers=1,
        pooling_method: str = "mean",
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.nhead = nhead
        self.num_layers = 1
        self.pooling_method = pooling_method
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        doc_passage_embeddings: List[Tensor],
    ):
        """
        Args:
            doc_passage_embeddings: list (batch length) of tensors of shape
            (seq, feature)

        Returns:
            ...

        """
        batch = torch.nested_tensor(doc_passage_embeddings).to_padded_tensor(
            padding=0.0
        )
        padding_mask = (batch == 0.0).sum(-1) == self.d_model
        output = self.transformer(batch, mask=None, src_key_padding_mask=padding_mask)
        if self.pooling_method == "mean":
            return torch.vstack(
                [o[~m].mean(dim=0) for o, m in zip(output, padding_mask)]
            )
        else:
            raise NotImplementedError("Unknown pooling method:", self.pooling_method)

    def encode_doc(self, doc: dict):
        """
        Args:
            doc: dict with key 'passages' with value List[dict], with each a 'passage embedding' key.
        """
        passage_embeddings = torch.tensor(
            [
                p["passage_embedding"]
                for p in sorted(doc["passages"], key=lambda p: p["passage_id"])
            ]
        )
        return self([passage_embeddings])

    def encode_docs(self, docs: dict):
        """
        Args:
            docs: dict with key 'passages' with value List[List[dict]], where
            the outer list lists docs, and the inner list lists passages within
            those docs. Passages must each have a 'passage embedding' key.
        """
        all_passage_embeddings = [
            torch.tensor([p["passage_embedding"] for p in passages])
            for passages in docs["passages"]
        ]
        return self(all_passage_embeddings)
