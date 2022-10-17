from typing import List

import torch
from torch.nn import Module, TransformerEncoder, TransformerEncoderLayer


class PassageTransformer(Module):
    def __init__(self, d_model=768, dim_feedforward=1024, nhead=8, num_layers=1):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.nhead = nhead
        self.num_layers = 1
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        """
        Args:
            x: (batch, seq, feature)"""
        return self.transformer(
            src, mask=mask, src_key_padding_mask=src_key_padding_mask
        ).mean(dim=1)

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
        # Add a batch dimension.
        return self(passage_embeddings.unsqueeze(0))

    def encode_docs(self, docs: dict):
        """
        Args:
            docs: dict with key 'passages' with value List[List[dict]], where
            the outer list lists docs, and the inner list lists passages within
            those docs. Passages must each have a 'passage embedding' key.
        """
        all_passage_embeddings = torch.nested_tensor(
            [
                torch.tensor([p["passage_embedding"] for p in passages])
                for passages in docs["passages"]
            ]
        ).to_padded_tensor(padding=0.0)
        padding_mask = (all_passage_embeddings == 0.0).sum(-1) == self.d_model
        return self(all_passage_embeddings, src_key_padding_mask=padding_mask)
