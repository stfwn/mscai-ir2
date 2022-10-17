from typing import List

import torch
from torch.nn import Module, TransformerEncoder, TransformerEncoderLayer


class PassageTransformer(Module):
    def __init__(self, d_model=768, dim_feedforward=1024, nhead=8, num_layers=1):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src, mask=None):
        """
        Args:
            x: (batch, seq, feature)"""
        return self.transformer(src, mask=mask).mean(dim=1)

    @staticmethod
    def _extract_passage_embeddings(doc: dict):
        return torch.tensor(
            [
                p["passage_embedding"]
                for p in sorted(doc["passages"], key=lambda p: p["passage_id"])
            ]
        )

    def encode_doc(self, doc: dict):
        """
        Args:
            doc: dict with key 'passages' with value List[dict], with each a 'passage embedding' key.
        """
        passage_embeddings = self._extract_passage_embeddings(doc)
        # Add a batch dimension.
        return self(passage_embeddings.unsqueeze(0))

    def encode_docs(self, docs: List[dict]):
        all_passage_embeddings = torch.nested_tensor(
            [self._extract_passage_embeddings(doc) for doc in docs]
        ).to_padded_tensor(padding=0.0)
        attn_mask = all_passage_embeddings == 0.0
        return self(all_passage_embeddings, attn_mask)
