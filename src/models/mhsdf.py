"""Lightweight MHSDF model: CNN (ResNet18) + BiLSTM text branch + MLP head.

Designed as the P7 baseline (CNN+BiLSTM) described in the project survey.
The visual branch uses a pretrained ResNet18 (conv backbone) and the text
branch uses a small embedding + BiLSTM encoder. The fused vector is passed
through a small MLP classifier head supporting multiclass or multilabel.
"""

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class MHSDF(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 300,
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
        bidirectional: bool = True,
        freeze_cnn: bool = True,
        num_classes: int = 6,
        multilabel: bool = False,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # Visual backbone (ResNet18 without classifier)
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        # Remove the final fully-connected layer
        modules = list(resnet.children())[:-1]
        self.cnn = nn.Sequential(*modules)  # output: (B, 512, 1, 1)
        self.visual_dim = 512

        if freeze_cnn:
            for p in self.cnn.parameters():
                p.requires_grad = False

        # Text branch
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        text_dim = lstm_hidden * (2 if bidirectional else 1)

        # Project text to visual dim for balanced fusion
        self.text_proj = nn.Linear(text_dim, self.visual_dim)

        # Classification head
        head_in = self.visual_dim + self.visual_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

        self.multilabel = multilabel

    def _encode_image(self, images: torch.Tensor) -> torch.Tensor:
        # images: (B, C, H, W)
        v = self.cnn(images)  # (B, 512, 1, 1)
        v = v.view(v.size(0), -1)  # (B, 512)
        return v

    def _encode_text(self, text_ids: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        # text_ids: (B, L)
        emb = self.embedding(text_ids)  # (B, L, emb_dim)
        if lengths is not None:
            # pack padded sequence for efficiency
            packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, (h_n, c_n) = self.lstm(packed)
            # h_n: (num_layers * num_directions, B, hidden)
            # Collect last layer hidden states
            if self.lstm.bidirectional:
                # concatenate forward and backward final states
                h_fwd = h_n[-2]
                h_bwd = h_n[-1]
                h = torch.cat([h_fwd, h_bwd], dim=-1)
            else:
                h = h_n[-1]
        else:
            out, (h_n, c_n) = self.lstm(emb)
            if self.lstm.bidirectional:
                h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            else:
                h = h_n[-1]

        # Project to visual dim
        t = self.text_proj(h)
        return t

    def forward(self, images: torch.Tensor, text_ids: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        v = self._encode_image(images)
        t = self._encode_text(text_ids, lengths=lengths)
        fused = torch.cat([v, t], dim=-1)
        logits = self.head(fused)
        if self.multilabel:
            return logits  # BCEWithLogitsLoss expects raw logits
        else:
            return logits  # CrossEntropyLoss expects raw logits
