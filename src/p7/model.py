"""MHSDF — CNN + BiLSTM multimodal hate speech detection model (P7).

Architecture (survey Section 10.3):
    image  → CNNVisualEncoder  → v ∈ R^512
    text   → BiLSTMTextEncoder → t ∈ R^512
             concat([v, t])    ∈ R^1024
             MLP head          → logits ∈ R^num_classes

Two-stage usage:
    Stage 1: MHSDF(vocab_size, num_classes=2)         binary
    Stage 2: MHSDF(vocab_size, num_classes=5)         category
             — backbone weights transferred from Stage 1
             — only the head is re-initialised
"""

import torch
import torch.nn as nn


class CNNVisualEncoder(nn.Module):
    """3-layer CNN that maps an image to a fixed-size visual feature vector.

    Architecture:
        Conv2d(3→64)  + ReLU + MaxPool2d(2)
        Conv2d(64→128) + ReLU + MaxPool2d(2)
        Conv2d(128→256) + ReLU + AdaptiveAvgPool2d(4)
        Flatten → Linear(256*4*4, out_dim) → ReLU → Dropout

    Input:  (B, 3, H, W)
    Output: (B, out_dim)
    """

    def __init__(self, out_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),

            nn.Flatten(),                               # → (B, 256*4*4 = 4096)
            nn.Linear(256 * 4 * 4, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor (B, 3, H, W).
        Returns:
            Visual feature (B, out_dim).
        """
        return self.net(x)


class BiLSTMTextEncoder(nn.Module):
    """Bidirectional LSTM that maps token IDs to a fixed-size text feature.

    Uses a learned embedding layer (initialised randomly) fed by BERT
    tokenizer IDs. The BiLSTM last hidden states (forward + backward)
    are concatenated and projected to out_dim.

    Input:  (B, seq_len) — LongTensor of token IDs
    Output: (B, out_dim)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        out_dim: int = 512,
        pad_idx: int = 0,
        pretrained_embeddings: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if pretrained_embeddings is not None:
            # Initialise from GloVe (or any pretrained matrix); keep trainable
            self.embed = nn.Embedding.from_pretrained(
                pretrained_embeddings, freeze=False, padding_idx=pad_idx
            )
        else:
            self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # hidden * 2 because bidirectional
        self.proj = nn.Linear(hidden * 2, out_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (B, seq_len) LongTensor.
        Returns:
            Text feature (B, out_dim).
        """
        x = self.embed(token_ids)          # (B, seq_len, embed_dim)
        _, (h, _) = self.lstm(x)           # h: (num_layers*2, B, hidden)
        # Last layer: h[-2] = last forward, h[-1] = last backward
        h_cat = torch.cat([h[-2], h[-1]], dim=-1)  # (B, hidden*2)
        return self.proj(h_cat)            # (B, out_dim)


class MHSDF(nn.Module):
    """Full MHSDF model: CNN visual + BiLSTM text → concat → MLP head.

    Supports all 4 P7 variations via num_classes:
        P7-A: num_classes=2  (binary)
        P7-B: num_classes=6  (direct 6-class)
        P7-C/D Stage 1: num_classes=2
        P7-C/D Stage 2: num_classes=5 (hate categories only)

    For P7-D (multilabel): apply sigmoid to logits at inference time.
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        cnn_out_dim: int = 512,
        embed_dim: int = 128,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
        head_hidden: int = 256,
        head_dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.visual = CNNVisualEncoder(out_dim=cnn_out_dim, dropout=head_dropout)
        self.textual = BiLSTMTextEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden=lstm_hidden,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
            out_dim=cnn_out_dim,
            pad_idx=pad_idx,
            pretrained_embeddings=pretrained_embeddings,
        )
        fusion_dim = cnn_out_dim * 2   # 1024 by default
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, num_classes),
        )
        self.num_classes = num_classes

    def forward(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            images:    (B, 3, H, W)
            token_ids: (B, seq_len)

        Returns:
            logits: (B, num_classes)  — raw (no sigmoid/softmax applied).
        """
        v = self.visual(images)        # (B, 512)
        t = self.textual(token_ids)    # (B, 512)
        fused = torch.cat([v, t], dim=-1)  # (B, 1024)
        return self.head(fused)        # (B, num_classes)

    def transfer_backbone(self, source: "MHSDF") -> None:
        """Copy CNN and BiLSTM weights from another MHSDF instance.

        Used for two-stage training: Stage 2 model inherits Stage 1
        backbone weights; only the head is re-trained from scratch.

        Args:
            source: Stage 1 MHSDF model whose backbone weights to copy.
        """
        self.visual.load_state_dict(source.visual.state_dict())
        self.textual.load_state_dict(source.textual.state_dict())

    @classmethod
    def from_config(
        cls,
        config,
        vocab_size: int,
        num_classes: int,
        pretrained_embeddings: Optional[torch.Tensor] = None,
    ) -> "MHSDF":
        """Construct MHSDF from a P7Config instance.

        Args:
            config:                 P7Config.
            vocab_size:             Tokenizer vocabulary size.
            num_classes:            Number of output classes.
            pretrained_embeddings:  Optional GloVe weight matrix (vocab_size, embed_dim).
        """
        return cls(
            vocab_size=vocab_size,
            num_classes=num_classes,
            cnn_out_dim=config.cnn_out_dim,
            embed_dim=config.embed_dim,
            lstm_hidden=config.lstm_hidden,
            lstm_layers=config.lstm_layers,
            lstm_dropout=config.lstm_dropout,
            head_hidden=config.head_hidden,
            head_dropout=config.head_dropout,
            pretrained_embeddings=pretrained_embeddings,
        )
