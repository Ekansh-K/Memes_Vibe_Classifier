"""RGCL model: CLIP encoders with projection and classification head."""

from __future__ import annotations

import torch
import torch.nn as nn
import clip


class RGCLModel(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int = 768, clip_model: str = "ViT-L/14"):
        super().__init__()
        self.clip_model_name = clip_model
        self.clip, _ = clip.load(clip_model)
        for p in self.clip.parameters():
            p.requires_grad = False

        clip_dim = self.clip.visual.output_dim
        self.proj = nn.Sequential(
            nn.Linear(clip_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def encode(self, images: torch.Tensor, texts: list[str]) -> torch.Tensor:
        with torch.no_grad():
            v = self.clip.encode_image(images).float()
            t = self.clip.encode_text(clip.tokenize(texts, truncate=True).to(images.device)).float()
        v = v / v.norm(dim=-1, keepdim=True)
        t = t / t.norm(dim=-1, keepdim=True)
        return self.proj(torch.cat([v, t], dim=-1))

    def forward(self, images: torch.Tensor, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.encode(images, texts)
        return self.head(emb), emb
