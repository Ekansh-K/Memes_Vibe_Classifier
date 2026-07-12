"""Hate-CLIPper fusion for MMHS + optional MemeCLIP adapters.

Reference: Kumar & Nandakumar, Hate-CLIPper (EMNLP 2022 NLP4PI)
           Shah et al., MemeCLIP (EMNLP 2024) — residual adapters
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)


class Adapter(nn.Module):
    """MemeCLIP-style bottleneck adapter."""

    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        hidden = max(dim // reduction, 64)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HateCLIPperMMHS(nn.Module):
    """Frozen CLIP + projection maps + align/concat/cross fusion → binary logit."""

    def __init__(
        self,
        clip_model: str = "openai/clip-vit-large-patch14",
        map_dim: int = 512,
        fusion: str = "align",
        num_mapping_layers: int = 1,
        num_pre_output_layers: int = 1,
        drop_map: float = 0.1,
        drop_fusion: float = 0.4,
        drop_pre: float = 0.2,
        freeze_encoders: bool = True,
        use_adapters: bool = True,
        adapter_ratio: float = 0.2,
    ):
        super().__init__()
        self.fusion = fusion
        self.map_dim = map_dim
        self.use_adapters = use_adapters
        self.adapter_ratio = adapter_ratio

        self.clip = CLIPModel.from_pretrained(clip_model)
        self.processor = CLIPProcessor.from_pretrained(clip_model)
        vis_dim = self.clip.config.vision_config.hidden_size
        txt_dim = self.clip.config.text_config.hidden_size

        self._freeze_encoders = freeze_encoders
        if freeze_encoders:
            for p in self.clip.parameters():
                p.requires_grad = False
            self.clip.eval()

        def make_map(in_dim: int) -> nn.Sequential:
            layers: list[nn.Module] = [
                nn.Linear(in_dim, map_dim),
                nn.Dropout(drop_map),
            ]
            for _ in range(1, num_mapping_layers):
                layers.extend([nn.ReLU(inplace=True), nn.Linear(map_dim, map_dim), nn.Dropout(drop_map)])
            return nn.Sequential(*layers)

        self.image_map = make_map(vis_dim)
        self.text_map = make_map(txt_dim)

        if use_adapters:
            self.img_adapter = Adapter(map_dim)
            self.txt_adapter = Adapter(map_dim)

        if fusion == "align":
            pre_in = map_dim
        elif fusion == "concat":
            pre_in = map_dim * 2
        elif fusion == "cross":
            pre_in = map_dim * map_dim
        else:
            raise ValueError(f"Unknown fusion={fusion}")

        pre_layers: list[nn.Module] = [nn.Dropout(drop_fusion)]
        out_dim = pre_in
        if num_pre_output_layers >= 1:
            pre_layers.extend(
                [nn.Linear(pre_in, map_dim), nn.ReLU(inplace=True), nn.Dropout(drop_pre)]
            )
            out_dim = map_dim
            for _ in range(1, num_pre_output_layers):
                pre_layers.extend(
                    [nn.Linear(map_dim, map_dim), nn.ReLU(inplace=True), nn.Dropout(drop_pre)]
                )
        self.pre_output = nn.Sequential(*pre_layers)
        self.output = nn.Linear(out_dim, 1)

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"[HateCLIPper] fusion={fusion} map_dim={map_dim} adapters={use_adapters} "
            f"trainable={n_train:,}"
        )

    def unfreeze_last_n_blocks(self, n: int) -> None:
        """Unfreeze last N transformer blocks of CLIP vision + text encoders.

        Useful partial FT: keep most of CLIP frozen, tune top layers only.
        """
        if n <= 0:
            return
        # Vision transformer layers
        vision_layers = getattr(self.clip.vision_model.encoder, "layers", None)
        if vision_layers is not None:
            for layer in list(vision_layers)[-n:]:
                for p in layer.parameters():
                    p.requires_grad = True
        # Text transformer layers
        text_layers = getattr(self.clip.text_model.encoder, "layers", None)
        if text_layers is not None:
            for layer in list(text_layers)[-n:]:
                for p in layer.parameters():
                    p.requires_grad = True
        # Always unfreeze final layer norms / projections if present
        for name, mod in (
            ("vision_model.post_layernorm", getattr(self.clip.vision_model, "post_layernorm", None)),
            ("text_model.final_layer_norm", getattr(self.clip.text_model, "final_layer_norm", None)),
            ("visual_projection", getattr(self.clip, "visual_projection", None)),
            ("text_projection", getattr(self.clip, "text_projection", None)),
        ):
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = True
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"[HateCLIPper] unfreeze_last_n={n} → trainable={n_train:,}")

    def encode_batch(
        self,
        images: list,  # PIL
        texts: list[str],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # CLIP image + text
        proc = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        proc = {k: v.to(device) for k, v in proc.items()}
        with torch.set_grad_enabled(any(p.requires_grad for p in self.clip.parameters())):
            # Use vision/text towers directly for pooler_output
            vis = self.clip.vision_model(pixel_values=proc["pixel_values"])
            img_feat = vis.pooler_output  # (B, vis_dim)
            txt = self.clip.text_model(
                input_ids=proc["input_ids"],
                attention_mask=proc["attention_mask"],
            )
            txt_feat = txt.pooler_output
        return img_feat, txt_feat

    def forward(self, images: list, texts: list[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        img_feat, txt_feat = self.encode_batch(images, texts, device)

        img_p = self.image_map(img_feat.float())
        txt_p = self.text_map(txt_feat.float())

        if self.use_adapters:
            a = self.adapter_ratio
            img_p = a * self.img_adapter(img_p) + (1 - a) * img_p
            txt_p = a * self.txt_adapter(txt_p) + (1 - a) * txt_p

        img_p = F.normalize(img_p, p=2, dim=1)
        txt_p = F.normalize(txt_p, p=2, dim=1)

        if self.fusion == "align":
            fused = img_p * txt_p
        elif self.fusion == "concat":
            fused = torch.cat([img_p, txt_p], dim=1)
        else:  # cross — outer product flattened
            # (B, d, 1) x (B, 1, d) → (B, d, d) → (B, d*d)
            fused = torch.bmm(img_p.unsqueeze(2), txt_p.unsqueeze(1)).flatten(1)

        h = self.pre_output(fused)
        return self.output(h).squeeze(-1)
