"""TCAM — Text-guided Cross-Attention Multimodal model (P2).

Architecture:
    image → CLIP ViT-L/14 (frozen, patch tokens via hook) → V ∈ R^(B, 257, 768)
    text  → TweetEval RoBERTa (frozen)                    → T ∈ R^(B, seq, 768)
                                    ↓
          proj_t: Linear(768→768, identity-init) → T_proj
          CrossAttention(Q=V, K=T_proj, V=T_proj, 8 heads) → V' (residual + LayerNorm)
                                    ↓
          mean_pool(V') || mean_pool(T_proj) → concat ∈ R^(B, 1536)
                                    ↓
          Head: Linear(1536→512) → GELU → Dropout(0.3) → Linear(512→num_classes)

Frozen: clip_visual, tweet_encoder (all requires_grad=False)
Trainable: proj_t, cross_attn, head (~3-5M params)
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CrossAttention(nn.Module):
    """Multi-head cross-attention with residual connection and LayerNorm.

    Q = V (image patch tokens)
    K = T_proj (projected text tokens)
    V = T_proj

    Output: LayerNorm(V + Attn(V, T_proj, T_proj))
    """

    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        V: torch.Tensor,
        T: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            V: image patch tokens (B, n_patches, d_model)
            T: projected text tokens (B, seq_len, d_model)
            key_padding_mask: (B, seq_len) — True for padded positions

        Returns:
            V': attended image tokens (B, n_patches, d_model)
        """
        V_prime, _ = self.attn(
            query=V, key=T, value=T,
            key_padding_mask=key_padding_mask,
        )
        return self.norm(V + V_prime)  # residual + norm


class TCAM(nn.Module):
    """Full TCAM model: frozen CLIP + frozen TweetEval → cross-attention → head.

    Supports all 4 P2 variations via num_classes:
        P2-A:       num_classes=1 (binary)
        P2-B:       num_classes=6 (direct 6-class)
        P2-C/D S1:  num_classes=1 (binary)
        P2-C/D S2:  num_classes=5 (hate categories only)
    """

    def __init__(
        self,
        num_classes: int,
        clip_model: str = "ViT-L/14",
        tweet_model: str = "cardiffnlp/twitter-roberta-base",
        d_model: int = 768,
        n_heads: int = 8,
        head_hidden: int = 512,
        dropout: float = 0.3,
        max_text_len: int = 128,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model
        self.max_text_len = max_text_len

        # ── Frozen CLIP visual encoder ──────────────────────────────────────
        import clip as clip_lib
        self._clip_model, self.clip_preprocess = clip_lib.load(
            clip_model, device="cpu"
        )
        for p in self._clip_model.parameters():
            p.requires_grad = False
        self._clip_model.eval()
        logger.info(f"[TCAM] CLIP {clip_model} loaded and frozen.")

        # ── Frozen TweetEval text encoder ───────────────────────────────────
        from transformers import AutoModel, AutoTokenizer
        self.tweet_tokenizer = AutoTokenizer.from_pretrained(tweet_model)
        self.tweet_encoder = AutoModel.from_pretrained(tweet_model)
        for p in self.tweet_encoder.parameters():
            p.requires_grad = False
        self.tweet_encoder.eval()
        logger.info(f"[TCAM] TweetEval {tweet_model} loaded and frozen.")

        # ── Learnable projection (identity-init) ───────────────────────────
        self.proj_t = nn.Linear(d_model, d_model)
        nn.init.eye_(self.proj_t.weight)
        nn.init.zeros_(self.proj_t.bias)

        # ── Learnable cross-attention ───────────────────────────────────────
        self.cross_attn = CrossAttention(d_model=d_model, n_heads=n_heads)

        # ── Learnable classification head ───────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes),
        )

    def _extract_patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Extract pre-pooled patch tokens from CLIP ViT-L/14.

        Registers a forward hook on the last transformer resblock to capture
        the sequence output BEFORE ln_post and projection pooling.

        CLIP ViT stores sequences as (seq_len, batch, d_model) internally,
        so we permute to (batch, seq_len, d_model) for batch-first ops.

        Args:
            images: (B, 3, 224, 224) preprocessed images (on device).

        Returns:
            (B, 257, 768) where 257 = 1 CLS + 256 patches (14×14 grid).
        """
        patch_tokens = {}

        def hook_fn(module, input, output):
            patch_tokens["v"] = output

        handle = self._clip_model.visual.transformer.resblocks[-1].register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                _ = self._clip_model.encode_image(images)
        finally:
            handle.remove()

        # CLIP internal: (seq=257, B, 768) → batch-first: (B, 257, 768)
        V = patch_tokens["v"].permute(1, 0, 2).float()
        return V

    def _encode_text(
        self, texts: list[str], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize and encode text through frozen TweetEval.

        Args:
            texts: list of B text strings.
            device: target device for the output tensors.

        Returns:
            T: (B, seq_len, 768) last hidden states
            pad_mask: (B, seq_len) True where padding
        """
        enc = self.tweet_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_text_len,
        )
        # Move tokenizer outputs to the correct device
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = self.tweet_encoder(**enc)

        T = out.last_hidden_state.float()       # (B, seq_len, 768)
        pad_mask = (enc["attention_mask"] == 0)  # (B, seq_len) True = pad
        return T, pad_mask

    def _preprocess_images(
        self, pil_images: list, device: torch.device
    ) -> torch.Tensor:
        """Apply CLIP preprocessing to a list of PIL images.

        Args:
            pil_images: list of B PIL.Image instances.
            device: target device.

        Returns:
            (B, 3, 224, 224) preprocessed image tensor on device.
        """
        tensors = [self.clip_preprocess(img) for img in pil_images]
        return torch.stack(tensors).to(device)

    def forward(
        self,
        images,
        texts: list[str],
    ) -> torch.Tensor:
        """Forward pass through TCAM.

        Args:
            images: list of PIL.Image OR pre-processed tensor (B, 3, 224, 224).
                    Pre-processed tensor is used when called under DataParallel
                    (DataParallel can only scatter tensors, not lists).
            texts:  list of B text strings.

        Returns:
            logits: (B, num_classes) — raw logits (no sigmoid/softmax).
        """
        # Determine device from trainable parameters
        device = self.proj_t.weight.device

        # ── Visual branch: CLIP patch tokens ──────────────────────────────
        if isinstance(images, torch.Tensor):
            # Already preprocessed tensor from trainer (DataParallel path)
            images_tensor = images.to(device)
        else:
            # List of PIL images — preprocess on the fly (single-GPU path)
            images_tensor = self._preprocess_images(images, device)
        V = self._extract_patch_tokens(images_tensor)  # (B, 257, 768)

        # ── Text branch: TweetEval encoding ───────────────────────────────
        T, pad_mask = self._encode_text(texts, device)  # (B, seq, 768)

        # ── Learnable fusion ──────────────────────────────────────────────
        T_proj = self.proj_t(T)                          # (B, seq, 768)
        V_prime = self.cross_attn(V, T_proj, key_padding_mask=pad_mask)

        # Mean-pool both branches, concatenate
        v_pool = V_prime.mean(dim=1)   # (B, 768)
        t_pool = T_proj.mean(dim=1)    # (B, 768)
        fused = torch.cat([v_pool, t_pool], dim=-1)  # (B, 1536)

        return self.head(fused)  # (B, num_classes)

    def reinit_for_stage2(self, new_num_classes: int) -> None:
        """Reinitialise cross_attn and head for Stage 2.

        KEEPS proj_t weights from Stage 1 — it learned useful image-text
        alignment that transfers well since Stage 2 uses the same text
        domain (just the hateful subset).

        Args:
            new_num_classes: number of output classes for Stage 2 (typically 5).
        """
        # Reset cross_attn parameters
        for m in self.cross_attn.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

        # Rebuild head with new output size
        head_hidden = self.head[0].out_features  # preserve hidden dim
        dropout = self.head[2].p  # preserve dropout rate
        d_input = self.head[0].in_features  # 1536

        self.head = nn.Sequential(
            nn.Linear(d_input, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, new_num_classes),
        )
        self.num_classes = new_num_classes

        # Count trainable params
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"[TCAM] reinit_for_stage2: num_classes={new_num_classes}, "
            f"proj_t weights KEPT. "
            f"Trainable: {n_trainable:,} / {n_total:,} total params."
        )

    @classmethod
    def from_config(cls, config, num_classes: int) -> "TCAM":
        """Construct TCAM from a P2Config instance."""
        return cls(
            num_classes=num_classes,
            clip_model=config.clip_model,
            tweet_model=config.tweet_model,
            d_model=config.d_model,
            n_heads=config.n_heads,
            head_hidden=config.head_hidden,
            dropout=config.head_dropout,
            max_text_len=config.max_text_len,
        )
