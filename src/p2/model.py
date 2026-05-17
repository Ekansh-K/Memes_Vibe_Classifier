"""TCAM — Text-guided Cross-Attention Multimodal model (P2).

Architecture:
    image → CLIP ViT-L/14 (frozen, patch tokens via hook) → V ∈ R^(B, 257, 1024)
    text  → TweetEval RoBERTa (frozen)                    → T ∈ R^(B, seq, 768)
                                    ↓
          proj_t: Linear(768→1024)  → T_proj  [projects text dim to visual dim]
          CrossAttention(Q=V[1024], K=T_proj[1024], V=T_proj[1024], 8 heads) → V'
                                    ↓
          mean_pool(V') || mean_pool(T_proj) → concat ∈ R^(B, 2048)
                                    ↓
          Head: Linear(2048→512) → GELU → Dropout(0.3) → Linear(512→num_classes)

Dimensions:
    d_v = 1024  ← CLIP ViT-L/14 visual embedding dim
    d_t = 768   ← TweetEval RoBERTa hidden dim
    d_fused = d_v * 2 = 2048  ← after mean-pooling + concat

Frozen: clip_visual, tweet_encoder (all requires_grad=False)
Trainable: proj_t, cross_attn, head (~5-7M params)
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
        d_v: int = 1024,          # CLIP ViT-L/14 visual patch embedding dim
        d_t: int = 768,           # TweetEval RoBERTa hidden dim
        n_heads: int = 8,
        head_hidden: int = 512,
        dropout: float = 0.3,
        max_text_len: int = 128,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.d_v = d_v
        self.d_t = d_t
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

        # ── Learnable projection: text dim (d_t=768) → visual dim (d_v=1024) ─
        # Partial-identity init: top-left d_t×d_t block = Identity, rest = 0.
        # At t=0, text tokens pass through unchanged in first d_t dims, giving
        # a stable starting point equivalent to identity for square proj_t.
        self.proj_t = nn.Linear(d_t, d_v, bias=True)
        nn.init.zeros_(self.proj_t.weight)
        nn.init.zeros_(self.proj_t.bias)
        with torch.no_grad():
            self.proj_t.weight[:d_t, :d_t].copy_(torch.eye(d_t))

        # ── Learnable cross-attention (operates in d_v=1024 space) ──────────
        self.cross_attn = CrossAttention(d_model=d_v, n_heads=n_heads)

        # ── Learnable classification head ────────────────────────────────────
        # Input: concat(v_pool[d_v], t_pool[d_v]) = d_v * 2 = 2048
        self.head = nn.Sequential(
            nn.Linear(d_v * 2, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes),
        )

    def _extract_patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Extract pre-pooled patch tokens from CLIP ViT-L/14.

        Registers a forward hook on the last transformer resblock to capture
        the sequence output BEFORE ln_post and projection pooling.

        CLIP ViT-L/14 stores sequences as (seq_len, batch, d_v) internally,
        so we permute to (batch, seq_len, d_v) for batch-first ops.

        Args:
            images: (B, 3, 224, 224) preprocessed images (on device).

        Returns:
            (B, 257, 1024) where 257 = 1 CLS + 256 patches (14×14 grid),
            and 1024 = d_v (CLIP ViT-L/14 internal embedding dim).
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

        # CLIP ViT-L/14 internal: (seq=257, B, d_v=1024) → batch-first: (B, 257, 1024)
        V = patch_tokens["v"].permute(1, 0, 2).float()
        return V  # (B, 257, 1024)

    def _encode_text(
        self, texts: list[str], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize and encode text through TweetEval (frozen or partially unfrozen).

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
        enc = {k: v.to(device) for k, v in enc.items()}

        # Use no_grad only when tweet_encoder is fully frozen.
        # When partially unfrozen (Phase 2), gradients must flow through
        # the unfrozen layers — no_grad would block them entirely.
        _tweet_has_grad = any(p.requires_grad for p in self.tweet_encoder.parameters())
        if _tweet_has_grad:
            out = self.tweet_encoder(**enc)
        else:
            with torch.no_grad():
                out = self.tweet_encoder(**enc)

        T = out.last_hidden_state.float()       # (B, seq_len, 768)
        pad_mask = (enc["attention_mask"] == 0)  # (B, seq_len) True = pad
        return T, pad_mask  # T: (B, seq, d_t=768)

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
        # Determine device from trainable parameters (stable, single-GPU).
        device = self.proj_t.weight.device

        # ── Visual branch: CLIP patch tokens ──────────────────────────────
        if isinstance(images, torch.Tensor):
            images_tensor = images.to(device)
        else:
            images_tensor = self._preprocess_images(images, device)
        V = self._extract_patch_tokens(images_tensor)  # (B, 257, d_v=1024)

        # ── Text branch: TweetEval encoding ──────────────────────────────
        T, pad_mask = self._encode_text(texts, device)  # (B, seq, d_t=768)

        # ── Learnable fusion ──────────────────────────────────────────────
        T_proj  = self.proj_t(T)                              # (B, seq, d_v=1024)
        V_prime = self.cross_attn(V, T_proj,
                                  key_padding_mask=pad_mask)  # (B, 257, d_v)

        # Mean-pool both branches, concatenate
        v_pool = V_prime.mean(dim=1)                          # (B, d_v=1024)
        t_pool = T_proj.mean(dim=1)                           # (B, d_v=1024)
        fused  = torch.cat([v_pool, t_pool], dim=-1)          # (B, 2048)

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
        d_input = self.head[0].in_features  # d_v * 2 = 2048

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
        model = cls(
            num_classes=num_classes,
            clip_model=config.clip_model,
            tweet_model=config.tweet_model,
            d_v=config.d_v,
            d_t=config.d_t,
            n_heads=config.n_heads,
            head_hidden=config.head_hidden,
            dropout=config.head_dropout,
            max_text_len=config.max_text_len,
        )
        # Optionally unfreeze last N TweetEval layers (Phase 2)
        n = getattr(config, "unfreeze_tweet_last_n", 0)
        if n > 0:
            model.unfreeze_tweet_layers(n)
        return model

    def unfreeze_tweet_layers(self, n: int) -> None:
        """Unfreeze the last n transformer layers of tweet_encoder.

        All other TweetEval layers remain frozen. Use with a much lower LR
        (tweet_encoder_lr ≈ 1e-5) than the head/cross_attn LR to avoid
        catastrophic forgetting of general Twitter language understanding.

        IMPORTANT — pooler is intentionally NOT unfrozen:
            TCAM reads `last_hidden_state` (token-level sequence output, shape
            [B, seq, 768]) for cross-attention. The pooler is a sentence-level
            Linear+tanh on the [CLS] token used by classification heads —
            it is architecturally irrelevant here. Unfreezing it would add
            ~590K dead parameters consuming optimizer capacity for zero benefit.
        """
        if n <= 0:
            return
        layers = self.tweet_encoder.encoder.layer
        total = len(layers)
        n = min(n, total)  # clamp to actual layer count
        for layer in layers[-n:]:
            for p in layer.parameters():
                p.requires_grad = True
        # pooler intentionally NOT unfrozen — see docstring
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"[TCAM] Unfrozen last {n}/{total} TweetEval layers (pooler frozen). "
            f"Trainable params: {n_trainable:,}"
        )

    def unfreeze_clip_last_layer(self) -> None:
        """Unfreeze only the final transformer block of CLIP's visual encoder.

        CLIP ViT-L/14 has 24 ResidualAttentionBlock layers.
        Use this as a LAST RESORT (Phase 3b) if TweetEval unfreezing alone
        cannot push Stage 1 F1 past 0.69.

        IMPORTANT — use a very low LR (clip_encoder_lr ≈ 1e-6):
            CLIP features encode visual semantics from 400M image-text contrastive
            pairs and are far more brittle than TweetEval text features. Using
            LR=1e-5 will corrupt the visual representation.
        """
        final_block = self._clip_model.visual.transformer.resblocks[-1]
        for p in final_block.parameters():
            p.requires_grad = True
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"[TCAM] Unfrozen CLIP final visual block. "
            f"Trainable params: {n_trainable:,}"
        )
