"""Multimodal Temporal Fusion Transformer with dynamic cross-attention fusion.

Fusion modes (same backbone, so ablations are apples-to-apples):
  * ``none``            – tabular TFT baseline.
  * ``static_scalar``   – Picnic-style MTFT: per-modality embeddings compressed to k dims
                          (k=10 in the paper) and appended as scalar static covariates.
  * ``static_pooled``   – aligned SigLIP embeddings -> tokenizer -> mean-pooled static variable.
  * ``cross_attention`` – proposed: decoder time-step queries attend to m multimodal latent tokens.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from .features import MultimodalTokenizer
from .layers import (
    CategoricalEmbedding,
    GateAddNorm,
    GatedResidualNetwork,
    InterpretableMultiHeadAttention,
    MonotoneQuantileHead,
    RealEmbedding,
    VariableSelectionNetwork,
)

FUSION_MODES = ("none", "static_scalar", "static_pooled", "cross_attention")


@dataclass
class MTFTConfig:
    static_categorical_cardinalities: List[int]
    n_static_reals: int
    n_observed_reals: int
    n_known_reals: int
    known_categorical_cardinalities: List[int]
    encoder_length: int
    horizon: int
    d_model: int = 64
    n_heads: int = 4
    dropout: float = 0.1
    lstm_layers: int = 1
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9)
    fusion: str = "cross_attention"
    mm_embed_dim: int = 768
    mm_token_embed_dim: Optional[int] = None
    mm_scalar_dim: int = 0
    mm_latents: int = 8
    mm_static_context: bool = True
    mm_feature_dropout: float = 0.0
    mm_modality_dropout: float = 0.0
    align_weight: float = 0.0
    # MTFT+ v4: lifecycle attention over lifecycle-aligned sibling / analogue tokens, and an age positional axis
    lifecycle_attention: bool = False
    la_siblings: bool = True
    la_analogues: bool = True
    n_shops: int = 0
    la_emb_dim: int = 0
    age_embedding: bool = False
    la_gate: bool = False  # v4.1 (exploratory): gate lifecycle attention by the number of observed own-history steps

    def __post_init__(self) -> None:
        if self.fusion not in FUSION_MODES:
            raise ValueError(f"fusion must be one of {FUSION_MODES}")
        if self.fusion == "static_scalar" and self.mm_scalar_dim <= 0:
            raise ValueError("static_scalar fusion requires mm_scalar_dim > 0")
        self.quantiles = tuple(float(q) for q in self.quantiles)

    def to_dict(self) -> Dict:
        return asdict(self)


class MultimodalCrossAttention(nn.Module):
    """Multi-head cross-attention  softmax(Q K^T / sqrt(d_k)) V  with Q from time steps, K/V from product tokens."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads, self.d_k = n_heads, d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, tokens: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, d = x.shape
        m = tokens.shape[1]
        q = self.w_q(x).view(b, t, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(tokens).view(b, m, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(tokens).view(b, m, self.n_heads, self.d_k).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / self.d_k**0.5  # (B, heads, T, m)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = (self.dropout(attn) @ v).transpose(1, 2).reshape(b, t, d)
        return self.w_o(out), attn.mean(1)


class SetAttention(nn.Module):
    """One query per time step attends over a set of N tokens at that step, plus a learned null token (so steps with
    no available token are well defined). q (B, S, d), tokens (B, N, S, d), mask (B, N, S) bool -> (B, S, d)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.h, self.dk = n_heads, d_model // n_heads
        self.w_q, self.w_k, self.w_v, self.w_o = (nn.Linear(d_model, d_model) for _ in range(4))
        self.null = nn.Parameter(torch.randn(2, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, n, s, d = tokens.shape
        qh = self.w_q(q).view(b, s, self.h, self.dk)
        kh = self.w_k(tokens).view(b, n, s, self.h, self.dk)
        vh = self.w_v(tokens).view(b, n, s, self.h, self.dk)
        scores = torch.einsum("bshd,bnshd->bshn", qh, kh) / self.dk**0.5
        scores = scores.masked_fill(~mask.permute(0, 2, 1)[:, :, None, :], float("-inf"))
        nk = self.w_k(self.null[0]).view(self.h, self.dk)
        nv = self.w_v(self.null[1]).view(self.h, self.dk)
        s_null = torch.einsum("bshd,hd->bsh", qh, nk).unsqueeze(-1) / self.dk**0.5
        attn = torch.softmax(torch.cat([s_null, scores], -1), -1)
        a = self.dropout(attn)
        out = a[..., :1] * nv + torch.einsum("bshn,bnshd->bshd", a[..., 1:], vh)
        return self.w_o(out.reshape(b, s, d)), attn


class LifecycleAttention(nn.Module):
    """MTFT+ v4: per-step attention over lifecycle-aligned siblings (same product, other shops, same product age; keys
    carry learned shop embeddings and the launch lag) and lifecycle-aligned analogues (similar products' mean curves at
    the same age; keys and queries carry projected product embeddings, so which analogues to trust is learned from
    content). Returns (B, S, n_branches, d) extra per-step variables for the encoder / decoder variable selection."""

    def __init__(self, cfg: "MTFTConfig", n_ages: int) -> None:
        super().__init__()
        d = cfg.d_model
        self.siblings, self.analogues = cfg.la_siblings, cfg.la_analogues
        self.age_q = nn.Embedding(n_ages, d)
        if self.siblings:
            self.shop = nn.Embedding(cfg.n_shops + 1, d, padding_idx=cfg.n_shops)
            self.sib_feat = nn.Linear(3, d)
            self.sib_q = nn.Linear(d, d)
            self.sib_att = SetAttention(d, cfg.n_heads, cfg.dropout)
            self.sib_norm = nn.LayerNorm(d)
        if self.analogues:
            self.emb_proj = nn.Sequential(nn.LayerNorm(cfg.la_emb_dim), nn.Linear(cfg.la_emb_dim, d))
            self.ana_feat = nn.Linear(4, d)
            self.ana_q = nn.Linear(d, d)
            self.ana_att = SetAttention(d, cfg.n_heads, cfg.dropout)
            self.ana_norm = nn.LayerNorm(d)

    def forward(self, batch: Dict[str, torch.Tensor], context: torch.Tensor) -> torch.Tensor:
        age = self.age_q(batch["age_idx"])  # (B, S, d)
        base = context.unsqueeze(1) + age
        outs = []
        if self.siblings:
            val, mask = batch["la_sib_val"], batch["la_sib_mask"].bool()
            lag = batch["la_sib_lag"].unsqueeze(-1).expand_as(val)
            x = self.sib_feat(torch.stack([val, mask.to(val.dtype), lag], -1)) + self.shop(batch["la_sib_shop"]).unsqueeze(2)
            q = self.sib_q(base + self.shop(batch["la_self_shop"]).unsqueeze(1))
            o, _ = self.sib_att(q, x, mask)
            outs.append(self.sib_norm(o))
        if self.analogues:
            val, mask = batch["la_ana_val"], batch["la_ana_mask"].bool()
            sim = batch["la_ana_sim"].unsqueeze(-1).expand_as(val)
            x = self.ana_feat(torch.stack([val, mask.to(val.dtype), batch["la_ana_support"], sim], -1))
            x = x + self.emb_proj(batch["la_ana_emb"]).unsqueeze(2)
            q = self.ana_q(base + self.emb_proj(batch["la_self_emb"]).unsqueeze(1))
            o, _ = self.ana_att(q, x, mask)
            outs.append(self.ana_norm(o))
        return torch.stack(outs, dim=-2)


class CrossAttentionTemporalFusionDecoder(nn.Module):
    """TFT temporal-fusion decoder with an inserted multimodal cross-attention block.

    temporal (B, L+H, d) -> static enrichment -> causal interpretable self-attention (decoder
    queries) -> cross-attention to product tokens (B, m, d) -> position-wise GRN -> gated skip
    to the LSTM output -> monotone quantile head (B, H, Q).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        quantiles: Sequence[float],
        encoder_length: int,
        horizon: int,
        use_cross_attention: bool = True,
    ) -> None:
        super().__init__()
        self.encoder_length, self.horizon = encoder_length, horizon
        self.enrichment = GatedResidualNetwork(d_model, d_model, d_model, d_model, dropout)
        self.self_attn = InterpretableMultiHeadAttention(d_model, n_heads, dropout)
        self.self_attn_gate = GateAddNorm(d_model, d_model, dropout)
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.cross_norm = nn.LayerNorm(d_model)
            self.cross_attn = MultimodalCrossAttention(d_model, n_heads, dropout)
            self.cross_gate = GateAddNorm(d_model, d_model, dropout)
        self.positionwise = GatedResidualNetwork(d_model, d_model, d_model, None, dropout)
        self.output_gate = GateAddNorm(d_model, d_model, 0.0)
        self.head = MonotoneQuantileHead(d_model, quantiles)
        total = encoder_length + horizon
        q_pos = torch.arange(horizon)[:, None] + encoder_length
        self.register_buffer("causal_mask", torch.arange(total)[None, :] > q_pos, persistent=False)

    def forward(
        self,
        temporal: torch.Tensor,
        static_enrichment: torch.Tensor,
        mm_tokens: Optional[torch.Tensor] = None,
        mm_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        L = self.encoder_length
        enriched = self.enrichment(temporal, static_enrichment)
        q = enriched[:, L:]
        attn_out, self_w = self.self_attn(q, enriched, enriched, mask=self.causal_mask)
        x = self.self_attn_gate(attn_out, q)
        cross_w = None
        if self.use_cross_attention:
            if mm_tokens is None:
                raise ValueError("cross-attention decoder requires mm_tokens")
            c, cross_w = self.cross_attn(self.cross_norm(x), mm_tokens, mm_key_padding_mask)
            x = self.cross_gate(c, x)
        x = self.positionwise(x)
        x = self.output_gate(x, temporal[:, L:])
        return self.head(x), {"self_attention": self_w, "cross_attention": cross_w}


class MultimodalTemporalFusionTransformer(nn.Module):
    """Global multi-horizon quantile forecaster.

    Batch keys:
        static_cat (B, n_sc) long, static_real (B, n_sr), enc_observed (B, L, n_o),
        enc_known_real (B, L, n_k), enc_known_cat (B, L, n_kc) long,
        dec_known_real (B, H, n_k), dec_known_cat (B, H, n_kc) long,
        mm_image / mm_text (B, D), mm_mask (B, 2) bool, mm_scalar (B, k) [static_scalar],
        optional mm_image_tokens (B, P, D'), mm_text_tokens (B, S, D'), mm_text_token_mask (B, S).
    """

    def __init__(self, cfg: MTFTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.static_cat_emb = CategoricalEmbedding(cfg.static_categorical_cardinalities, d)
        n_sr = cfg.n_static_reals + (cfg.mm_scalar_dim if cfg.fusion == "static_scalar" else 0)
        self.static_real_emb = RealEmbedding(n_sr, d) if n_sr else None
        self.use_tokenizer = cfg.fusion in ("static_pooled", "cross_attention")
        if self.use_tokenizer:
            self.mm_tokenizer = MultimodalTokenizer(
                cfg.mm_embed_dim, d, cfg.mm_latents, cfg.n_heads, cfg.dropout, cfg.mm_token_embed_dim,
                cfg.mm_feature_dropout, cfg.mm_modality_dropout,
            )
        self.mm_as_static = cfg.fusion == "static_pooled" or (cfg.fusion == "cross_attention" and cfg.mm_static_context)
        n_static_vars = len(cfg.static_categorical_cardinalities) + n_sr + int(self.mm_as_static)
        self.static_vsn = VariableSelectionNetwork(n_static_vars, d, None, cfg.dropout)
        self.static_contexts = nn.ModuleDict(
            {k: GatedResidualNetwork(d, d, d, None, cfg.dropout) for k in ("selection", "enrichment", "state_h", "state_c")}
        )
        self.observed_emb = RealEmbedding(cfg.n_observed_reals, d)
        self.known_real_emb = RealEmbedding(cfg.n_known_reals, d)
        self.known_cat_emb = CategoricalEmbedding(cfg.known_categorical_cardinalities, d)
        n_kc = len(cfg.known_categorical_cardinalities)
        from .lifecycle import AGE_MAX, AGE_MIN

        n_ages = AGE_MAX - AGE_MIN + 1
        self.lifecycle = LifecycleAttention(cfg, n_ages) if cfg.lifecycle_attention else None
        n_la = (int(cfg.la_siblings) + int(cfg.la_analogues)) if cfg.lifecycle_attention else 0
        self.age_emb = nn.Embedding(n_ages, d) if cfg.age_embedding else None
        self.la_gate = nn.Parameter(torch.zeros(2)) if (cfg.la_gate and cfg.lifecycle_attention) else None  # (a, b), g = sigmoid(a n/L + b)
        self.encoder_vsn = VariableSelectionNetwork(cfg.n_observed_reals + cfg.n_known_reals + n_kc + n_la, d, d, cfg.dropout)
        self.decoder_vsn = VariableSelectionNetwork(cfg.n_known_reals + n_kc + n_la, d, d, cfg.dropout)
        lstm_do = cfg.dropout if cfg.lstm_layers > 1 else 0.0
        self.lstm_encoder = nn.LSTM(d, d, cfg.lstm_layers, batch_first=True, dropout=lstm_do)
        self.lstm_decoder = nn.LSTM(d, d, cfg.lstm_layers, batch_first=True, dropout=lstm_do)
        self.post_lstm_gate = GateAddNorm(d, d, cfg.dropout)
        self.decoder = CrossAttentionTemporalFusionDecoder(
            d, cfg.n_heads, cfg.dropout, cfg.quantiles, cfg.encoder_length, cfg.horizon, cfg.fusion == "cross_attention"
        )
        self.return_mm_attribution = False  # set True at inference to get per-step modality attribution

    def _known(self, reals: torch.Tensor, cats: torch.Tensor) -> List[torch.Tensor]:
        parts = [self.known_real_emb(reals)]
        if len(self.known_cat_emb):
            parts.append(self.known_cat_emb(cats))
        return parts

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        static_parts = [self.static_cat_emb(batch["static_cat"])]
        if self.static_real_emb is not None:
            reals = batch["static_real"]
            if cfg.fusion == "static_scalar":
                reals = torch.cat([reals, batch["mm_scalar"]], dim=-1)
            static_parts.append(self.static_real_emb(reals))
        mm_tokens, align = None, {}
        if self.use_tokenizer:
            mm_tokens, align = self.mm_tokenizer(
                batch["mm_image"],
                batch["mm_text"],
                batch.get("mm_mask"),
                batch.get("mm_image_tokens"),
                batch.get("mm_text_tokens"),
                batch.get("mm_text_token_mask"),
                need_weights=self.return_mm_attribution,
            )
            if self.mm_as_static:
                static_parts.append(mm_tokens.mean(1, keepdim=True))
        static_emb, static_w = self.static_vsn(torch.cat(static_parts, dim=1))
        c_sel, c_enr, c_h, c_c = (self.static_contexts[k](static_emb) for k in ("selection", "enrichment", "state_h", "state_c"))

        enc_parts = [self.observed_emb(batch["enc_observed"]), *self._known(batch["enc_known_real"], batch["enc_known_cat"])]
        dec_parts = list(self._known(batch["dec_known_real"], batch["dec_known_cat"]))
        L = cfg.encoder_length
        if self.lifecycle is not None:
            la = self.lifecycle(batch, c_sel)  # (B, L+H, n_la, d)
            if self.la_gate is not None:
                from .data import KNOWN_REALS

                n_obs = batch["enc_known_real"][..., KNOWN_REALS.index("launched")].sum(1) / L  # share of observed own history
                g = torch.sigmoid(self.la_gate[0] * n_obs + self.la_gate[1])
                la = la * g[:, None, None, None]
            enc_parts.append(la[:, :L])
            dec_parts.append(la[:, L:])
        enc_x, enc_w = self.encoder_vsn(torch.cat(enc_parts, dim=-2), c_sel)
        dec_x, dec_w = self.decoder_vsn(torch.cat(dec_parts, dim=-2), c_sel)
        if self.age_emb is not None:
            age = self.age_emb(batch["age_idx"])
            enc_x, dec_x = enc_x + age[:, :L], dec_x + age[:, L:]

        layers = cfg.lstm_layers
        state = (c_h.unsqueeze(0).repeat(layers, 1, 1).contiguous(), c_c.unsqueeze(0).repeat(layers, 1, 1).contiguous())
        enc_out, state = self.lstm_encoder(enc_x, state)
        dec_out, _ = self.lstm_decoder(dec_x, state)
        temporal = self.post_lstm_gate(torch.cat([enc_out, dec_out], 1), torch.cat([enc_x, dec_x], 1))

        pred, interp = self.decoder(temporal, c_enr, mm_tokens, None)
        attribution = None
        if interp["cross_attention"] is not None and "resampler_attention" in align:
            # (B, H, m) @ (B, m, n_tokens): how much each decoder step draws on image / text / fused evidence
            attribution = interp["cross_attention"] @ align["resampler_attention"]
        return {
            "prediction": pred,
            "static_weights": static_w,
            "encoder_weights": enc_w,
            "decoder_weights": dec_w,
            "self_attention": interp["self_attention"],
            "cross_attention": interp["cross_attention"],
            "modality_attribution": attribution,
            "align": align,
        }
