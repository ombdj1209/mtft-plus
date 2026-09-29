"""Joint vision-language product representations.

* ``MultimodalFeatureExtractor``: frozen SigLIP (HF transformers) or OpenCLIP encoder with a
  content-addressed on-disk cache, producing an ``EmbeddingBank`` of L2-normalised image/text
  embeddings in the *shared* contrastive space (optionally patch/word tokens as well).
* ``MultimodalTokenizer``: trainable projection of the frozen embeddings into the forecaster's
  d_model space, a fused (image ⊙ text) token that is only meaningful because both modalities
  live in one aligned space, and a Perceiver-style resampler that emits m latent product tokens
  used as keys/values by the temporal cross-attention.
* ``siglip_alignment_loss``: sigmoid contrastive regulariser keeping the *projected* image and
  text tokens aligned while the forecaster fine-tunes the projections.
"""
from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from .layers import GatedResidualNetwork

ImageLike = Union[str, Path, bytes, "PIL.Image.Image", None]  # noqa: F821


@dataclass
class ArticleRecord:
    article_id: str
    text: Optional[str] = None
    image: ImageLike = None


@dataclass
class EmbeddingBank:
    """Row-aligned multimodal embeddings for a catalogue of articles."""

    article_ids: List[str]
    image: torch.Tensor  # (N, D) L2-normalised, zeros where missing
    text: torch.Tensor  # (N, D)
    mask: torch.Tensor  # (N, 2) bool: [has_image, has_text]
    image_tokens: Optional[torch.Tensor] = None  # (N, P, D_v)
    text_tokens: Optional[torch.Tensor] = None  # (N, S, D_t)
    text_token_mask: Optional[torch.Tensor] = None  # (N, S) bool, True = valid
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._pos = {a: i for i, a in enumerate(self.article_ids)}

    def __len__(self) -> int:
        return len(self.article_ids)

    @property
    def dim(self) -> int:
        return int(self.image.shape[-1])

    def indices(self, article_ids: Iterable[str]) -> torch.Tensor:
        return torch.tensor([self._pos[a] for a in article_ids], dtype=torch.long)

    def save(self, path: Union[str, Path]) -> None:
        state = {k: getattr(self, k) for k in ("article_ids", "image", "text", "mask", "image_tokens", "text_tokens", "text_token_mask", "meta")}
        torch.save(state, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EmbeddingBank":
        return cls(**torch.load(path, map_location="cpu", weights_only=False))


def _as_tensor(out: Any) -> torch.Tensor:
    """Normalise transformers return types across versions (tensor vs ModelOutput)."""
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("pooler_output", "image_embeds", "text_embeds", "last_hidden_state"):
        val = getattr(out, attr, None)
        if isinstance(val, torch.Tensor):
            return val
    raise TypeError(f"cannot extract tensor from {type(out)!r}")


class MultimodalFeatureExtractor:
    """Frozen joint vision-language encoder with a persistent embedding cache.

    Args:
        model_name: HF id (``backend="siglip"``) or OpenCLIP name, e.g. ``"hf-hub:timm/ViT-B-16-SigLIP"``.
        backend: ``"siglip"`` (transformers ``AutoModel``/``AutoProcessor``) or ``"open_clip"``.
        cache_dir: directory for per-article ``.pt`` entries keyed by sha1(model, text, image bytes).
        return_tokens: also cache patch/word tokens (SigLIP backend only) for token-level cross-attention.
        model / processor: optional pre-built objects (dependency injection, tests, custom checkpoints).
    """

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        backend: str = "siglip",
        cache_dir: Union[str, Path] = ".mm_cache",
        device: Optional[str] = None,
        batch_size: int = 32,
        max_text_length: int = 64,
        return_tokens: bool = False,
        cache_dtype: torch.dtype = torch.float16,
        pretrained: Optional[str] = None,
        model: Optional[nn.Module] = None,
        processor: Optional[Any] = None,
    ) -> None:
        if backend not in ("siglip", "open_clip"):
            raise ValueError("backend must be 'siglip' or 'open_clip'")
        if return_tokens and backend != "siglip":
            raise ValueError("token caching is implemented for the SigLIP backend only")
        self.model_name, self.backend, self.pretrained = model_name, backend, pretrained
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size, self.max_text_length = batch_size, max_text_length
        self.return_tokens, self.cache_dtype = return_tokens, cache_dtype
        self.model, self.processor, self._tokenizer = model, processor, None
        if self.model is not None:
            self.model.eval().to(self.device)

    # ------------------------------------------------------------------ loading
    def _ensure_loaded(self) -> None:
        if self.model is not None and (self.processor is not None or self.backend == "open_clip"):
            return
        if self.backend == "siglip":
            from transformers import AutoModel, AutoProcessor

            self.model = AutoModel.from_pretrained(self.model_name).eval().to(self.device)
            self.processor = AutoProcessor.from_pretrained(self.model_name)
        else:
            import open_clip

            model, _, preprocess = open_clip.create_model_and_transforms(self.model_name, pretrained=self.pretrained)
            self.model, self.processor = model.eval().to(self.device), preprocess
            self._tokenizer = open_clip.get_tokenizer(self.model_name)
        for p in self.model.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------ caching
    @staticmethod
    def _image_bytes(image: ImageLike) -> bytes:
        if image is None:
            return b""
        if isinstance(image, bytes):
            return image
        if isinstance(image, (str, Path)):
            return Path(image).read_bytes()
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    def cache_key(self, rec: ArticleRecord) -> str:
        h = hashlib.sha1()
        h.update(f"{self.backend}|{self.model_name}|{self.pretrained}|tok={self.return_tokens}|L={self.max_text_length}|".encode())
        h.update((rec.text or "").encode("utf-8"))
        h.update(b"|")
        h.update(hashlib.sha1(self._image_bytes(rec.image)).digest())
        return h.hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.pt"

    # ------------------------------------------------------------------ encoding
    @staticmethod
    def _load_image(image: ImageLike):
        from PIL import Image

        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        if isinstance(image, bytes):
            return Image.open(io.BytesIO(image)).convert("RGB")
        return image.convert("RGB")

    @torch.inference_mode()
    def _encode_batch(self, recs: Sequence[ArticleRecord]) -> List[Dict[str, torch.Tensor]]:
        self._ensure_loaded()
        n = len(recs)
        has_img = [r.image is not None for r in recs]
        has_txt = [bool(r.text) for r in recs]
        loaded = {}
        for i in range(n):  # unreadable files (e.g. one corrupt PNG in VISUELLE 2.0) are treated as a missing image
            if has_img[i]:
                try:
                    loaded[i] = self._load_image(recs[i].image)
                except OSError as e:
                    print(f"warning: image of {recs[i].article_id} unreadable ({e}); treated as missing", flush=True)
                    has_img[i] = False
        img_idx = [i for i in range(n) if has_img[i]]
        images = [loaded[i] for i in img_idx]
        texts = [r.text or "" for r in recs]
        img_pooled = img_tok = txt_pooled = txt_tok = txt_tok_mask = None

        if self.backend == "siglip":
            txt_in = self.processor(text=texts, padding="max_length", truncation=True, max_length=self.max_text_length, return_tensors="pt")
            input_ids = txt_in["input_ids"].to(self.device)
            if self.return_tokens:
                tout = self.model.text_model(input_ids=input_ids)
                txt_tok, txt_pooled = tout.last_hidden_state, tout.pooler_output
                pad_id = getattr(getattr(self.processor, "tokenizer", None), "pad_token_id", None)
                txt_tok_mask = input_ids != pad_id if pad_id is not None else torch.ones_like(input_ids, dtype=torch.bool)
            else:
                txt_pooled = _as_tensor(self.model.get_text_features(input_ids=input_ids))
            if images:
                pix = self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)
                if self.return_tokens:
                    vout = self.model.vision_model(pixel_values=pix)
                    img_tok, img_pooled = vout.last_hidden_state, vout.pooler_output
                else:
                    img_pooled = _as_tensor(self.model.get_image_features(pixel_values=pix))
        else:
            txt_pooled = self.model.encode_text(self._tokenizer(texts).to(self.device))
            if images:
                pix = torch.stack([self.processor(im) for im in images]).to(self.device)
                img_pooled = self.model.encode_image(pix)

        d = txt_pooled.shape[-1]
        img_full = torch.zeros(n, d, device=self.device)
        if img_pooled is not None:
            img_full[img_idx] = F.normalize(img_pooled.float(), dim=-1)
        txt_full = F.normalize(txt_pooled.float(), dim=-1) * torch.tensor(has_txt, device=self.device, dtype=torch.float32)[:, None]

        out: List[Dict[str, torch.Tensor]] = []
        pos = {j: k for k, j in enumerate(img_idx)}
        for i in range(n):
            entry = {
                "image": img_full[i].to(self.cache_dtype).cpu(),
                "text": txt_full[i].to(self.cache_dtype).cpu(),
                "mask": torch.tensor([has_img[i], has_txt[i]]),
            }
            if self.return_tokens:
                if img_tok is not None and i in pos:
                    entry["image_tokens"] = img_tok[pos[i]].to(self.cache_dtype).cpu()
                entry["text_tokens"] = txt_tok[i].to(self.cache_dtype).cpu()
                entry["text_token_mask"] = (txt_tok_mask[i] & has_txt[i]).cpu()
            out.append(entry)
        return out

    def encode(self, records: Sequence[ArticleRecord], overwrite: bool = False) -> EmbeddingBank:
        """Encode (or load from cache) every record; returns a row-aligned ``EmbeddingBank``."""
        keys = [self.cache_key(r) for r in records]
        entries: List[Optional[Dict[str, torch.Tensor]]] = [None] * len(records)
        todo = []
        for i, k in enumerate(keys):
            p = self._cache_path(k)
            if p.exists() and not overwrite:
                entries[i] = torch.load(p, map_location="cpu", weights_only=True)
            else:
                todo.append(i)
        for s in range(0, len(todo), self.batch_size):
            chunk = todo[s : s + self.batch_size]
            for i, entry in zip(chunk, self._encode_batch([records[j] for j in chunk])):
                p = self._cache_path(keys[i])
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp = p.with_suffix(".tmp")
                torch.save(entry, tmp)
                tmp.replace(p)  # atomic publish: safe under concurrent workers
                entries[i] = entry
        return self._stack(records, entries)

    def _stack(self, records: Sequence[ArticleRecord], entries: Sequence[Dict[str, torch.Tensor]]) -> EmbeddingBank:
        image = torch.stack([e["image"] for e in entries])
        text = torch.stack([e["text"] for e in entries])
        mask = torch.stack([e["mask"] for e in entries]).bool()
        image_tokens = text_tokens = text_token_mask = None
        if self.return_tokens:
            ref = next((e["image_tokens"] for e in entries if "image_tokens" in e), None)
            if ref is not None:
                image_tokens = torch.stack([e.get("image_tokens", torch.zeros_like(ref)) for e in entries])
            text_tokens = torch.stack([e["text_tokens"] for e in entries])
            text_token_mask = torch.stack([e["text_token_mask"] for e in entries]).bool()
        return EmbeddingBank(
            article_ids=[r.article_id for r in records],
            image=image,
            text=text,
            mask=mask,
            image_tokens=image_tokens,
            text_tokens=text_tokens,
            text_token_mask=text_token_mask,
            meta={"backend": self.backend, "model_name": self.model_name},
        )


def _mlp(d_in: int, d_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d_out), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_out, d_out))


class MultimodalTokenizer(nn.Module):
    """Frozen joint embeddings -> m latent product tokens in d_model space.

    Input tokens: [image, text, image ⊙ text] (+ optional patch / word tokens), each with a
    learned modality-type embedding. A set of learned latent queries cross-attends to them
    (Perceiver resampler) so downstream cost is O(H·m) regardless of patch count.
    """

    TYPE_IMAGE, TYPE_TEXT, TYPE_FUSED, TYPE_PATCH, TYPE_WORD = range(5)

    def __init__(
        self,
        d_embed: int,
        d_model: int,
        n_latents: int = 8,
        n_heads: int = 4,
        dropout: float = 0.1,
        d_token_embed: Optional[int] = None,
        feature_dropout: float = 0.0,
        modality_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.feature_dropout, self.modality_dropout = feature_dropout, modality_dropout
        self.image_proj = _mlp(d_embed, d_model, dropout)
        self.text_proj = _mlp(d_embed, d_model, dropout)
        self.fused_proj = _mlp(d_embed, d_model, dropout)
        d_tok = d_token_embed or d_embed
        self.patch_proj = nn.Sequential(nn.LayerNorm(d_tok), nn.Linear(d_tok, d_model))
        self.word_proj = nn.Sequential(nn.LayerNorm(d_tok), nn.Linear(d_tok, d_model))
        self.type_embedding = nn.Embedding(5, d_model)
        self.latents = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)
        self.resampler = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.resampler_norm = nn.LayerNorm(d_model)
        self.ff = GatedResidualNetwork(d_model, 2 * d_model, d_model, None, dropout)
        # SigLIP temperature / bias initialisation (Zhai et al., 2023)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.logit_bias = nn.Parameter(torch.tensor(-10.0))

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
        image_tokens: Optional[torch.Tensor] = None,
        text_tokens: Optional[torch.Tensor] = None,
        text_token_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        b = image.shape[0]
        if modality_mask is None:
            modality_mask = torch.ones(b, 2, dtype=torch.bool, device=image.device)
        mi, mt = modality_mask[:, 0], modality_mask[:, 1]
        if self.training and self.modality_dropout > 0:
            # drop exactly one modality for a random subset of rows (never both): robustness to missing photos/text
            drop = torch.rand(b, device=image.device) < self.modality_dropout
            which = torch.rand(b, device=image.device) < 0.5
            both = mi & mt
            mi = mi & ~(drop & which & both)
            mt = mt & ~(drop & ~which & both)
        zi = F.normalize(image.float(), dim=-1) * mi[:, None]
        zt = F.normalize(text.float(), dim=-1) * mt[:, None]
        if self.training and self.feature_dropout > 0:
            zi = F.dropout(zi, self.feature_dropout)
            zt = F.dropout(zt, self.feature_dropout)
        scale = math.sqrt(zi.shape[-1])  # keep the Hadamard product at O(1) magnitude
        ti, tt = self.image_proj(zi), self.text_proj(zt)
        tf = self.fused_proj(zi * zt * scale)
        types = self.type_embedding.weight
        tokens = [ti + types[self.TYPE_IMAGE], tt + types[self.TYPE_TEXT], tf + types[self.TYPE_FUSED]]
        pad = [~mi, ~mt, torch.zeros_like(mi)]  # fused token never padded -> no all-masked rows
        tokens = [t.unsqueeze(1) for t in tokens]
        pad = [p.unsqueeze(1) for p in pad]
        if image_tokens is not None:
            tokens.append(self.patch_proj(image_tokens.float()) + types[self.TYPE_PATCH])
            pad.append((~mi)[:, None].expand(-1, image_tokens.shape[1]))
        if text_tokens is not None:
            tokens.append(self.word_proj(text_tokens.float()) + types[self.TYPE_WORD])
            tm = text_token_mask if text_token_mask is not None else mt[:, None].expand(-1, text_tokens.shape[1])
            pad.append(~tm.bool())
        kv = torch.cat(tokens, dim=1)
        kpm = torch.cat(pad, dim=1)
        q = self.latents.unsqueeze(0).expand(b, -1, -1)
        attn_out, attn_w = self.resampler(q, kv, kv, key_padding_mask=kpm, need_weights=need_weights, average_attn_weights=True)
        latents = self.ff(self.resampler_norm(q + attn_out))
        aux = {"image_latent": ti, "text_latent": tt, "pair_mask": mi & mt}
        if need_weights:
            aux["resampler_attention"] = attn_w  # (B, m, n_tokens); first three = image, text, fused
        return latents, aux


def siglip_alignment_loss(
    image_latent: torch.Tensor,
    text_latent: torch.Tensor,
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
    group_ids: Optional[torch.Tensor] = None,
    valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pairwise sigmoid loss  L = -1/n Σ_ij log σ(z_ij (t·<x_i, y_j> + b)),  z_ij = +1 iff i = j.

    ``group_ids`` de-duplicates articles that appear several times in a batch (one row per
    fulfilment centre), which would otherwise create false negatives.
    """
    if valid is not None:
        image_latent, text_latent = image_latent[valid], text_latent[valid]
        group_ids = group_ids[valid] if group_ids is not None else None
    if group_ids is not None and group_ids.numel() > 0:
        uniq, inv = torch.unique(group_ids, return_inverse=True)
        first = torch.full((uniq.numel(),), inv.numel(), device=inv.device, dtype=torch.long)
        first = first.scatter_reduce(0, inv, torch.arange(inv.numel(), device=inv.device), reduce="amin")
        image_latent, text_latent = image_latent[first], text_latent[first]
    n = image_latent.shape[0]
    if n < 2:
        return image_latent.sum() * 0.0
    x = F.normalize(image_latent, dim=-1)
    y = F.normalize(text_latent, dim=-1)
    logits = x @ y.T * logit_scale.exp() + logit_bias
    labels = 2.0 * torch.eye(n, device=logits.device, dtype=logits.dtype) - 1.0
    return -F.logsigmoid(labels * logits).sum() / n
