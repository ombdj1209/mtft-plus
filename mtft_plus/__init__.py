"""MTFT+ : contrastive multimodal, cross-attention Temporal Fusion Transformer with MinT reconciliation."""
from .features import ArticleRecord, EmbeddingBank, MultimodalFeatureExtractor, MultimodalTokenizer, siglip_alignment_loss
from .losses import QuantileLoss, newsvendor_quantile
from .model import (
    CrossAttentionTemporalFusionDecoder,
    MTFTConfig,
    MultimodalCrossAttention,
    MultimodalTemporalFusionTransformer,
)
from .reconciliation import Hierarchy, MinTReconciler, SeasonalRidgeForecaster, shrinkage_intensity, swanson_mean

__all__ = [
    "ArticleRecord",
    "EmbeddingBank",
    "MultimodalFeatureExtractor",
    "MultimodalTokenizer",
    "siglip_alignment_loss",
    "QuantileLoss",
    "newsvendor_quantile",
    "CrossAttentionTemporalFusionDecoder",
    "MTFTConfig",
    "MultimodalCrossAttention",
    "MultimodalTemporalFusionTransformer",
    "Hierarchy",
    "MinTReconciler",
    "SeasonalRidgeForecaster",
    "shrinkage_intensity",
    "swanson_mean",
]
__version__ = "0.1.0"
