"""PyTorch Lightning wrapper: quantile objective + optional SigLIP alignment regulariser."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import lightning.pytorch as pl
import torch

from .features import siglip_alignment_loss
from .losses import QuantileLoss
from .model import MTFTConfig, MultimodalTemporalFusionTransformer


class MTFTLightningModule(pl.LightningModule):
    def __init__(
        self,
        config: Dict[str, Any],
        learning_rate: float = 3e-3,
        weight_decay: float = 1e-4,
        quantile_weights: Optional[Sequence[float]] = None,
        warmup_frac: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.cfg = MTFTConfig(**config)
        self.model = MultimodalTemporalFusionTransformer(self.cfg)
        self.loss_fn = QuantileLoss(self.cfg.quantiles, quantile_weights)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(batch)

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        out = self.model(batch)
        q_loss = self.loss_fn(out["prediction"], batch["target"], mask=batch["target_mask"])
        loss = q_loss
        align = out["align"]
        if self.cfg.align_weight > 0 and align:
            tok = self.model.mm_tokenizer
            a_loss = siglip_alignment_loss(
                align["image_latent"], align["text_latent"], tok.logit_scale, tok.logit_bias, batch.get("article_idx"), align["pair_mask"]
            )
            loss = loss + self.cfg.align_weight * a_loss
            self.log(f"{stage}_align", a_loss, batch_size=batch["target"].shape[0])
        bs = batch["target"].shape[0]
        self.log(f"{stage}_quantile_loss", q_loss, prog_bar=stage == "val", batch_size=bs)
        self.log(f"{stage}_loss", loss, batch_size=bs)
        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        return self.model(batch)["prediction"]

    def configure_optimizers(self):
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim < 2 or "embedding" in name or name.endswith("latents") else decay).append(p)
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": self.hparams.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=self.hparams.learning_rate,
        )
        total = max(int(self.trainer.estimated_stepping_batches), 1)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.hparams.learning_rate, total_steps=total, pct_start=self.hparams.warmup_frac, anneal_strategy="cos"
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
