"""Re-run the published VISUELLE 2.0 baselines with the official code (pre-registration amendment (d) 2).

    python examples/visuelle2_published.py --model crossattn --task 2-1 --use-img 1 --seeds 1,2,3,4,5
    python examples/visuelle2_published.py --model gtm --task demand --seeds 1,2,3,4,5
    python examples/visuelle2_published.py --model crossattn --task 2-1 --use-img 1 --official-selection   # diagnostic

Official code: third_party/visuelle2.0-code-main (CrossAttnRNN21 / CrossAttnRNN210 / CrossAttnRNNDemand, dataset.py)
and third_party/GTM-Transformer-main (GTM). The model classes, dataset framing, losses, optimiser and metrics are
the official ones. Patches (all listed in results/visuelle2/published_baselines.md):
  1. ``fairseq.optim.adafactor.Adafactor`` -> ``transformers.optimization.Adafactor`` (same algorithm and arguments);
  2. PyTorch Lightning 1.6 ``validation_epoch_end`` -> Lightning 2 ``on_validation_epoch_end`` (same metrics);
  3. frozen image / text encoders are evaluated once and cached (inference mode):
     * CrossAttnRNN 2-1 / 2-10: Inception-v3 up to Mixed_7a is frozen in the official code and cached here; the
       trainable Mixed_7b, Mixed_7c and all heads are trained as in the official code (only difference: the frozen
       trunk's batch-norm runs in inference mode);
     * CrossAttnRNN demand: the released code feeds 299-px images to a ResNet-101 whose output it reshapes as 8x8,
       which fails; images are resized to 256 px (8x8 maps). Since amendment (h) the full-budget and diagnostic runs
       cache ResNet-101 only up to layer2 and fine-tune layer3 and layer4 as the released code does
       (CrossAttnRNNDemand.py lines 74-82); layer3 blocks use gradient checkpointing (exact: batch-norm momentum is 0
       during the recomputation, so running averages update once per step, as without checkpointing). The early-
       stopped (secondary) runs froze and cached the full ResNet-101;
     * GTM-Transformer: ResNet-50 and BERT are frozen in the official code, so caching is exact;
  4. GTM-Transformer is a product-level model; a store embedding is added to its static features (as in the official
     CrossAttnRNN demand model) so that it forecasts per shop. It has no own-history input: demand task only.
  5. Validation: checkpoints are selected on the 10 % most recent *training* pairs (the split our models use);
     ``--official-selection`` reproduces the official choice (all training pairs, test set as validation, seed 21),
     for diagnosis only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

# The official dataset caches TensorDatasets with torch.save and reloads them with torch.load, whose default became
# weights_only=True in PyTorch 2.6; the files are this runner's own caches, so the old default is restored here.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "third_party" / "visuelle2.0-code-main"
GTMDIR = ROOT / "third_party" / "GTM-Transformer-main"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- patch 1: fairseq Adafactor shim (must exist before the official modules are imported)
from transformers.optimization import Adafactor as _Adafactor

_fs = types.ModuleType("fairseq"); _fo = types.ModuleType("fairseq.optim"); _fa = types.ModuleType("fairseq.optim.adafactor")
_fa.Adafactor = _Adafactor
sys.modules.update({"fairseq": _fs, "fairseq.optim": _fo, "fairseq.optim.adafactor": _fa})

import pytorch_lightning as pl  # the official modules subclass pytorch_lightning.LightningModule

W = [str(i) for i in range(12)]
CACHE = ROOT / ".published_cache"
TASK_MODE = {"2-1": 0, "2-10": 1}


# ---------------------------------------------------------------------------------------------------- image features
def _img_tensor(path: Path, size: int):
    from PIL import Image
    from torchvision.transforms import Compose, Normalize, Resize, ToTensor

    tf = Compose([Resize((size, size)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    try:
        return tf(Image.open(path).convert("RGB")), True
    except OSError:  # one corrupt file in the distributed archive (AI19/04442.png): zero features
        return torch.zeros(3, size, size), False


class Bank:
    """Cached image features, one row per image: ``x[pos[image_path]]``."""

    def __init__(self, paths: list, x: torch.Tensor):
        self.pos, self.x = {p: i for i, p in enumerate(paths)}, x


def cached_features(kind: str, paths: list, src: Path, device: str = "cuda", save: bool = True) -> Bank:
    """Feature bank; kind: inception7a (1280x8x8 @299), resnet101 / resnet50 (2048x8x8 @256), resnet101_l2 (ResNet-101
    up to layer2, 512x32x32 @256; layer3-4 are trained). Stored fp16."""
    out_path = CACHE / f"{kind}.pt"
    if save and out_path.exists():
        obj = torch.load(out_path, weights_only=False)
        if isinstance(obj, dict) and "bank" in obj:
            return Bank(obj["paths"], obj["bank"])
        keys = list(obj)
        return Bank(keys, torch.stack([obj.pop(k) for k in keys]))  # older caches: {path: tensor}
    return _compute_features(kind, paths, src, torch.device(device), out_path if save else None)


@torch.inference_mode()
def _compute_features(kind: str, paths: list, src: Path, dev: torch.device, out_path) -> Bank:
    import torchvision.models as tvm

    if kind == "inception7a":
        net = tvm.inception_v3(weights=tvm.Inception_V3_Weights.IMAGENET1K_V1)  # transform_input=True as with pretrained=True
        net.eval().to(dev)
        stem = ["Conv2d_1a_3x3", "Conv2d_2a_3x3", "Conv2d_2b_3x3", "maxpool1", "Conv2d_3b_1x1", "Conv2d_4a_3x3", "maxpool2",
                "Mixed_5b", "Mixed_5c", "Mixed_5d", "Mixed_6a", "Mixed_6b", "Mixed_6c", "Mixed_6d", "Mixed_6e", "Mixed_7a"]
        f = lambda x: _seq(net, stem, net._transform_input(x))  # noqa: E731
        size = 299
    else:
        net = (tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1) if kind == "resnet50" else tvm.resnet101(weights=tvm.ResNet101_Weights.IMAGENET1K_V1))
        # released demand ImageEncoder: children()[:-2]; children()[6:] (layer3, layer4) are the fine-tuned part
        trunk = nn.Sequential(*list(net.children())[: 6 if kind == "resnet101_l2" else -2]).eval().to(dev)
        f = trunk
        size = 256
    bad = []
    uniq = sorted(set(paths))
    bank = None
    for s in range(0, len(uniq), 64):
        chunk = uniq[s : s + 64]
        xs, oks = zip(*[_img_tensor(src / "images" / p, size) for p in chunk])
        y = f(torch.stack(xs).to(dev)).half().cpu()
        if bank is None:
            bank = torch.empty((len(uniq), *y.shape[1:]), dtype=torch.float16)
        for j, (p, ok) in enumerate(zip(chunk, oks)):
            bank[s + j] = y[j] if ok else 0
            if not ok:
                bad.append(p)
    if out_path is not None:
        CACHE.mkdir(exist_ok=True)
        torch.save({"paths": uniq, "bank": bank}, out_path)
    print(f"cached {kind}: {len(uniq)} images, unreadable -> zeros: {bad}", flush=True)
    return Bank(uniq, bank)


def _seq(net, names, x):
    for n in names:
        x = getattr(net, n)(x)
    return x


class CachedInception(nn.Module):
    """Official ImageEncoder(embedding_dim, fine_tune=True) with the frozen trunk up to Mixed_7a precomputed."""

    def __init__(self, embedding_dim, fine_tune=True):
        super().__init__()
        import torchvision.models as tvm

        net = tvm.inception_v3(weights=tvm.Inception_V3_Weights.IMAGENET1K_V1)
        self.Mixed_7b, self.Mixed_7c = net.Mixed_7b, net.Mixed_7c  # the blocks the official code fine-tunes
        self.fc = nn.Linear(2048, embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):  # x: cached Mixed_7a output (B, 1280, 8, 8)
        out = self.Mixed_7c(self.Mixed_7b(x.float()))
        out = torch.flatten(out, 1).reshape(-1, 64, 2048)  # identical to the official avgpool/fc = Identity path
        return self.dropout(self.fc(out))


class CachedResNet(nn.Module):
    """Official demand ImageEncoder with the (here fully frozen) ResNet-101 output precomputed."""

    def __init__(self, embedding_dim=300):
        super().__init__()
        self.fc = nn.Linear(2048, embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x.float().view(-1, 64, 2048)  # the official (channel-scrambling) view is kept
        return self.dropout(self.fc(x))


class _NoRunningStatUpdate:
    """Context for a checkpoint recomputation: batch-norm momentum 0, so running averages are not updated twice."""

    def __init__(self, module: nn.Module):
        self.bns = [m for m in module.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]

    def __enter__(self):
        self.saved = [m.momentum for m in self.bns]
        for m in self.bns:
            m.momentum = 0.0

    def __exit__(self, *exc):
        for m, mom in zip(self.bns, self.saved):
            m.momentum = mom


class CachedResNetL2(nn.Module):
    """Released demand ImageEncoder (CrossAttnRNNDemand.py lines 70-92) on cached ResNet-101 layer2 outputs: layer3 and
    layer4 are trainable (the released ``children()[6:]``), then the released reshape / fc / dropout."""

    def __init__(self, embedding_dim=512, checkpoint_layer3: bool = True, checkpoint_layer4: bool = False):
        super().__init__()
        import torchvision.models as tvm

        net = tvm.resnet101(weights=tvm.ResNet101_Weights.IMAGENET1K_V1)
        self.layer3, self.layer4 = net.layer3, net.layer4
        self.fc = nn.Linear(2048, embedding_dim)
        self.dropout = nn.Dropout(0.1)
        self.checkpoint_layer3, self.checkpoint_layer4 = checkpoint_layer3, checkpoint_layer4

    def _run(self, layer, x, ckpt: bool):
        if not (ckpt and self.training and torch.is_grad_enabled()):
            return layer(x)
        from contextlib import nullcontext
        from torch.utils.checkpoint import checkpoint

        for block in layer:  # fp32 activations of layer3's 23 blocks at batch 128 exceed 8 GB; recomputation is exact
            x = checkpoint(block, x, use_reentrant=False, context_fn=lambda b=block: (nullcontext(), _NoRunningStatUpdate(b)))
        return x

    def forward(self, x):
        x = self._run(self.layer4, self._run(self.layer3, x.float(), self.checkpoint_layer3), self.checkpoint_layer4)
        x = x.view(-1, 64, 2048)  # the released (channel-scrambling) view
        return self.dropout(self.fc(x))


# ---------------------------------------------------------------------------------------------------- data
def split_frames(src: Path, official: bool):
    tr = pd.read_csv(src / "stfore_train.csv", parse_dates=["release_date"])
    te = pd.read_csv(src / "stfore_test.csv", parse_dates=["release_date"])
    if official:
        return tr.reset_index(drop=True), te.reset_index(drop=True), te.reset_index(drop=True)
    o = tr.assign(_c=tr.external_code.astype(str), _r=tr.retail.astype(str)).sort_values(["release_date", "_c", "_r"], kind="stable")
    n_val = int(round(0.1 * len(o)))
    va = tr.loc[o.index[-n_val:]].reset_index(drop=True)
    trn = tr.drop(index=o.index[-n_val:]).reset_index(drop=True)
    return trn, va, te.reset_index(drop=True)


def official_dataset(df, src: Path, demand: bool, bank: "Bank", tag: str):
    sys.path.insert(0, str(V2))
    from dataset import Visuelle2  # official framing: restock cleaning, windows, gtrends, temporal features

    cat, col, fab = (torch.load(src / f"{n}_labels.pt", weights_only=False) for n in ("category", "color", "fabric"))
    gtrends = pd.read_csv(src / "vis2_gtrends_data.csv", index_col=[0], parse_dates=True)
    CACHE.mkdir(exist_ok=True)
    ds = Visuelle2(df, str(src / "images"), gtrends, cat, col, fab, 52, demand, str(CACHE / f"{tag}.pt"))
    # one bank row per image, indexed per data row (a copy per row needs ~0.16 MB x rows: 14+ GB for the SO-fore train set)
    img = bank.x
    row = torch.as_tensor([bank.pos[p] for p in df.image_path], dtype=torch.long)

    class WithFeatures(torch.utils.data.Dataset):
        def __len__(self):
            return len(ds.dataset)

        def __getitem__(self, i):
            return ds.dataset[i], img[row[i]]

    return WithFeatures(), cat, col, fab


# ---------------------------------------------------------------------------------------------------- models (patch 2)
def with_epoch_end(cls, monitor_scale: float = 53.0):
    """Subclass an official LightningModule: collect validation outputs and compute the official metrics in
    on_validation_epoch_end (Lightning 2), instead of the removed validation_epoch_end."""

    class Patched(cls):
        validation_epoch_end = None  # removed in Lightning 2 (its presence raises); replaced by on_validation_epoch_end below

        def on_validation_epoch_start(self):
            super().on_validation_epoch_start() if hasattr(super(), "on_validation_epoch_start") else None
            self._outs = []

        def validation_step(self, batch, batch_idx):
            out = super().validation_step(batch, batch_idx)
            self._outs.append((out[0].detach(), out[1].detach()))

        def on_validation_epoch_end(self):
            y = torch.vstack([o[0].reshape(len(o[0]), -1) for o in self._outs]).squeeze()
            p = torch.vstack([o[1].reshape(len(o[1]), -1) for o in self._outs]).squeeze()
            ys, ps = y * monitor_scale, p * monitor_scale
            self.log("val_loss", F.mse_loss(y, p))
            self.log("val_mae", F.l1_loss(ys, ps))
            self.log("val_wWAPE", 100 * torch.sum(torch.abs(ys - ps)) / torch.sum(ys))

    Patched.__name__ = cls.__name__
    return Patched


def build_crossattn(task: str, use_img: int, cat, col, fab, demand_finetune: bool = False, memory_level: int = 0):
    sys.path.insert(0, str(V2))
    import models.CrossAttnRNN21 as m21
    import models.CrossAttnRNN210 as m210
    import models.CrossAttnRNNDemand as md

    m21.ImageEncoder = CachedInception  # patch 3 (see module docstring)
    m210.ImageEncoder = CachedInception
    # patch 6: the released demand model calls ImageEncoder() with its default 300-d output while its attention layers
    # use embedding_dim (512 by default), which fails at the first batch; the encoder is built with embedding_dim.
    md.ImageEncoder = (lambda *a, **k: CachedResNetL2(512, checkpoint_layer4=memory_level >= 1)) if demand_finetune else (lambda *a, **k: CachedResNet(512))
    # official forecast_dl.py defaults: embedding / attention / hidden 512, output_len 10, teacher forcing off
    if task == "2-1":
        return with_epoch_end(m21.CrossAttnRNN)(attention_dim=512, embedding_dim=512, hidden_dim=512, use_img=use_img, out_len=10)
    if task == "2-10":
        return with_epoch_end(m210.CrossAttnRNN)(attention_dim=512, embedding_dim=512, hidden_dim=512, use_img=use_img, out_len=10,
                                                 use_teacher_forcing=False, teacher_forcing_ratio=0.5)
    return with_epoch_end(md.CrossAttnRNN)(attention_dim=512, embedding_dim=512, hidden_dim=512, num_trends=3, cat_dict=cat,
                                           col_dict=col, fab_dict=fab, store_num=125, use_img=use_img, use_att=1, use_date=1,
                                           use_trends=1, out_len=12, use_teacher_forcing=False, teacher_forcing_ratio=0.5)


def build_gtm(cat, col, fab, text_feats: dict):
    sys.path.insert(0, str(GTMDIR))
    import models.GTM as G

    inv = ({v: k for k, v in cat.items()}, {v: k for k, v in col.items()}, {v: k for k, v in fab.items()})

    class CachedText(nn.Module):  # official TextEmbedder with the frozen BERT features precomputed (exact)
        def __init__(self, embedding_dim, cat_dict, col_dict, fab_dict, gpu_num):
            super().__init__()
            self.fc, self.dropout = nn.Linear(768, embedding_dim), nn.Dropout(0.1)

        def forward(self, category, color, fabric):
            keys = [inv[1][c] + " " + inv[2][f] + " " + inv[0][k] for k, c, f in zip(category.tolist(), color.tolist(), fabric.tolist())]
            e = torch.stack([text_feats[k] for k in keys]).to(category.device)
            return self.dropout(self.fc(e))

    class CachedImg(nn.Module):  # official ImageEmbedder: frozen ResNet-50, precomputed (exact)
        def forward(self, images):
            return images.float()

    G.TextEmbedder, G.ImageEmbedder = CachedText, CachedImg

    class GTMStore(G.GTM):
        """Official GTM + a store embedding added to its temporal ("dummy") features, so that it forecasts per shop."""

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.store_embedding = nn.Embedding(126, self.embedding_dim)

        def forward(self, category, color, fabric, stores, temporal_features, gtrends, images):
            img_encoding = self.image_encoder(images)
            dummy_encoding = self.dummy_encoder(temporal_features) + self.store_embedding(stores)
            text_encoding = self.text_encoder(category, color, fabric)
            gtrend_encoding = self.gtrend_encoder(gtrends)
            static = self.static_feature_encoder(img_encoding, text_encoding, dummy_encoding)
            # official path: nn.TransformerDecoder with one official TransformerDecoderLayer (returns (out, attn)).
            # Current PyTorch passes *_is_causal kwargs that layer does not accept, so the single layer is called directly.
            decoder_out, attn = self.decoder.layers[0](static.unsqueeze(0), gtrend_encoding)
            return self.decoder_fc(decoder_out).view(-1, self.output_len), attn

        def training_step(self, batch, batch_idx):
            (ts, cat_, col_, fab_, stores, temporal, gtr), images = batch
            loss = F.mse_loss(ts, self.forward(cat_, col_, fab_, stores, temporal, gtr, images)[0].squeeze())
            self.log("train_loss", loss)
            return loss

        def validation_step(self, batch, batch_idx):
            (ts, cat_, col_, fab_, stores, temporal, gtr), images = batch
            return ts, self.forward(cat_, col_, fab_, stores, temporal, gtr, images)[0]

        def on_validation_epoch_start(self):
            pass

    # official train.py defaults: embedding 32, hidden 64, 4 heads, 1 layer, use_text / use_img / encoder mask on
    return with_epoch_end(GTMStore)(embedding_dim=32, hidden_dim=64, output_dim=12, num_heads=4, num_layers=1, use_text=1, use_img=1,
                                    cat_dict=cat, col_dict=col, fab_dict=fab, trend_len=52, num_trends=3, gpu_num=0,
                                    use_encoder_mask=1, autoregressive=0)


@torch.inference_mode()
def bert_features(cat, col, fab) -> dict:
    """Exactly the official TextEmbedder features: HF feature-extraction pipeline, tokens [1:-1] averaged."""
    out = CACHE / "bert_text.pt"
    if out.exists():
        return torch.load(out, weights_only=False)
    from transformers import pipeline

    pipe = pipeline("feature-extraction", model=str(ROOT / ".hf_models" / "bert-base-uncased"), device=0)
    keys = [f"{c} {f} {k}" for k in cat for c in col for f in fab]
    feats = {}
    for s in range(0, len(keys), 256):
        chunk = keys[s : s + 256]
        for k, x in zip(chunk, pipe(chunk)):
            feats[k] = torch.FloatTensor(x[0][1:-1]).mean(axis=0)
    torch.save(feats, out)
    return feats


# ---------------------------------------------------------------------------------------------------- run
def calc_error_metrics(gt, forecasts):
    sys.path.insert(0, str(V2))
    from utils import calc_error_metrics as official  # sklearn MAE and 100 * sum|e| / sum(gt), rounded

    return official(gt, forecasts)


MEMORY_PATHS = {0: "checkpointing layer3", 1: "checkpointing layer3 + layer4",
                2: "checkpointing layer3 + layer4, micro-batches of 64 with 2-step gradient accumulation (effective batch 128)"}


def run(args, seed: int, memory_level: int = 0) -> dict:
    src = Path(args.src)
    demand = args.task == "demand"
    demand_finetune = args.model == "crossattn" and demand and (args.full_budget or args.official_selection)  # amendment (h)
    kind = "inception7a" if (args.model == "crossattn" and not demand) else (("resnet101_l2" if demand_finetune else "resnet101") if args.model == "crossattn" else "resnet50")
    trn, va, te = split_frames(src, args.official_selection)
    paths = pd.concat([trn.image_path, va.image_path, te.image_path]).tolist()
    feats = cached_features(kind, paths, src)
    sel = "official" if args.official_selection else "val"
    dtr, cat, col, fab = official_dataset(trn, src, demand, feats, f"{sel}_train_{'demand' if demand else 'stfore'}")
    dva, *_ = official_dataset(va, src, demand, feats, f"{sel}_val_{'demand' if demand else 'stfore'}")
    dte, *_ = official_dataset(te, src, demand, feats, f"test_{'demand' if demand else 'stfore'}")
    pl.seed_everything(seed)
    model = build_gtm(cat, col, fab, bert_features(cat, col, fab)) if args.model == "gtm" else build_crossattn(args.task, args.use_img, cat, col, fab, demand_finetune, memory_level)
    monitor = "val_mae" if args.model == "gtm" else "val_wWAPE"
    ck_dir = CACHE / "ckpt" / f"{args.model}_{args.task}_{args.use_img}_{sel}_{seed}"
    import shutil

    shutil.rmtree(ck_dir, ignore_errors=True)  # no stale checkpoint from an earlier or failed (out-of-memory) attempt
    ck = pl.callbacks.ModelCheckpoint(dirpath=str(ck_dir), monitor=monitor, mode="min", save_top_k=1)
    epochs = args.epochs or (200 if args.model == "gtm" else 30)  # official defaults (GTM train.py; visuelle2 train_dl.py)
    callbacks = [ck]
    early = not (args.official_selection or args.full_budget)
    if early:  # pre-registration amendment (e): stop after 5 validation checks without improvement (secondary since (h))
        callbacks.append(pl.callbacks.EarlyStopping(monitor=monitor, mode="min", patience=5))
    micro = memory_level >= 2  # amendment (h): batch-norm batch statistics then come from 64-sample micro-batches
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=epochs, check_val_every_n_epoch=5 if args.model == "gtm" else 1,
                         accumulate_grad_batches=2 if micro else 1,
                         callbacks=callbacks, logger=False, enable_progress_bar=False, enable_model_summary=False)
    dl = lambda d, sh, bs=128: torch.utils.data.DataLoader(d, batch_size=bs, shuffle=sh, num_workers=0)  # noqa: E731
    t0 = time.time()
    trainer.fit(model, dl(dtr, True, 64 if micro else 128), dl(dva, False))
    best = model
    best.load_state_dict(torch.load(ck.best_model_path, weights_only=False)["state_dict"])  # best validation checkpoint
    shutil.rmtree(ck_dir, ignore_errors=True)  # the best weights are in memory
    best.to("cuda").eval()
    gts, preds = [], []
    with torch.no_grad():
        for batch in dl(dte, False):
            batch = [[t.to("cuda") for t in batch[0]], batch[1].to("cuda")]
            if demand:
                (ts, c_, co, fa, st, tf, gt_), img = batch
                p = best(c_, co, fa, st, tf, gt_, img)[0] if args.model == "gtm" else best(ts, c_, co, fa, st, tf, gt_, img)[0]
                gts.append(ts.cpu()); preds.append(p.reshape(len(ts), -1).cpu())
            else:
                (X, y, *_), img = batch
                gts.append(y.reshape(len(y), -1).cpu()); preds.append(best(X, y, img)[0].reshape(len(y), -1).cpu())
    ns = float(np.load(src / "stfore_sales_norm_scalar.npy"))
    g, p = torch.cat(gts).numpy() * ns, torch.cat(preds).numpy() * ns
    mae, wape = calc_error_metrics(g, p)
    new = ~te.external_code.astype(str).isin(set(trn.external_code.astype(str)) | set(va.external_code.astype(str))).to_numpy()
    w = lambda s: float(100 * np.abs(g[s] - p[s]).sum() / g[s].sum())  # noqa: E731
    return {"model": args.label, "task": args.task, "seed": seed, "selection": sel, "WAPE_official_fn": float(wape), "MAE_official_fn": float(mae),
            "all": {"WAPE": w(np.ones(len(g), bool)), "MAE": float(np.abs(g - p).mean())},
            "new_product": {"WAPE": w(new), "MAE": float(np.abs(g[new] - p[new]).mean())},
            "seen_product": {"WAPE": w(~new), "MAE": float(np.abs(g[~new] - p[~new]).mean())},
            "best_val": float(ck.best_model_score), "epochs": epochs, "epochs_run": int(trainer.current_epoch), "train_seconds": round(time.time() - t0, 1),
            "per_pair": {"abs_err": np.abs(g - p).sum(1).round(4).tolist(), "y": g.sum(1).round(4).tolist(),
                         "product": te.external_code.astype(str).tolist(), "new_product": new.tolist()},
            "spec": {"published_code": True, "model": args.model, "use_img": args.use_img, "full_budget": not early, "early_stopping": early,
                     "image_trunk": ("ResNet-101 layer3-4 fine-tuned (released)" if demand_finetune else "frozen, cached") if args.model == "crossattn" else "frozen, cached",
                     "memory_path": MEMORY_PATHS[memory_level] if demand_finetune else None}}


MEMORY_LEVEL = {v: k for k, v in MEMORY_PATHS.items()}


def _is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower() or "CUBLAS_STATUS_ALLOC_FAILED" in str(e)


def run_with_fallback(args, seed: int, level: int, finetune: bool, runner=None) -> dict:
    """Owner's rule (amendment (h)): for the fine-tuned demand encoder, a CUDA out-of-memory failure restarts the seed
    from scratch with layer4 checkpointing as well, then with 64-sample micro-batches and 2-step accumulation."""
    import gc

    runner = runner or run
    while True:
        oom = False
        try:
            res = runner(args, seed, level)
            res["memory_level"] = level
            return res
        except Exception as e:  # noqa: BLE001
            if not (finetune and _is_oom(e) and level < 2):
                raise
            print(f"[{args.task} seed {seed}] out of memory with {MEMORY_PATHS[level]}; retrying with {MEMORY_PATHS[level + 1]}", flush=True)
            oom = True
        if oom:  # outside the except block, so the failed run's frames (model, optimiser state) can be freed
            level += 1
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["crossattn", "gtm"], required=True)
    ap.add_argument("--task", choices=["2-1", "2-10", "demand"], required=True)
    ap.add_argument("--use-img", type=int, default=1)
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--epochs", type=int, default=0, help="0 = official default")
    ap.add_argument("--official-selection", action="store_true", help="diagnostic: official split, test set as validation, seed 21")
    ap.add_argument("--full-budget", action="store_true", help="amendment (h): the released epoch budget, no early stopping, validation-only checkpoint selection")
    ap.add_argument("--src", default="visuelle2")
    ap.add_argument("--out", default="results_visuelle2_official")
    args = ap.parse_args()
    base = "GTM-Transformer (official code, + store)" if args.model == "gtm" else ("CrossAttnRNN w/ image (official code)" if args.use_img else "CrossAttnRNN (official code)")
    args.label = base + (" [official test-set selection, diagnostic]" if args.official_selection else "")
    seeds = [21] if args.official_selection else [int(s) for s in args.seeds.split(",")]
    part = Path(args.out) / ("diagnostics" if args.official_selection else "partials")
    part.mkdir(parents=True, exist_ok=True)
    from benchmark import slug

    finetune = args.model == "crossattn" and args.task == "demand" and (args.full_budget or args.official_selection)
    level = 0
    for seed in seeds:
        path = part / f"{args.task}__{seed}__{slug(args.label)}.json"
        if path.exists():
            continue
        res = run_with_fallback(args, seed, level, finetune)
        level = MEMORY_LEVEL[res["spec"]["memory_path"]] if finetune else 0  # later seeds start at the level that fit
        path.write_text(json.dumps(res))
        print(f"[{args.task} seed {seed}] {args.label:52s} WAPE {res['all']['WAPE']:6.2f} (official fn {res['WAPE_official_fn']}) "
              f"MAE {res['all']['MAE']:.3f} | new-product {res['new_product']['WAPE']:6.2f} | best val {res['best_val']:.3f} | {res['train_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
