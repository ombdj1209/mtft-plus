"""Separate-encoder product features for the MTFT-Picnic baseline (HANDOFF §3.9).

    python examples/unimodal_features.py --data-dir data/visuelle2_converted

Writes <data-dir>/unimodal_image.npy (A, 2048; ResNet-152 ImageNet, global-average-pooled) and
<data-dir>/unimodal_text.npy (A, 768; DistilBERT, mean-pooled last hidden state) in catalog.csv row order,
which examples/prepare_panel.py picks up. The two encoders are trained independently, so their spaces are
not aligned; MTFT-Picnic then compresses each to 10 dims with PCA fitted on warm articles (benchmark.py).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class Images(Dataset):
    """ConvNext-style preprocessing from the model's preprocessor_config.json (what the HF image processor does, without
    its torchvision dependency): bicubic resize of the short edge to size / crop_pct, centre crop, ImageNet normalise."""

    def __init__(self, paths, cfg: dict) -> None:
        self.paths = list(paths)
        self.size = int(cfg.get("size", 224))
        self.resize = int(round(self.size / float(cfg.get("crop_pct", 0.875))))
        self.resample = int(cfg.get("resample", 3))
        self.mean = np.asarray(cfg.get("image_mean", [0.485, 0.456, 0.406]), np.float32)
        self.std = np.asarray(cfg.get("image_std", [0.229, 0.224, 0.225]), np.float32)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
            ok = True
        except OSError:  # unreadable file: zero feature vector (missing image), flagged below
            img, ok = Image.new("RGB", (self.size, self.size), (255, 255, 255)), False
        w, h = img.size
        s = self.resize / min(w, h)
        img = img.resize((max(self.size, round(w * s)), max(self.size, round(h * s))), resample=self.resample)
        w, h = img.size
        l, t = (w - self.size) // 2, (h - self.size) // 2
        x = np.asarray(img.crop((l, t, l + self.size, t + self.size)), np.float32) / 255.0
        x = (x - self.mean) / self.std
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), ok


@torch.inference_mode()
def main() -> None:
    import json

    from transformers import AutoModel, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--image-model", default="microsoft/resnet-152")
    ap.add_argument("--text-model", default="distilbert-base-uncased")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    d = Path(args.data_dir)
    cat = pd.read_csv(d / "catalog.csv")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pcfg_path = Path(args.image_model) / "preprocessor_config.json"
    if pcfg_path.exists():
        pcfg = json.loads(pcfg_path.read_text())
    else:
        from huggingface_hub import hf_hub_download

        pcfg = json.loads(Path(hf_hub_download(args.image_model, "preprocessor_config.json")).read_text())
    net = AutoModel.from_pretrained(args.image_model).eval().to(dev)
    dl = DataLoader(Images(cat.image_path, pcfg), batch_size=args.batch_size, num_workers=args.workers)
    feats, oks = [], []
    for i, (px, ok) in enumerate(dl):
        feats.append(net(pixel_values=px.to(dev)).pooler_output.flatten(1).float().cpu())
        oks.append(ok)
        if i % 20 == 0:
            print(f"images {min((i + 1) * args.batch_size, len(cat))}/{len(cat)}", flush=True)
    img = torch.cat(feats).numpy()
    bad = ~torch.cat(oks).numpy().astype(bool)
    img[bad] = 0.0
    if bad.any():
        print(f"warning: {int(bad.sum())} unreadable image(s) -> zero features: {cat.article_id[bad].tolist()}")
    np.save(d / "unimodal_image.npy", img)

    tok = AutoTokenizer.from_pretrained(args.text_model)
    enc = AutoModel.from_pretrained(args.text_model).eval().to(dev)
    txt = []
    texts = cat.text.fillna("").tolist()
    for s in range(0, len(texts), 256):
        b = tok(texts[s : s + 256], padding=True, truncation=True, max_length=32, return_tensors="pt").to(dev)
        h = enc(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        txt.append(((h * m).sum(1) / m.sum(1)).float().cpu())
    txt = torch.cat(txt).numpy()
    np.save(d / "unimodal_text.npy", txt)
    print(f"unimodal_image {img.shape}, unimodal_text {txt.shape} -> {d}")


if __name__ == "__main__":
    main()
