"""CPU checks for the amendment (h) demand encoder (cached ResNet-101 layer2 + trainable layer3-4).

    CUDA_VISIBLE_DEVICES= python examples/test_demand_encoder.py

(A) eval mode: our CachedResNetL2 on cached layer2 features vs the released ImageEncoder
    (third_party/visuelle2.0-code-main/models/CrossAttnRNNDemand.py) on the raw 256-px images, same weights;
(B) train mode: gradient-checkpointed vs plain layer3: outputs, gradients and batch-norm running statistics after
    one step must agree (the recomputation must not update running averages a second time);
(C) one training_step of the released demand CrossAttnRNN with this encoder on 8 real rows: gradients reach
    layer3 and layer4.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import visuelle2_published as P

torch.manual_seed(0)
src = Path("visuelle2")
trn, va, te = P.split_frames(src, official=False)
paths = trn.image_path.drop_duplicates().iloc[:8].tolist()

# (A)
sys.path.insert(0, str(P.V2))
import models.CrossAttnRNNDemand as md

released = md.ImageEncoder(512).eval()
ours = P.CachedResNetL2(512).eval()
ours.layer3.load_state_dict(released.cnn[6].state_dict())
ours.layer4.load_state_dict(released.cnn[7].state_dict())
ours.fc.load_state_dict(released.fc.state_dict())
raw = torch.stack([P._img_tensor(src / "images" / p, 256)[0] for p in paths])
bank = P.cached_features("resnet101_l2", paths, src, device="cpu", save=False)
x_l2 = bank.x[[bank.pos[p] for p in paths]]
with torch.no_grad():
    a, b = released(raw), ours(x_l2)
    b32 = ours(torch.nn.Sequential(*list(released.cnn.children())[:6])(raw))
print(f"(A) released vs ours, fp16 cache: max |diff| {(a - b).abs().max():.2e} (max |out| {a.abs().max():.2e}); "
      f"fp32 layer2 (no cache rounding): max |diff| {(a - b32).abs().max():.2e}; shapes {tuple(a.shape)} {tuple(b.shape)}")

# (B)
m1 = P.CachedResNetL2(512, checkpoint_layer3=True).train()
m2 = copy.deepcopy(m1)
m2.checkpoint_layer3 = False
x = x_l2.float()
outs = []
for m in (m1, m2):
    torch.manual_seed(1)
    y = m(x)
    (y ** 2).mean().backward()
    outs.append(y.detach())
g = max((p1.grad - p2.grad).abs().max().item() for p1, p2 in zip(m1.parameters(), m2.parameters()) if p1.grad is not None)
bn1 = [mm for mm in m1.modules() if isinstance(mm, torch.nn.BatchNorm2d)]
bn2 = [mm for mm in m2.modules() if isinstance(mm, torch.nn.BatchNorm2d)]
rs = max(max((u.running_mean - v.running_mean).abs().max().item(), (u.running_var - v.running_var).abs().max().item()) for u, v in zip(bn1, bn2))
fresh = P.CachedResNetL2(512)
moved = max((u.running_mean - v.running_mean).abs().max().item() for u, v in zip(bn1, [mm for mm in fresh.modules() if isinstance(mm, torch.nn.BatchNorm2d)]))
print(f"(B) checkpointed vs plain: max |out diff| {(outs[0] - outs[1]).abs().max():.2e} | max |grad diff| {g:.2e} | "
      f"max |running-stat diff| {rs:.2e} (running stats moved by {moved:.2e} from init, so the update happened once)")

# (C)
cat, col, fab = (torch.load(src / f"{n}_labels.pt", weights_only=False) for n in ("category", "color", "fabric"))
sub = trn[trn.image_path.isin(paths)].head(8).reset_index(drop=True)
ds, cat, col, fab = P.official_dataset(sub, src, True, bank, "unit_test_demand")
model = P.build_crossattn("demand", 1, cat, col, fab, demand_finetune=True).train()
batch = torch.utils.data.default_collate([ds[i] for i in range(len(ds))])
loss = model.training_step(batch, 0)
loss.backward()
enc = model.image_encoder
gl3 = sum(p.grad.abs().sum().item() for p in enc.layer3.parameters() if p.grad is not None)
gl4 = sum(p.grad.abs().sum().item() for p in enc.layer4.parameters() if p.grad is not None)
n_train = sum(p.numel() for p in enc.parameters() if p.requires_grad)
print(f"(C) training_step loss {loss.item():.4f} | sum |grad| layer3 {gl3:.3e}, layer4 {gl4:.3e} | trainable encoder params {n_train:,}")
(P.CACHE / "unit_test_demand.pt").unlink(missing_ok=True)

# (D) layer3 + layer4 checkpointing (first out-of-memory fallback) is exact as well
m3 = P.CachedResNetL2(512, checkpoint_layer3=True, checkpoint_layer4=True).train()
m4 = copy.deepcopy(m3)
m4.checkpoint_layer3 = m4.checkpoint_layer4 = False
outs = []
for m in (m3, m4):
    torch.manual_seed(1)
    y = m(x)
    (y ** 2).mean().backward()
    outs.append(y.detach())
g = max((p1.grad - p2.grad).abs().max().item() for p1, p2 in zip(m3.parameters(), m4.parameters()) if p1.grad is not None)
rs = max((u.running_var - v.running_var).abs().max().item() for u, v in zip(
    [mm for mm in m3.modules() if isinstance(mm, torch.nn.BatchNorm2d)], [mm for mm in m4.modules() if isinstance(mm, torch.nn.BatchNorm2d)]))
print(f"(D) layer3+layer4 checkpointed vs plain: max |out diff| {(outs[0] - outs[1]).abs().max():.2e} | max |grad diff| {g:.2e} | max |running-stat diff| {rs:.2e}")

# (E) out-of-memory fallback control flow (fake runner, no GPU)
import types

calls = []


def fake(args, seed, level):
    calls.append(level)
    if level < 2:
        raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
    return {"spec": {"memory_path": P.MEMORY_PATHS[level]}}


a = types.SimpleNamespace(task="demand")
r = P.run_with_fallback(a, 1, 0, True, runner=fake)
ok_path = calls == [0, 1, 2] and r["memory_level"] == 2
try:
    P.run_with_fallback(a, 1, 0, False, runner=fake)  # other models: an out-of-memory error is not retried
    ok_other = False
except torch.cuda.OutOfMemoryError:
    ok_other = True
try:
    P.run_with_fallback(a, 1, 0, True, runner=lambda *_: (_ for _ in ()).throw(ValueError("not a memory error")))
    ok_err = False
except ValueError:
    ok_err = True
print(f"(E) fallback levels tried {calls} -> final '{r['spec']['memory_path']}' ({ok_path}); non-finetune not retried ({ok_other}); other errors re-raised ({ok_err})")
