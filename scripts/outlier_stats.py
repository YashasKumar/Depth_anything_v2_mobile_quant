"""
Where are the activation outliers in Depth Anything V2 Small's DINOv2 encoder?

For each of the 12 ViT blocks, on a few NYU train images (quantization/calib_train200_nhwc.npy):
  residual stream (block output)
    token  outliers  max / median token L2-norm, and how many tokens are > 10x the median
                     (the high-norm "artifact" tokens described in "Vision Transformers Need Registers")
    channel outliers max|x| of the largest channel / median channel, and which channel
                     (the persistent outlier channels targeted by LLM.int8() / SmoothQuant / FQ-ViT)
  FC inputs (LayerNorm outputs: norm1 -> qkv, norm2 -> fc1)
    channel outliers as above

Result for ViT-S (8 images): no high-norm tokens (max/median token norm <= 3.5; in blocks 2-10 the largest is the CLS
token); one persistent outlier channel (278) in the residual stream from block 2 on, 8-18x the median
channel in blocks 2-10; at the FC inputs mostly 3-7x (exception: block 0's fc1 input, 14x). So int8 hurts the residual stream / LayerNorm, while the FC
layers tolerate it - the same split the quantization search found.

Run (from the repo root, after quantization/make_calib.py):
  python scripts/outlier_stats.py [--n 8]
"""

import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, "Depth-Anything-V2")
from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="depth_anything_v2_vits.pth")
    ap.add_argument("--calib", default="quantization/calib_train200_nhwc.npy")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    m = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    m.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    m.eval()
    calib = np.load(args.calib, mmap_mode="r")
    idx = np.linspace(3, len(calib) - 4, args.n).round().astype(int)
    X = torch.from_numpy(np.ascontiguousarray(calib[idx].transpose(0, 3, 1, 2)))

    f = {}
    def hook(key):
        return lambda mod, i, o: f.setdefault(key, []).append(o.detach())
    for k, b in enumerate(m.pretrained.blocks):
        b.register_forward_hook(hook((k, "out")))
        b.norm1.register_forward_hook(hook((k, "norm1")))
        b.norm2.register_forward_hook(hook((k, "norm2")))
    with torch.no_grad():
        for x in X:
            m(x[None])

    def channel(t):
        ch = t.abs().amax(dim=(0, 1))
        top = torch.topk(ch, 1)
        return (top.values[0] / ch.median()).item(), top.indices.item()

    print("            residual stream (block output)                        |  FC inputs: channel top/median (ch)")
    print("block | token max/median  #tokens>10x  max is CLS | channel top/median (ch) |   norm1->qkv     norm2->fc1")
    for k in range(12):
        t = torch.cat(f[(k, "out")])
        norms = t.norm(dim=-1)
        med = norms.median(1, keepdim=True).values
        tok = (norms.max(1).values / med[:, 0]).mean().item()
        nhi = (norms > 10 * med).sum(1).float().mean().item()
        cls = (norms.argmax(1) == 0).float().mean().item() * 100
        c, ci = channel(t)
        n1, n1i = channel(torch.cat(f[(k, "norm1")]))
        n2, n2i = channel(torch.cat(f[(k, "norm2")]))
        print(f"{k:5d} | {tok:16.1f} {nhi:12.1f} {cls:9.0f}% | {c:17.1f} ({ci:3d}) | {n1:8.1f} ({n1i:3d}) {n2:8.1f} ({n2i:3d})")


if __name__ == "__main__":
    main()
