"""
Phase 2a - Export Depth Anything V2 Small to ONNX (fixed 392x518) and verify it.

Why the "pos-embedding bake" step:
  DINOv2 resizes its positional embeddings with a BICUBIC interpolation every
  forward pass. For a fixed input size that result never changes, and bicubic
  resize is badly supported on mobile runtimes / the Hexagon DSP. So we compute
  it once and store it as a constant. The output is mathematically identical,
  and we check that below.

GELU:
  --gelu exact (default)  erf GELU, as in PyTorch. Fine for ONNX Runtime, but onnx2tf turns Erf into a
                          Flex op that the standard TFLite runtime on Android cannot load.
  --gelu tanh             tanh-approximate GELU, only builtin TFLite ops after onnx2tf. Used for every
                          TFLite model here (output differs from exact by ~1e-3 relative, no metric change).

Install:   pip install onnx onnxruntime
Run (from the repo root, after scripts/phase1_baseline.py):
           python scripts/pt2onnx.py                      # -> models/..._exact_gelu.onnx
           python scripts/pt2onnx.py --gelu tanh          # -> models/..._tanh_gelu.onnx
"""

import argparse
import glob
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, "Depth-Anything-V2")
from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402

IN_H, IN_W = 392, 518
PATCH = 14


def bake_pos_embedding(model):
    vit = model.pretrained
    n_tokens = (IN_H // PATCH) * (IN_W // PATCH) + 1  # patches + cls token
    dummy = torch.zeros(1, n_tokens, vit.embed_dim)
    with torch.no_grad():
        fixed = vit.interpolate_pos_encoding(dummy, IN_H, IN_W).detach().clone()
    vit.register_buffer("baked_pos_embed", fixed)
    # Instance attribute overrides the class method; same call signature.
    vit.interpolate_pos_encoding = lambda x, w, h: vit.baked_pos_embed.to(x.dtype)
    return model


def compare(a, b):
    diff = np.abs(a - b)
    rel = diff.mean() / (np.abs(b).mean() + 1e-8)
    return diff.max(), rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="depth_anything_v2_vits.pth")
    ap.add_argument("--ref_dir", default="results/fp32_392x518")
    ap.add_argument("--gelu", choices=["exact", "tanh"], default="exact")
    ap.add_argument("--out", default=None, help="default: models/depth_anything_v2_vits_392x518_<gelu>_gelu.onnx")
    ap.add_argument("--num_check", type=int, default=0, help="0 = all reference images (654)")
    ap.add_argument("--opset", type=int, default=12)
    args = ap.parse_args()
    if args.out is None:
        args.out = f"models/depth_anything_v2_vits_392x518_{args.gelu}_gelu.onnx"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()
    model = bake_pos_embedding(model)
    if args.gelu == "tanh":
        n = 0
        for m in model.modules():
            if isinstance(m, torch.nn.GELU):
                m.approximate = "tanh"
                n += 1
        print(f"GELU -> tanh approximation in {n} modules")

    inputs = sorted(glob.glob(os.path.join(args.ref_dir, "inputs", "*.npy")))
    if args.num_check > 0:
        inputs = inputs[: args.num_check]
    if not inputs:
        sys.exit(f"No reference inputs in {args.ref_dir}/inputs - run phase1_baseline.py first.")

    # --- Check 1: baked model (PyTorch, CPU) still matches the Phase 1 reference
    print("Check 1: baked PyTorch model vs FP32 reference")
    worst = 0.0
    for p in inputs[:5]:
        ref = np.load(p.replace("inputs", "raw"))
        with torch.no_grad():
            out = model(torch.from_numpy(np.load(p)))[0].numpy()
        mx, rel = compare(out, ref)
        worst = max(worst, rel)
    print(f"  worst mean relative error: {worst:.2e}  (expect ~1e-5 or smaller; GPU vs CPU math)\n")

    # --- Export
    dummy = torch.from_numpy(np.load(inputs[0]))
    export_kwargs = dict(
        input_names=["image"],
        output_names=["depth"],
        opset_version=args.opset,
        do_constant_folding=True,
    )
    try:
        # Newer PyTorch defaults to the dynamo exporter; the classic one is
        # more predictable for onnx2tf, so ask for it explicitly.
        torch.onnx.export(model, dummy, args.out, dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, args.out, **export_kwargs)
    print(f"Exported -> {args.out}  ({os.path.getsize(args.out) / 1e6:.1f} MB)")

    import onnx
    import onnxruntime as ort

    m = onnx.load(args.out)
    onnx.checker.check_model(m)
    ops = Counter(n.op_type for n in m.graph.node)
    print("Op types in graph:")
    for op, c in sorted(ops.items(), key=lambda kv: -kv[1]):
        print(f"  {op:<22}{c}")
    if "Resize" in ops:
        modes = Counter(
            next((a.s.decode() for a in n.attribute if a.name == "mode"), "nearest")
            for n in m.graph.node if n.op_type == "Resize"
        )
        print(f"  Resize modes: {dict(modes)}  (should be linear only - no cubic)")

    # --- Check 2: ONNX Runtime vs FP32 reference
    print("\nCheck 2: ONNX Runtime (CPU) vs FP32 reference")
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    maxes, rels = [], []
    for k, p in enumerate(inputs):
        out = sess.run(["depth"], {"image": np.load(p)})[0][0]
        mx, rel = compare(out, np.load(p.replace("inputs", "raw")))
        maxes.append(mx)
        rels.append(rel)
        if (k + 1) % 100 == 0 or k + 1 == len(inputs):
            print(f"  [{k + 1}/{len(inputs)}] worst rel error so far: {max(rels):.2e}")
    rels = np.array(rels)
    worst = int(np.argmax(rels))
    print(f"\n  images checked:        {len(inputs)}")
    print(f"  max abs diff (any img): {max(maxes):.4e}")
    print(f"  rel error  mean:        {rels.mean():.2e}")
    print(f"  rel error  median:      {np.median(rels):.2e}")
    print(f"  rel error  worst:       {rels.max():.2e}  ({os.path.basename(inputs[worst])})")
    print("  (all should be ~1e-5 or smaller)")


if __name__ == "__main__":
    main()