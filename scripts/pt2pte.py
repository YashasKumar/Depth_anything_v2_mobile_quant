"""
Export Depth Anything V2 Small (fixed 392x518) from the .pth to an ExecuTorch .pte and verify it.

Same model setup as pt2onnx.py: the DINOv2 positional embedding is resized once for the fixed
input size and stored as a constant (no bicubic resize at run time). GELU stays exact (erf);
ExecuTorch runs the PyTorch graph directly, so there is no onnx2tf / Flex-op issue.

The graph is lowered to the XNNPACK backend (CPU, FP32): every op XNNPACK supports runs inside
the delegate; anything else falls back to ExecuTorch's portable kernels.

Env:   conda env "executorch" (executorch==1.5.1 and the torch it pins)
Run (from the repo root, after scripts/phase1_baseline.py):
       python scripts/pt2pte.py                 # export + verify on 20 reference images
       python scripts/pt2pte.py --num_check 0   # verify on all 654
"""

import argparse
import glob
import os
import sys
import types

import numpy as np
import torch

try:
    import cv2  # noqa: F401
except ImportError:  # the repo only uses cv2 for image preprocessing, never in forward()
    sys.modules["cv2"] = types.SimpleNamespace(INTER_CUBIC=2, INTER_AREA=3, INTER_NEAREST=0)

sys.path.insert(0, "Depth-Anything-V2")
from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402

IN_H, IN_W = 392, 518
PATCH = 14


def build_model(ckpt):
    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    vit = model.pretrained
    n_tokens = (IN_H // PATCH) * (IN_W // PATCH) + 1
    with torch.no_grad():
        fixed = vit.interpolate_pos_encoding(torch.zeros(1, n_tokens, vit.embed_dim), IN_H, IN_W).clone()
    vit.register_buffer("baked_pos_embed", fixed)
    vit.interpolate_pos_encoding = lambda x, w, h: vit.baked_pos_embed.to(x.dtype)
    return model


def rel_err(a, b):
    return np.abs(a - b).mean() / (np.abs(b).mean() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="depth_anything_v2_vits.pth")
    ap.add_argument("--ref_dir", default="results/fp32_392x518")
    ap.add_argument("--out", default="models/depth_anything_v2_vits_392x518_xnnpack.pte")
    ap.add_argument("--num_check", type=int, default=20, help="0 = all reference images")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower
    from executorch.runtime import Runtime

    model = build_model(args.ckpt)
    example = (torch.zeros(1, 3, IN_H, IN_W),)

    print("Exporting (torch.export)...")
    with torch.no_grad():
        ep = torch.export.export(model, example)
    print("Lowering to XNNPACK...")
    program = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()]).to_executorch()
    with open(args.out, "wb") as f:
        f.write(program.buffer)
    print(f"Wrote {args.out}  ({os.path.getsize(args.out) / 1e6:.1f} MB)")

    # --- what got delegated
    gm = program.exported_program().graph_module
    calls = [n for n in gm.graph.nodes if n.op == "call_function"]
    delegated = sum("executorch_call_delegate" in str(n.target) for n in calls)
    other = sorted({str(n.target) for n in calls
                    if "executorch_call_delegate" not in str(n.target) and "getitem" not in str(n.target)})
    print(f"XNNPACK delegate calls: {delegated}; ops left to portable kernels: {other or 'none'}")

    # --- verify with the ExecuTorch runtime against the PyTorch FP32 reference (phase1_baseline.py)
    inputs = sorted(glob.glob(os.path.join(args.ref_dir, "inputs", "*.npy")))
    if args.num_check > 0:
        inputs = inputs[: args.num_check]
    if not inputs:
        print(f"No reference inputs in {args.ref_dir}/inputs - skipping verification.")
        return
    method = Runtime.get().load_program(args.out).load_method("forward")
    rels, maxes = [], []
    for k, p in enumerate(inputs):
        out = method.execute([torch.from_numpy(np.load(p))])[0].numpy()[0]
        ref = np.load(p.replace("inputs", "raw"))
        rels.append(rel_err(out, ref))
        maxes.append(np.abs(out - ref).max())
        if (k + 1) % 100 == 0 or k + 1 == len(inputs):
            print(f"  [{k + 1}/{len(inputs)}] worst rel error so far: {max(rels):.2e}")
    rels = np.array(rels)
    print(f"\nExecuTorch (host, XNNPACK) vs PyTorch FP32 reference, {len(inputs)} images:")
    print(f"  max abs diff:  {max(maxes):.3e}")
    print(f"  rel err mean:  {rels.mean():.2e}   median: {np.median(rels):.2e}   worst: {rels.max():.2e}")


if __name__ == "__main__":
    main()
