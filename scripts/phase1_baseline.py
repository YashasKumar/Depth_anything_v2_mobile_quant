"""
Phase 1 - FP32 baseline for Depth Anything V2 Small (the "correct" reference).

What this does:
  1. Loads Depth Anything V2 Small in PyTorch (FP32).
  2. Runs it on the official NYU Depth V2 test split (654 images) at a FIXED
     input size (default 392x518, the size the phone model will use).
  3. Measures accuracy against ground-truth depth (AbsRel, delta1).
  4. Saves every raw model output as .npy -> these are the reference outputs
     we compare the phone's FP16 / INT8 outputs against in Phase 2 and 3.
  5. Saves a few side-by-side images so you can see the result.

Extra download for the test split (1-indexed list of the 654 test images):
  wget http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat
  pip install scipy

Runs:
  python phase1_baseline.py                          # phone resolution 392x518 (the reference)
  python phase1_baseline.py --input_h 518 --input_w 686 --no_save   # paper-like resolution
  python phase1_baseline.py --input_h 518 --input_w 686 --no_save --gt raw --out_dir results/fp32_518x686_rawgt

Outputs go to results/fp32_<H>x<W>/ so the two runs don't overwrite each other.
Paper numbers for V2-Small on NYU-D (zero-shot): AbsRel 0.053, delta1 0.973.
"""

import argparse
import os
import sys
import time

import cv2
import h5py
import numpy as np
import scipy.io
import torch

sys.path.insert(0, "Depth-Anything-V2")
from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402

# Network input size is set by --input_h/--input_w. Both must be multiples of 14
# (ViT patch size). 392x518 keeps NYU's 4:3 aspect ratio; the phone uses this shape.
IN_H, IN_W = 392, 518
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Standard NYU evaluation crop (Eigen crop) and valid depth range in meters.
CROP = (slice(45, 471), slice(41, 601))
MIN_DEPTH, MAX_DEPTH = 1e-3, 10.0


def preprocess(rgb_uint8):
    """HxWx3 RGB uint8 -> 1x3xIN_HxIN_W float32 (same steps the phone must do)."""
    img = rgb_uint8.astype(np.float32) / 255.0
    img = cv2.resize(img, (IN_W, IN_H), interpolation=cv2.INTER_CUBIC)
    img = (img - MEAN) / STD
    return img.transpose(2, 0, 1)[None].astype(np.float32)


def align_and_score(pred_disp, gt_depth):
    """
    The model predicts *relative* inverse depth (disparity), not meters.
    So we fit a scale and shift to match ground truth (standard practice),
    then compute the usual metrics.
    """
    gt = gt_depth[CROP]
    pred = pred_disp[CROP]
    valid = (gt > MIN_DEPTH) & (gt < MAX_DEPTH)

    gt_disp = 1.0 / gt[valid]
    p = pred[valid]
    A = np.stack([p, np.ones_like(p)], axis=1)
    s, t = np.linalg.lstsq(A, gt_disp, rcond=None)[0]

    aligned_disp = np.clip(s * p + t, 1.0 / MAX_DEPTH, None)
    pred_depth = np.clip(1.0 / aligned_disp, MIN_DEPTH, MAX_DEPTH)
    g = gt[valid]

    abs_rel = np.mean(np.abs(pred_depth - g) / g)
    ratio = np.maximum(pred_depth / g, g / pred_depth)
    delta1 = np.mean(ratio < 1.25)
    return abs_rel, delta1


def colorize(x):
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nyu_mat", default="nyu_depth_v2_labeled.mat")
    ap.add_argument("--ckpt", default="depth_anything_v2_vits.pth")
    ap.add_argument("--splits", default="splits.mat")
    ap.add_argument("--num_images", type=int, default=0, help="0 = all 654 test images")
    ap.add_argument("--input_h", type=int, default=392)
    ap.add_argument("--input_w", type=int, default=518)
    ap.add_argument("--out_dir", default=None, help="default: results/fp32_<H>x<W>")
    ap.add_argument("--num_vis", type=int, default=5)
    ap.add_argument("--gt", choices=["filled", "raw"], default="filled",
                    help="filled = in-painted 'depths' (holes filled by an algorithm); "
                         "raw = 'rawDepths' (real sensor readings only, holes = 0 and ignored)")
    ap.add_argument("--no_save", action="store_true",
                    help="don't save inputs/raw outputs (use for the paper-check run, saves ~3.7 GB)")
    args = ap.parse_args()

    global IN_H, IN_W
    IN_H, IN_W = args.input_h, args.input_w
    if IN_H % 14 or IN_W % 14:
        sys.exit("input_h and input_w must be multiples of 14")
    if args.out_dir is None:
        args.out_dir = f"results/fp32_{IN_H}x{IN_W}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # NVIDIA GPUs run convs (and sometimes matmuls) in TF32 by default, which is
    # less precise than real FP32. Turn it off so this is a true FP32 reference.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    os.makedirs(os.path.join(args.out_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "inputs"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "vis"), exist_ok=True)

    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model = model.to(device).eval()

    # Official test split: indices are 1-based (MATLAB), so subtract 1.
    test_idx = scipy.io.loadmat(args.splits)["testNdxs"].flatten().astype(int) - 1
    assert len(test_idx) == 654, f"expected 654 test images, got {len(test_idx)}"
    if args.num_images > 0:
        test_idx = test_idx[: args.num_images]
    np.save(os.path.join(args.out_dir, "test_indices.npy"), test_idx)

    f = h5py.File(args.nyu_mat, "r")
    n = len(test_idx)
    abs_rels, delta1s, times = [], [], []

    for i, idx in enumerate(test_idx):
        rgb = f["images"][idx].transpose(2, 1, 0)   # -> 480x640x3, RGB uint8
        gt_key = "depths" if args.gt == "filled" else "rawDepths"
        gt = f[gt_key][idx].T.astype(np.float32)    # -> 480x640, meters

        x = preprocess(rgb)
        if not args.no_save:
            np.save(os.path.join(args.out_dir, "inputs", f"{i:04d}.npy"), x)

        xt = torch.from_numpy(x).to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(xt)                        # 1 x IN_H x IN_W
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

        raw = out[0].cpu().numpy().astype(np.float32)
        if not args.no_save:
            np.save(os.path.join(args.out_dir, "raw", f"{i:04d}.npy"), raw)

        disp = cv2.resize(raw, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        ar, d1 = align_and_score(disp, gt)
        abs_rels.append(ar)
        delta1s.append(d1)

        if i < args.num_vis:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            pred_vis = colorize(disp)
            gt_vis = colorize(1.0 / np.clip(gt, 0.5, None))  # show GT as disparity too
            cv2.imwrite(os.path.join(args.out_dir, "vis", f"{i:04d}.png"),
                        np.hstack([bgr, pred_vis, gt_vis]))

        if (i + 1) % 50 == 0 or i + 1 == n:
            print(f"[{i + 1}/{n}] AbsRel={np.mean(abs_rels):.4f}  delta1={np.mean(delta1s):.4f}")

    t = np.array(times[5:] if len(times) > 10 else times) * 1000  # skip warm-up runs
    summary = (
        f"Images: {n} (official NYU test split)\n"
        f"Input size: {IN_H}x{IN_W}\n"
        f"Ground truth: {args.gt}\n"
        f"AbsRel (lower is better):  {np.mean(abs_rels):.4f}\n"
        f"delta1 (higher is better): {np.mean(delta1s):.4f}\n"
        f"Latency on {device}: mean {t.mean():.1f} ms, median {np.median(t):.1f} ms\n"
    )
    print("\n" + summary)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as fh:
        fh.write(summary)


if __name__ == "__main__":
    main()  