#!/usr/bin/env python3
"""Test inputs for the host evaluation and the demo video: the first N images of the official NYU Depth V2
test split (splits.mat testNdxs, the order phase1_baseline.py uses), raw sensor depth as ground truth.

Writes (in the current folder):
  test<N>_nhwc.npy   N x 392 x 518 x 3 float32, preprocessed like the device app:
                     RGB/255 -> INTER_CUBIC 518x392 -> ImageNet mean/std
  test<N>_gt.npy     N x 480 x 640 float32, metres (rawDepths; 0 = no measurement)
  test<N>_rgb.npy    N x 480 x 640 x 3 uint8 RGB (for the demo video)
Usage:  python prepare_test.py --nyu_mat ../nyu_depth_v2_labeled.mat --splits ../splits.mat [-n 50]
"""
import argparse

import h5py
import numpy as np
import scipy.io

from make_calib import preprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nyu_mat", default="../nyu_depth_v2_labeled.mat")
    ap.add_argument("--splits", default="../splits.mat")
    ap.add_argument("-n", "--num", type=int, default=50)
    args = ap.parse_args()

    test_idx = scipy.io.loadmat(args.splits)["testNdxs"].flatten().astype(int) - 1
    f = h5py.File(args.nyu_mat, "r")
    rgb = np.stack([f["images"][i].transpose(2, 1, 0) for i in test_idx[: args.num]])
    gt = np.stack([f["rawDepths"][i].T.astype(np.float32) for i in test_idx[: args.num]])
    np.save(f"test{args.num}_rgb.npy", rgb)
    np.save(f"test{args.num}_gt.npy", gt)
    np.save(f"test{args.num}_nhwc.npy", np.stack([preprocess(r) for r in rgb]).astype(np.float32))
    print("wrote", rgb.shape, gt.shape)


if __name__ == "__main__":
    main()
