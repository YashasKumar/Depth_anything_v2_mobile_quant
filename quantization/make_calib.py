#!/usr/bin/env python3
"""Calibration data for quantizing Depth Anything V2 (TFLite), from the NYU Depth V2 *train* split.

Never uses test images (the test split is what the models are scored on).
Preprocessing is identical to the device app (and phase1_baseline.py): RGB/255 -> INTER_CUBIC 518x392 ->
ImageNet mean/std, stored NHWC float32.

Writes (in the current folder):
  calib_train<N>_nhwc.npy                                      N x 392 x 518 x 3, normalized
  calibration_image_sample_data_20x128x128x3_float32.npy       onnx2tf's sample data (it would
                                                               otherwise download it), 0..1 RGB
Usage:  python make_calib.py --nyu_mat ../nyu_depth_v2_labeled.mat --splits ../splits.mat -n 200
"""
import argparse

import cv2
import h5py
import numpy as np
import scipy.io

IN_H, IN_W = 392, 518
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def preprocess(rgb_u8):
    x = rgb_u8.astype(np.float32) / 255.0
    x = cv2.resize(x, (IN_W, IN_H), interpolation=cv2.INTER_CUBIC)
    return (x - MEAN) / STD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nyu_mat", default="../nyu_depth_v2_labeled.mat")
    ap.add_argument("--splits", default="../splits.mat")
    ap.add_argument("-n", "--num", type=int, default=200)
    args = ap.parse_args()

    train_idx = scipy.io.loadmat(args.splits)["trainNdxs"].flatten().astype(int) - 1
    # spread evenly over the train split so many different scenes are covered
    pick = train_idx[np.linspace(0, len(train_idx) - 1, args.num).round().astype(int)]
    f = h5py.File(args.nyu_mat, "r")
    rgbs = [f["images"][i].transpose(2, 1, 0) for i in pick]  # 480x640x3 RGB uint8

    calib = np.stack([preprocess(r) for r in rgbs]).astype(np.float32)
    np.save(f"calib_train{args.num}_nhwc.npy", calib)
    sample = np.stack([cv2.resize(r, (128, 128), interpolation=cv2.INTER_AREA) for r in rgbs[:20]])
    np.save("calibration_image_sample_data_20x128x128x3_float32.npy", (sample / 255.0).astype(np.float32))
    print("calib", calib.shape, "train indices", pick[:5], "...")


if __name__ == "__main__":
    main()
