#!/usr/bin/env python3
"""Score TFLite models on the NYU test images on the host, with the same metrics as the device runs
(and phase1_baseline.py).

Inputs come from <test>_nhwc.npy / <test>_gt.npy, made by prepare_test.py (default test50 = the first 50
test images, preprocessed exactly as on device; prepare_test.py -n 654 + --test test654 = full split). For each model: AbsRel, delta1-3, RMSE vs ground truth (LSQ scale+shift on disparity,
Eigen crop, 1e-3..10 m), and output agreement vs the FP32 model (mean|a-b| / mean|b| per image).

Usage:  python eval_tflite_nyu.py model1.tflite [model2.tflite ...] [--ref fp32.tflite] [--test test50] [-n N]
"""
import argparse
import glob
import os
import time

import numpy as np
import tensorflow as tf

GT_H, GT_W = 480, 640
CROP = (45, 470, 41, 600)
MIN_D, MAX_D = 1e-3, 10.0


def align_and_score(raw, gt):
    disp = tf.image.resize(raw[None, :, :, None], (GT_H, GT_W), method="bilinear").numpy()[0, :, :, 0]
    r0, r1, c0, c1 = CROP
    d, g = disp[r0:r1, c0:c1].astype(np.float64), gt[r0:r1, c0:c1].astype(np.float64)
    v = (g > MIN_D) & (g < MAX_D)
    p, gd = d[v], 1.0 / g[v]
    A = np.stack([p, np.ones_like(p)], 1)
    s, t = np.linalg.lstsq(A, gd, rcond=None)[0]
    aligned = np.maximum(s * p + t, 1.0 / MAX_D)
    pd = np.clip(1.0 / aligned, MIN_D, MAX_D)
    gv = g[v]
    ratio = np.maximum(pd / gv, gv / pd)
    return dict(absrel=np.mean(np.abs(pd - gv) / gv), d1=np.mean(ratio < 1.25),
                d2=np.mean(ratio < 1.25 ** 2), d3=np.mean(ratio < 1.25 ** 3),
                rmse=np.sqrt(np.mean((pd - gv) ** 2)))


def run_model(path, X, threads):
    it = tf.lite.Interpreter(model_path=path, num_threads=threads)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    outs, t0 = [], time.time()
    for x in X:
        xi = x[None]
        if inp["dtype"] != np.float32:  # integer I/O
            sc, zp = inp["quantization"]
            xi = np.clip(np.round(xi / sc + zp), np.iinfo(inp["dtype"]).min, np.iinfo(inp["dtype"]).max).astype(inp["dtype"])
        it.set_tensor(inp["index"], xi)
        it.invoke()
        o = it.get_tensor(out["index"])
        if out["dtype"] != np.float32:
            sc, zp = out["quantization"]
            o = (o.astype(np.float32) - zp) * sc
        outs.append(o.reshape(392, 518).astype(np.float32))
    return np.stack(outs), (time.time() - t0) / len(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--ref", default=(glob.glob("saved_model/*_float32.tflite") or [""])[0],
                    help="FP32 reference (default: the onnx2tf FP32 model in saved_model/)")
    ap.add_argument("--test", default="test50", help="prefix of the prepare_test.py files")
    ap.add_argument("-n", type=int, default=0, help="first n images (0 = all in the file)")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--cache", default="host_outputs")
    args = ap.parse_args()

    X = np.load(f"{args.test}_nhwc.npy", mmap_mode="r")
    args.n = args.n or len(X)
    X = X[: args.n]
    G = np.load(f"{args.test}_gt.npy", mmap_mode="r")[: args.n]
    os.makedirs(args.cache, exist_ok=True)

    def outputs(path):
        c = os.path.join(args.cache, os.path.basename(path) + f".n{args.n}.npy")
        if os.path.isfile(c) and os.path.getmtime(c) > os.path.getmtime(path):
            return np.load(c), float("nan")
        o, sec = run_model(path, X, args.threads)
        np.save(c, o)
        return o, sec

    ref, _ = outputs(args.ref)
    print(f"{'model':58s} {'MB':>6s} {'AbsRel':>7s} {'d1':>7s} {'d2':>7s} {'d3':>7s} {'RMSE':>6s}"
          f" {'relErr':>8s} {'worst':>8s} {'s/img':>6s}")
    for path in [args.ref] + [m for m in args.models if m != args.ref]:
        o, sec = outputs(path)
        ms = [align_and_score(o[i], G[i]) for i in range(len(o))]
        rel = [np.mean(np.abs(o[i] - ref[i])) / (np.mean(np.abs(ref[i])) + 1e-8) for i in range(len(o))]
        agg = {k: np.mean([m[k] for m in ms]) for k in ms[0]}
        print(f"{os.path.basename(path)[:58]:58s} {os.path.getsize(path) / 1e6:6.1f} {agg['absrel']:7.4f}"
              f" {agg['d1']:7.4f} {agg['d2']:7.4f} {agg['d3']:7.4f} {agg['rmse']:6.3f}"
              f" {np.mean(rel):8.2e} {np.max(rel):8.2e} {sec:6.2f}", flush=True)


if __name__ == "__main__":
    main()
