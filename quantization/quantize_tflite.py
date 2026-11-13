#!/usr/bin/env python3
"""Post-training quantization variants of Depth Anything V2 Small (TFLite), from the onnx2tf SavedModel.

Variants (all keep float32 input/output, so the device code does not change):
  dynamic   int8 weights, float activations (dynamic-range quantization)
  int8      int8 weights (per-channel) + int8 activations, calibrated (full integer)
  int16x8   int8 weights + int16 activations, calibrated
  fp16 comes straight out of onnx2tf (saved_model/*_float16.tflite)

Mixed precision (the "which layers to quantize" step) is int8 with a denylist of ops/nodes kept in
float, via the QuantizationDebugger: --denylist_types and/or --denylist_nodes.

Calibration: calib_train200_nhwc.npy (NYU *train* split, see make_calib.py).
Usage:
  python quantize_tflite.py dynamic int8 int16x8
  python quantize_tflite.py mixed --denylist_types SOFTMAX,MEAN --name int8_noln
"""
import argparse

import numpy as np
import tensorflow as tf

SM = "saved_model"
BASE = "depth_anything_v2_vits_392x518"


def rep_dataset(n):
    calib = np.load("calib_train200_nhwc.npy", mmap_mode="r")
    idx = np.linspace(0, len(calib) - 1, n).round().astype(int)

    def gen():
        for i in idx:
            yield [np.asarray(calib[i : i + 1], np.float32)]

    return gen


def converter(kind, n_calib):
    c = tf.lite.TFLiteConverter.from_saved_model(SM)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    if kind == "dynamic":
        return c
    c.representative_dataset = rep_dataset(n_calib)
    if kind == "int16x8":
        c.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS]  # fall back to float for ops without int16 kernels
    else:
        c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="+", choices=["dynamic", "int8", "int16x8", "mixed"])
    ap.add_argument("--calib", type=int, default=200, help="calibration images (<=200)")
    ap.add_argument("--denylist_types", default="", help="mixed: comma-separated TFLite op types kept float")
    ap.add_argument("--denylist_nodes", default="", help="mixed: file with one tensor name per line kept float")
    ap.add_argument("--name", default="mixed")
    args = ap.parse_args()

    for v in args.variants:
        if v != "mixed":
            model = converter(v, args.calib).convert()
            out = f"{BASE}_{v}.tflite"
        else:
            c = converter("int8", args.calib)
            types = [t for t in args.denylist_types.split(",") if t]
            nodes = [l.strip() for l in open(args.denylist_nodes)] if args.denylist_nodes else []
            opts = tf.lite.experimental.QuantizationDebugOptions(
                denylisted_ops=types, denylisted_nodes=[n for n in nodes if n])
            dbg = tf.lite.experimental.QuantizationDebugger(
                converter=c, debug_dataset=rep_dataset(8), debug_options=opts)
            model = dbg.get_nondebug_quantized_model()
            out = f"{BASE}_{args.name}.tflite"
        open(out, "wb").write(model)
        print("wrote", out, f"{len(model) / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
