#!/usr/bin/env python3
"""Per-layer int8 sensitivity of Depth Anything V2 (TFLite) with TF's QuantizationDebugger.

1. Calibrates the SavedModel once on calib_train200_nhwc.npy (NYU train split) and saves the
   calibrated (not yet quantized) model to calibrated.tflite, so every mixed-precision variant can be
   produced from it without recalibrating (see make_mixed.py).
2. Runs the debug model (float path + NumericVerify per quantized layer) on --debug_n calibration images
   and writes layer_stats.csv: per output tensor, the op type and the int8 error relative to that
   tensor's quantization step (rmse/scale; ~0.29 is ideal rounding-only error).

Usage:  python layer_sensitivity.py [--debug_n 8]
"""
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

from quantize_tflite import SM, rep_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", type=int, default=200)
    ap.add_argument("--debug_n", type=int, default=8)
    args = ap.parse_args()

    c = tf.lite.TFLiteConverter.from_saved_model(SM)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset = rep_dataset(args.calib)
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]

    dbg = tf.lite.experimental.QuantizationDebugger(
        converter=c, debug_dataset=rep_dataset(args.debug_n),
        debug_options=tf.lite.experimental.QuantizationDebugOptions(
            layer_debug_metrics={"mean_abs_error": lambda d: np.mean(np.abs(d))}))
    open("calibrated.tflite", "wb").write(dbg.calibrated_model)
    print("calibrated model saved", flush=True)

    dbg.run()
    with open("layer_stats_raw.csv", "w") as f:
        dbg.layer_statistics_dump(f)

    df = pd.read_csv("layer_stats_raw.csv")
    df["range"] = 255.0 * df["scale"]
    df["rmse/scale"] = df.apply(lambda r: np.sqrt(r["mean_squared_error"]) / r["scale"], axis=1)
    df.to_csv("layer_stats.csv", index=False)
    print(df.groupby("op_name")["rmse/scale"].describe().sort_values("max", ascending=False).to_string())
    cols = ["op_name", "tensor_name", "scale", "rmse/scale", "max_abs_error"]
    print(df.sort_values("rmse/scale", ascending=False)[cols].head(40).to_string())


if __name__ == "__main__":
    main()
