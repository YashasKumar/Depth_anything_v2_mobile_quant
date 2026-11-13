#!/usr/bin/env python3
"""Mixed-precision int8 variants of Depth Anything V2 from calibrated.tflite (layer_sensitivity.py).

Everything is int8 (per-channel weights, int8 activations, float I/O) except what is denylisted, which
stays float. No recalibration: the same calibration statistics are reused for every variant.

  --types   TFLite op types kept float, e.g. SOFTMAX,MEAN,RSQRT
  --regex   regex on output tensor names kept float (names come from the ONNX graph, e.g. 'norm|attn')
  --top_k   the K most sensitive layers by rmse/scale in layer_stats.csv kept float
  --roles   graph roles kept float, from roles.json (graph_tags.py): layernorm,attention,gelu,residual

Usage:  python make_mixed.py --name ln_float --types MEAN,RSQRT
"""
import argparse
import json
import re

import pandas as pd
import tensorflow as tf
from tensorflow.lite.python import convert

BASE = "depth_anything_v2_vits_392x518"


def tensor_names(model_bytes):
    it = tf.lite.Interpreter(model_content=model_bytes)
    return [it._get_tensor_details(o, 0)["name"] for op in it._get_ops_details() for o in op["outputs"]]


def build(name, types=(), regex="", top_k=0, roles=(), calibrated="calibrated.tflite"):
    cal = open(calibrated, "rb").read()
    nodes = []
    if regex:
        nodes += [n for n in tensor_names(cal) if re.search(regex, n)]
    if top_k:
        df = pd.read_csv("layer_stats.csv").sort_values("rmse/scale", ascending=False)
        nodes += list(df["tensor_name"].head(top_k))
    if roles:
        rj = json.load(open("roles.json"))
        nodes += [n for r in roles for n in rj[r]]
    nodes = sorted(set(nodes))
    model = convert.mlir_quantize(cal, disable_per_channel=False, fully_quantize=False,
                                  enable_numeric_verify=False, denylisted_ops=list(types),
                                  denylisted_nodes=nodes)
    out = f"{BASE}_mixed_{name}.tflite"
    open(out, "wb").write(model)
    print(f"wrote {out} {len(model) / 1e6:.1f} MB  float types={list(types)}  float nodes={len(nodes)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--types", default="")
    ap.add_argument("--regex", default="")
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--roles", default="")
    a = ap.parse_args()
    build(a.name, [t for t in a.types.split(",") if t], a.regex, a.top_k, [r for r in a.roles.split(",") if r])


if __name__ == "__main__":
    main()
