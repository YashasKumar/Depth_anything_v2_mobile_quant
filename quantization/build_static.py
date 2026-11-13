#!/usr/bin/env python3
"""Static-only mixed precision (no dynamic-range ops): every op is either FP32 or static int8.
Combines position (stages.json: block0..11, head) and role (roles.json: layernorm, attention, gelu,
residual). Per config:
  int8     stages fully static int8
  fc_int8  stages where only the compute (FC/CONV/...) is int8; LayerNorm/attention/GELU/residual float
  rest     FP32
Usage:  python build_static.py [config ...]
"""
import json
import sys

from tensorflow.lite.python import convert

BASE = "depth_anything_v2_vits_392x518"
B = [f"block{k}" for k in range(12)]
TOLERANT = ["block1", "block2", "block9", "block11"]
SENSITIVE = ["block3", "block4", "block5", "block8", "block10"]
CONFIGS = {
    "st_fcint8_tolerant_head": dict(int8=["head"], fc_int8=TOLERANT),
    "st_fcint8_safe_head": dict(int8=["head"], fc_int8=[b for b in B if b not in SENSITIVE]),
    "st_fcint8_no345_head": dict(int8=["head"], fc_int8=[b for b in B if b not in ("block3", "block4", "block5")]),
}

cal = open("calibrated.tflite", "rb").read()
stages, roles = json.load(open("stages.json")), json.load(open("roles.json"))
role_nodes = {n for v in roles.values() for n in v}
for name in sys.argv[1:] or CONFIGS:
    c = CONFIGS[name]
    fl = set()
    for s, nodes in stages.items():
        if s in c["int8"]:
            continue
        fl |= (set(nodes) & role_nodes) if s in c["fc_int8"] else set(nodes)
    m = convert.mlir_quantize(cal, fully_quantize=False, denylisted_ops=[], denylisted_nodes=sorted(fl))
    open(f"{BASE}_{name}.tflite", "wb").write(m)
    print(f"wrote {BASE}_{name}.tflite  float nodes={len(fl)}", flush=True)
