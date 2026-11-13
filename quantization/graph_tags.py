#!/usr/bin/env python3
"""Tag every op of calibrated.tflite (layer_sensitivity.py) by role and by position, for the
mixed-precision builders (build_static.py, make_mixed.py --roles).

  roles.json   role -> output tensor names
               layernorm  SUB(x, MEAN(x)) and the elementwise ops after it (var, eps, RSQRT, gamma, beta)
               attention  BATCH_MATMUL (QK^T, attn@V), SOFTMAX, and the MUL feeding a BATCH_MATMUL
               gelu       TANH and the elementwise ops around it (tanh-approximate GELU)
               residual   remaining ADDs with two non-constant inputs
  stages.json  stage -> output tensor names
               block0..block11  ops each ViT block's output depends on, minus the previous block's
               head             everything the last block's output does not depend on (DPT head)

Usage:  python graph_tags.py
"""
import json

import tensorflow as tf

ELEMWISE = {"MUL", "ADD", "SUB", "MEAN", "RSQRT"}


def tag_roles(model_bytes):
    it = tf.lite.Interpreter(model_content=model_bytes)
    ops = [o for o in it._get_ops_details() if o["op_name"] not in ("QUANTIZE", "DEQUANTIZE")]
    td = {t["index"]: t for t in it.get_tensor_details()}
    prod = {t: o["index"] for o in ops for t in o["outputs"]}
    cons = {}
    for o in ops:
        for t in o["inputs"]:
            cons.setdefault(t, []).append(o["index"])
    by = {o["index"]: o for o in ops}
    name = lambda o: td[o["outputs"][0]]["name"]
    is_const = lambda t: t not in prod and t != it.get_input_details()[0]["index"]

    roles = {"layernorm": set(), "attention": set(), "gelu": set(), "residual": set()}
    for o in ops:  # LayerNorm: SUB(x, MEAN(x))
        if o["op_name"] == "SUB" and any(prod.get(t) is not None and by[prod[t]]["op_name"] == "MEAN"
                                         for t in o["inputs"]):
            m = [prod[t] for t in o["inputs"] if prod.get(t) is not None and by[prod[t]]["op_name"] == "MEAN"][0]
            roles["layernorm"].add(m)
            stack, seen = [o["index"]], set()
            while stack:
                k = stack.pop()
                if k in seen:
                    continue
                seen.add(k)
                roles["layernorm"].add(k)
                for c in cons.get(by[k]["outputs"][0], []):
                    if by[c]["op_name"] in ELEMWISE:
                        stack.append(c)
                # stop after the final ADD (beta): its consumers are not elementwise-LN ops anymore
    for o in ops:
        if o["op_name"] in ("BATCH_MATMUL", "SOFTMAX"):
            roles["attention"].add(o["index"])
        if o["op_name"] == "MUL" and any(by[c]["op_name"] == "BATCH_MATMUL" for c in cons.get(o["outputs"][0], [])):
            roles["attention"].add(o["index"])
    for o in ops:  # GELU (tanh approx): ops within 3 hops of a TANH that are elementwise
        if o["op_name"] == "TANH":
            frontier = {o["index"]}
            roles["gelu"].add(o["index"])
            for _ in range(3):
                nxt = set()
                for k in frontier:
                    nb = [prod[t] for t in by[k]["inputs"] if t in prod] + cons.get(by[k]["outputs"][0], [])
                    nxt |= {n for n in nb if by[n]["op_name"] in {"MUL", "ADD"} and n not in roles["layernorm"]}
                roles["gelu"] |= nxt
                frontier = nxt
    taken = set().union(*roles.values())
    for o in ops:
        if o["op_name"] == "ADD" and o["index"] not in taken and not any(is_const(t) for t in o["inputs"]):
            roles["residual"].add(o["index"])
    out = {r: sorted(name(by[k]) for k in v) for r, v in roles.items()}
    for r, v in roles.items():
        print(f"{r:10s} {len(v):4d} ops", sorted({by[k]['op_name'] for k in v}))
    return out


def tag_stages(model_bytes):
    it = tf.lite.Interpreter(model_content=model_bytes)
    ops = it._get_ops_details()
    td = {t["index"]: t["name"] for t in it.get_tensor_details()}
    prod = {t: o["index"] for o in ops for t in o["outputs"]}
    cons = {}
    for o in ops:
        for t in o["inputs"]:
            cons.setdefault(t, []).append(o["index"])
    by = {o["index"]: o for o in ops}

    fc = [o["index"] for o in ops if o["op_name"] == "FULLY_CONNECTED"]
    assert len(fc) == 48, len(fc)
    block_out = []
    for k in range(12):  # fc2 -> (MUL layer-scale) -> ADD residual
        cur, found = fc[4 * k + 3], None
        for _ in range(3):
            nxt = cons.get(by[cur]["outputs"][0], [])
            adds = [c for c in nxt if by[c]["op_name"] == "ADD"]
            if adds:
                found = adds[0]
                break
            cur = nxt[0]
        assert found is not None, k
        block_out.append(found)

    def ancestors(op):
        seen, stack = set(), [op]
        while stack:
            k = stack.pop()
            if k in seen:
                continue
            seen.add(k)
            stack += [prod[t] for t in by[k]["inputs"] if t in prod]
        return seen

    stages, prev = {}, set()
    for k in range(12):
        a = ancestors(block_out[k])
        stages[f"block{k}"] = a - prev
        prev = a
    stages["head"] = set(by) - prev
    for s, v in stages.items():
        print(f"{s:8s} {len(v):4d} ops  FC={sum(by[i]['op_name'] == 'FULLY_CONNECTED' for i in v)}"
              f"  CONV={sum(by[i]['op_name'] in ('CONV_2D', 'TRANSPOSE_CONV') for i in v)}")
    # block0 also holds the patch embedding; the stem is small
    return {s: sorted({td[by[i]["outputs"][0]] for i in v}) for s, v in stages.items()}


def main():
    cal = open("calibrated.tflite", "rb").read()
    json.dump(tag_roles(cal), open("roles.json", "w"), indent=1)
    json.dump(tag_stages(cal), open("stages.json", "w"), indent=1)
    print("wrote roles.json, stages.json")


if __name__ == "__main__":
    main()
