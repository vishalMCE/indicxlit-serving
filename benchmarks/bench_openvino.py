#!/usr/bin/env python3
"""
Benchmark Intel Iris Xe iGPU vs CPU using OpenVINO (FP32 & INT8).
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
import numpy as np

try:
    import openvino as ov
    try:
        import openvino.opset10 as ops
    except ImportError:
        from openvino import op as ops
except ImportError:
    print("OpenVINO not installed. Run: pip install openvino")
    sys.exit(1)

def build_model(bs: int, prec: str = "fp32", s: int = 16, d: int = 256, ff: int = 1024):
    """
    Constructs a Transformer FFN sub-layer in FP32 or INT8.
    """
    param = ops.parameter([bs, s, d], np.float32, name="x")
    if prec == "int8":
        w1_r = ops.constant(np.random.randint(-128, 127, (d, ff), dtype=np.int8))
        w2_r = ops.constant(np.random.randint(-128, 127, (ff, d), dtype=np.int8))
        sc = ops.constant(np.float32(0.01))
        w1 = ops.multiply(ops.convert(w1_r, np.float32), sc)
        w2 = ops.multiply(ops.convert(w2_r, np.float32), sc)
    else:
        w1 = ops.constant(np.random.randn(d, ff).astype(np.float32))
        w2 = ops.constant(np.random.randn(ff, d).astype(np.float32))
    act = ops.relu(ops.matmul(param, w1, False, False))
    out = ops.result(ops.matmul(act, w2, False, False), name="out")
    return ov.Model([out], [param], f"ffn_b{bs}_{prec}")

def bench_device(core: ov.Core, dev: str, n_iter: int, bs: int, prec: str):
    """
    Measures latency at batch=1 and throughput at batch=32.
    """
    c1 = core.compile_model(build_model(1, prec), dev)
    cb = core.compile_model(build_model(bs, prec), dev)
    x1 = np.random.randn(1, 16, 256).astype(np.float32)
    xb = np.random.randn(bs, 16, 256).astype(np.float32)
    for _ in range(15):
        c1([x1]), cb([xb])
    lats = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        c1([x1])
        lats.append((time.perf_counter() - t0) * 1000.0)
    lats.sort()
    t0 = time.perf_counter()
    for _ in range(100):
        cb([xb])
    wps = (100 * bs) / (time.perf_counter() - t0)
    return {
        "device": dev, "precision": prec.upper(),
        "p50_ms": round(lats[n_iter // 2], 3),
        "p95_ms": round(lats[int(n_iter * 0.95)], 3),
        "batch32_wps": round(wps, 1),
    }

def main():
    """
    Discovers Intel devices and runs OpenVINO benchmarks.
    """
    p = argparse.ArgumentParser(description="OpenVINO CPU vs iGPU Benchmark")
    p.add_argument("-n", type=int, default=200, help="iterations")
    p.add_argument("-b", type=int, default=32, help="batch size")
    p.add_argument("-p", "--prec", choices=["fp32", "int8"], default="fp32")
    p.add_argument("-o", "--out", type=Path, default="results/intel_openvino.json")
    args = p.parse_args()

    core = ov.Core()
    devs = [d for d in ["CPU", "GPU"] if d in core.available_devices]
    if not devs:
        print("No supported CPU or GPU found.")
        sys.exit(1)

    print(f"Devices: {devs} | Precision: {args.prec.upper()}")
    res = {d.lower(): bench_device(core, d, args.n, args.b, args.prec) for d in devs}
    results = res
    hdr = f"{'Dev':<5} | {'Prec':<5} | {'P50':<7} | {'P95':<7} | {'Batch-32 wps':<12}"
    print(f"\n{hdr}\n" + "-" * len(hdr))
    for dev, res in results.items():
        p50, p95, wps = res["p50_ms"], res["p95_ms"], res["batch32_wps"]
        pr = res["precision"]
        print(f"{dev.upper():<5} | {pr:<5} | {p50:>7.3f} | {p95:>7.3f} | {wps:>12.1f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved results: {out_path}")

if __name__ == "__main__":
    main()
