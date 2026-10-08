#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
warnings.filterwarnings("ignore")

BENCH_DIR = Path(__file__).resolve().parent
SUBMISSION_DIR = BENCH_DIR.parent
ROOT = SUBMISSION_DIR.parent

for p in (str(BENCH_DIR), str(SUBMISSION_DIR), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from engine import OptimizedXlitEngine, get_engine
from workloads import QUALITY_SAMPLES, generate_workload


def print_progress(done: int, total: int, latencies: list[float], elapsed: float) -> None:
    """
    Renders a sleek in-place progress bar without external dependencies.
    """
    width = 25
    fraction = done / total if total > 0 else 0.0
    filled = int(width * fraction)
    bar = "█" * filled + "░" * (width - filled)
    p50 = statistics.median(latencies) if latencies else 0.0
    rps = done / elapsed if elapsed > 0 else 0.0
    sys.stdout.write(
        f"\r  [{bar}] {int(fraction * 100):>3}% | {done:,}/{total:,} | "
        f"P50: {p50:>5.2f}ms | {rps:>5.1f} req/s"
    )
    sys.stdout.flush()


def summarize_latencies(
    latencies: list[float], total_time: float, total_words: int
) -> dict[str, Any]:
    """
    Calculates latency percentiles and throughput.
    """
    if not latencies:
        return {}
    s, n = sorted(latencies), len(latencies)
    return {
        "requests": n,
        "total_words": total_words,
        "total_time_seconds": round(total_time, 3),
        "mean_ms": round(statistics.mean(s), 3),
        "std_ms": round(statistics.stdev(s) if n > 1 else 0.0, 3),
        "min_ms": round(s[0], 3),
        "p50_ms": round(s[int(n * 0.50)], 3),
        "p90_ms": round(s[int(n * 0.90)], 3),
        "p95_ms": round(s[int(n * 0.95)], 3),
        "p99_ms": round(s[min(int(n * 0.99), n - 1)], 3),
        "max_ms": round(s[-1], 3),
        "throughput_req_per_sec": round(n / total_time, 1) if total_time > 0 else 0.0,
        "words_per_sec": round(total_words / total_time, 1) if total_time > 0 else 0.0,
    }


def execute_benchmark_loop(
    title: str, infer_fn: Callable[[Any], None], batches: list[Any], warmup: list[Any], concurrency: int = 1
) -> dict[str, Any]:
    """
    Unified runner: executes warmup, timed loop with live progress, and returns stats.
    """
    print(f"\n>> {title}")
    for b in warmup:
        infer_fn(b)

    total, total_words, latencies = len(batches), sum(len(b) if isinstance(b, list) else 1 for b in batches), []
    start_time, step = time.perf_counter(), max(1, len(batches) // 50)

    def timed_run(item: Any) -> float:
        t0 = time.perf_counter()
        infer_fn(item)
        return (time.perf_counter() - t0) * 1000.0

    it = (timed_run(b) for b in batches) if concurrency <= 1 else ThreadPoolExecutor(concurrency).map(timed_run, batches)
    for idx, lat in enumerate(it):
        latencies.append(lat)
        if (idx + 1) % step == 0 or (idx + 1) == total:
            print_progress(idx + 1, total, latencies, time.perf_counter() - start_time)

    duration = time.perf_counter() - start_time
    sys.stdout.write("\n")
    stats = summarize_latencies(latencies, duration, total_words)
    print(f"   P50: {stats['p50_ms']}ms | P95: {stats['p95_ms']}ms | Rate: {stats['throughput_req_per_sec']} req/s")
    return stats


def run_ct2_backend(
    batches: list[list[str]], warmup: list[list[str]], beam: int, topk: int, concurrency: int, no_cache: bool
) -> dict[str, Any]:
    """
    Benchmarks optimized CTranslate2 INT8 model with configurable concurrency.
    """
    engine = get_engine(
        backend="ct2", model_dir=ROOT / "models" / "ct2_indicxlit_int8",
        inter_threads=concurrency, intra_threads=1,
        cache_size=0 if no_cache else 10000, beam_width=beam,
    )

    def infer(b: list[str]) -> None:
        if len(b) == 1:
            engine.translit_word(b[0], topk=topk)
        else:
            engine.batch_transliterate_words(b, topk=topk)

    return execute_benchmark_loop(
        f"CTranslate2 INT8 (beam={beam}, topk={topk}, c={concurrency})",
        infer, batches, warmup, concurrency,
    )


_stock_worker_engine: Any = None


def _init_stock_worker(beam_width: int) -> None:
    global _stock_worker_engine
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    warnings.filterwarnings("ignore")
    import contextlib
    import io
    import torch

    torch.set_num_threads(1)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        _stock_worker_engine = get_engine(backend="stock", beam_width=beam_width)


def _infer_stock_worker(item: tuple[list[str], int]) -> float:
    global _stock_worker_engine
    b, topk = item
    word = b[0] if isinstance(b, list) else b
    t0 = time.perf_counter()
    _stock_worker_engine.translit_word(word, topk=topk)
    return (time.perf_counter() - t0) * 1000.0


def run_stock_backend(batches: list[list[str]], warmup: list[list[str]], topk: int, concurrency: int = 1) -> dict[str, Any]:
    """
    Benchmarks stock Fairseq FP32 baseline engine using multi-process pool (matching bench.py).
    """
    total, total_words = len(batches), sum(len(b) if isinstance(b, list) else 1 for b in batches)
    latencies: list[float] = []
    step = max(1, total // 50)
    title = f"Stock Fairseq FP32 ({total:,} requests, c={concurrency}, proc)" if concurrency > 1 else f"Stock Fairseq FP32 ({total:,} requests)"

    if concurrency <= 1:
        try:
            engine = get_engine(backend="stock", beam_width=topk)
        except Exception as exc:
            print(f"\n>> Stock Fairseq engine unavailable: {exc}. Skipping.")
            return {"status": "skipped"}

        def infer(b: list[str]) -> None:
            engine.translit_word(b[0], topk=topk)

        return execute_benchmark_loop(title, infer, batches, warmup, 1)

    print(f"\n>> {title}")
    try:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=concurrency, mp_context=ctx, initializer=_init_stock_worker, initargs=(topk,)) as pool:
            if warmup:
                warm_items = [(b, topk) for b in warmup]
                for _ in pool.map(_infer_stock_worker, warm_items):
                    pass

            items = [(b, topk) for b in batches]
            start_time = time.perf_counter()
            for idx, lat in enumerate(pool.map(_infer_stock_worker, items)):
                latencies.append(lat)
                if (idx + 1) % step == 0 or (idx + 1) == total:
                    print_progress(idx + 1, total, latencies, time.perf_counter() - start_time)

            duration = time.perf_counter() - start_time
            sys.stdout.write("\n")
            stats = summarize_latencies(latencies, duration, total_words)
            print(f"   P50: {stats['p50_ms']}ms | P95: {stats['p95_ms']}ms | Rate: {stats['throughput_req_per_sec']} req/s")
            return stats
    except Exception as exc:
        print(f"\n>> Stock Fairseq engine unavailable: {exc}. Skipping.")
        return {"status": "skipped"}


def run_quality_check() -> dict[str, Any]:
    """
    Performs quick sanity accuracy check against curated courtroom gold pairs.
    """
    engine = get_engine(backend="ct2", model_dir=ROOT / "models" / "ct2_indicxlit_int8", cache_size=0)
    total = len(QUALITY_SAMPLES)
    top1 = sum(1 for r, g in QUALITY_SAMPLES if (engine.translit_word(r, topk=4) or [""])[0] == g)
    top4 = sum(1 for r, g in QUALITY_SAMPLES if g in engine.translit_word(r, topk=4))
    print(f"\n>> Gold Quality Check: Top-1: {top1}/{total} ({(top1/total)*100:.1f}%) | Top-4: {top4}/{total} ({(top4/total)*100:.1f}%)")
    return {"top1_accuracy": round((top1 / total) * 100.0, 2), "top4_accuracy": round((top4 / total) * 100.0, 2)}


def parse_backends(value: str) -> list[str]:
    """
    Parses comma-separated backend arguments into validated backend identifiers.
    """
    valid = {"all", "ct2", "stock"}
    items = [b.strip().lower() for b in value.split(",") if b.strip()]
    for item in items:
        if item not in valid:
            raise argparse.ArgumentTypeError(f"Invalid backend '{item}'. Choose from: {', '.join(sorted(valid))}")
    return items


def parse_args() -> argparse.Namespace:
    """
    Configures and parses command-line arguments for the benchmark runner.
    """
    p = argparse.ArgumentParser(description="IndicXlit Benchmark Suite", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-n", "--requests", type=int, default=10000, help="Total benchmark requests")
    p.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    p.add_argument("-c", "--concurrency", type=int, default=1, help="Worker concurrency threads")
    p.add_argument("-B", "--batch-size", type=int, default=1, help="Words per batch")
    p.add_argument("-b", "--beam-width", type=int, default=4, help="Beam width")
    p.add_argument("-k", "--topk", type=int, default=4, help="Top-K candidates")
    p.add_argument("-w", "--workload", choices=["mixed", "short", "medium", "long", "smoke"], default="mixed", help="Workload bucket")
    p.add_argument("--backends", "--backend", type=parse_backends, default=["all"], help="Backends: ct2, stock, all")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--no-cache", action="store_true", help="Disable LRU cache")
    p.add_argument("--eval-quality", action="store_true", help="Quick quality smoke check")
    p.add_argument("--out", type=Path, default=None, help="Output JSON path")
    p.add_argument("--no-save", action="store_true", help="Do not save output JSON")
    return p.parse_args()


def main() -> None:
    """
    Entrypoint orchestrating benchmark runs across selected backends.
    """
    args = parse_args()
    sel = set(args.backends)
    run_all = "all" in sel
    batches, seed = generate_workload(args.requests, args.batch_size, args.workload, args.seed)
    warmup, _ = generate_workload(args.warmup, args.batch_size, "smoke", seed)

    print("=" * 78)
    print(f"  INDICXLIT RUNTIME BENCHMARK SUITE | Requests: {args.requests:,} | Workers: {args.concurrency}")
    print(f"  Workload: {args.workload} (seed={seed}) | Batch: {args.batch_size} | Beam: {args.beam_width} | TopK: {args.topk}")
    print(f"  Backends: {'ALL' if run_all else ', '.join(sorted(sel))}")
    print("=" * 78)

    results: dict[str, Any] = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "config": vars(args)}

    if run_all or "ct2" in sel:
        results["ctranslate2"] = run_ct2_backend(batches, warmup, args.beam_width, args.topk, args.concurrency, args.no_cache)
    if run_all or "stock" in sel:
        results["stock_fairseq"] = run_stock_backend(batches, warmup, args.topk, concurrency=args.concurrency)
    if args.eval_quality:
        results["quality"] = run_quality_check()

    if not args.no_save:
        out_file = args.out or (
            ROOT / "results" / f"bench_n{args.requests}_c{args.concurrency}_{args.workload}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w") as fp:
            json.dump(results, fp, indent=2, default=str)
        print(f"\n[Artifact Saved] -> {out_file}")

    print("\n" + "=" * 78)
    print("  [SUCCESS] Benchmark suite completed cleanly.")
    print("=" * 78)


if __name__ == "__main__":
    main()
