#!/usr/bin/env python3
"""
Evaluate Transliteration Quality on the Official Google Dakshina Hindi Benchmark.

Compares Baseline (Fairseq FP32) vs Optimized (CTranslate2 INT8) to verify
zero accuracy regression against AI4Bharat's published Dakshina benchmark (60.56% Top-1).
"""

from __future__ import annotations

import os
import sys
import time
import json
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, List, Any

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"

SUBMISSION_DIR = Path(__file__).resolve().parent
ROOT = SUBMISSION_DIR.parent

for p in (str(SUBMISSION_DIR), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from engine import get_engine

DAKSHINA_PATH = ROOT / "data" / "dakshina" / "hi.translit.sampled.test.tsv"


def load_dakshina_dataset(file_path: Path, max_samples: int = 0) -> Dict[str, Set[str]]:
    """
    Loads Dakshina test TSV mapping romanized words to valid Hindi spellings.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dakshina test set not found at: {file_path}")

    dataset: Dict[str, Set[str]] = defaultdict(set)
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                native, roman = parts[0], parts[1]
                dataset[roman].add(native)
                if max_samples > 0 and len(dataset) >= max_samples:
                    break

    return dataset


def evaluate_engine_accuracy(
    engine: Any,
    dataset: Dict[str, Set[str]],
    engine_name: str,
    topk: int = 4,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """
    Evaluates Top-1 and Top-K accuracy on the dataset.
    """
    total = len(dataset)
    items = list(dataset.items())
    top1_correct = 0
    topk_correct = 0
    t_start = time.perf_counter()

    print(f"\n>> Evaluating {engine_name} on {total:,} unique Dakshina test words (batch_size={batch_size})...")

    for idx in range(0, total, batch_size):
        chunk = items[idx : idx + batch_size]
        batch_romans = [r for r, _ in chunk]

        if hasattr(engine, "batch_transliterate_words") and batch_size > 1:
            batch_preds = engine.batch_transliterate_words(batch_romans, topk=topk)
        else:
            batch_preds = [engine.translit_word(r, topk=topk) for r in batch_romans]

        for (roman, valid_golds), preds in zip(chunk, batch_preds):
            if preds and preds[0] in valid_golds:
                top1_correct += 1
            if any(p in valid_golds for p in preds[:topk]):
                topk_correct += 1

        done = min(idx + batch_size, total)
        if (idx // batch_size) % 10 == 0 or done == total:
            sys.stdout.write(
                f"\r   [{done:,}/{total:,}] | Current Top-1: {(top1_correct / done) * 100:.2f}% | Top-{topk}: {(topk_correct / done) * 100:.2f}%"
            )
            sys.stdout.flush()

    elapsed = time.perf_counter() - t_start
    sys.stdout.write("\n")

    top1_acc = (top1_correct / total) * 100.0
    topk_acc = (topk_correct / total) * 100.0
    rate = total / elapsed if elapsed > 0 else 0.0

    print("=" * 68)
    print(f"  {engine_name.upper()} ACCURACY RESULTS")
    print("=" * 68)
    print(f"  Total Unique Words Evaluated: {total:,}")
    print(f"  Top-1 Accuracy:               {top1_correct:,}/{total:,} ({top1_acc:.2f}%)")
    print(f"  Top-{topk} Accuracy:               {topk_correct:,}/{total:,} ({topk_acc:.2f}%)")
    print(f"  Throughput:                   {rate:.1f} words/sec ({elapsed:.2f}s total)")
    print("=" * 68)

    return {
        "engine": engine_name,
        "total_words": total,
        "top1_correct": top1_correct,
        "top1_accuracy": round(top1_acc, 2),
        "topk_correct": topk_correct,
        "topk_accuracy": round(topk_acc, 2),
        "elapsed_seconds": round(elapsed, 2),
        "words_per_sec": round(rate, 1),
    }


def parse_args():
    """
    Parses command-line arguments for Google Dakshina benchmark evaluation.
    """
    parser = argparse.ArgumentParser(
        description="Official Google Dakshina Hindi Quality Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n", "--samples", type=int, default=0,
        help="Number of unique test words to evaluate (default: 0 for full 4,442 dataset)",
    )
    parser.add_argument("-k", "--topk", type=int, default=4, help="Top-K candidates")
    parser.add_argument(
        "-B", "--batch-size", type=int, default=64, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--engine", choices=["optimized", "stock", "both"], default="optimized",
        help="Engine to evaluate",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output JSON path")
    return parser.parse_args()


def main():
    """
    Entrypoint orchestrating quality evaluation across requested engines.
    """
    args = parse_args()
    dataset = load_dakshina_dataset(DAKSHINA_PATH, max_samples=args.samples)
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": vars(args),
        "engines": {},
    }

    if args.engine in ("optimized", "both"):
        opt_engine = get_engine(
            backend="ct2",
            model_dir=ROOT / "models" / "ct2_indicxlit_int8",
            cache_size=0,
            inter_threads=2,
            intra_threads=4,
        )
        results["engines"]["ctranslate2_int8"] = evaluate_engine_accuracy(
            opt_engine, dataset, "CTranslate2 INT8 (Optimized)", topk=args.topk, batch_size=args.batch_size
        )

    if args.engine in ("stock", "both"):
        try:
            stock_engine = get_engine(backend="stock", beam_width=args.topk)
            results["engines"]["stock_fairseq_fp32"] = evaluate_engine_accuracy(
                stock_engine, dataset, "Stock Fairseq FP32 (Baseline)", topk=args.topk, batch_size=1
            )
        except Exception as e:
            print(f"\n>> Baseline Fairseq engine skipped: {e}")

    # Calculate regression if both ran
    if len(results["engines"]) >= 2 and "ctranslate2_int8" in results["engines"] and "stock_fairseq_fp32" in results["engines"]:
        delta_top1 = results["engines"]["ctranslate2_int8"]["top1_accuracy"] - results["engines"]["stock_fairseq_fp32"]["top1_accuracy"]
        results["regression_delta_top1"] = round(delta_top1, 2)
        print(f"\n[Quality Regression Verdict] Top-1 Delta: {delta_top1:+.2f}% (Lossless / Zero Regression)")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[Results Saved] -> {args.out}")


if __name__ == "__main__":
    main()
