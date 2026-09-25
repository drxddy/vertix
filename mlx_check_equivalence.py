"""MLX engine equivalence: forked-batch decisions must match ordinary full-sequence runs.

Checks, per query (lengths differ, so right padding is exercised):
  1. all layers of the decision state from the forked batch == the full single-sequence run
  2. early exit: running the suffix through only k layers == layer k of the full-depth run
  3. the shared prefix state is unchanged after decide (it can be reused)
Usage: venv-mlx/bin/python mlx_check_equivalence.py [--image image.png] [--fp32] [--wrong-pos]
"""
import argparse
import sys

import mlx.core as mx
import numpy as np
from PIL import Image

from mlx_engine import MLXDecisionEngine

QUERIES = [
    "Is a human present?",
    "Is the path clear for navigation, or is something blocking the way forward?",
    "Any hazards?",
    "Is the camera lens obstructed by dirt, water droplets, or a finger?",
]


def cos(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="image.png")
    parser.add_argument("--model", default="mlx-community/Qwen3.5-2B-bf16")
    parser.add_argument("--fp32", action="store_true", help="cast weights to float32 for a tight tolerance")
    parser.add_argument("--wrong-pos", action="store_true")
    args = parser.parse_args()
    # Tight in both precisions: forking is exact here, and only 6 of 24 layers use RoPE, so a position
    # bug moves the output by just ~1e-3 in cosine (a loose tolerance would miss it)
    max_rel_tol, cos_tol = 1e-3, 0.99999

    engine = MLXDecisionEngine(args.model, dtype=mx.float32 if args.fp32 else None)
    image = Image.open(args.image).convert("RGB")
    prefix = engine.encode_frame(image)
    print(f"prefix_len={prefix.prefix_len} next_pos={prefix.next_pos} visual_tokens={prefix.n_visual_tokens}")
    if args.wrong_pos:
        prefix.next_pos = prefix.prefix_len

    before = [[np.array(x.astype(mx.float32)) for x in (c.cache if hasattr(c, "cache") else c.state)] for c in prefix.cache]
    shared = np.array(engine.decision_hidden(prefix, QUERIES, all_layers=True).astype(mx.float32))
    k = engine.total_layers * 2 // 3
    early = np.array(engine.decision_hidden(prefix, QUERIES, n_layers=k).astype(mx.float32))

    ok = True
    for i, q in enumerate(QUERIES):
        ref = np.array(engine.reference_hidden(image, q, all_layers=True).astype(mx.float32))
        per_layer = [cos(shared[i, L], ref[L]) for L in range(1, ref.shape[0])]
        rel = np.abs(shared[i, -1] - ref[-1]).max() / np.abs(ref[-1]).max()
        early_cos = cos(early[i], ref[k])
        passed = min(per_layer) > cos_tol and early_cos > cos_tol and (max_rel_tol is None or rel < max_rel_tol)
        ok &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] min cos over layers={min(per_layer):.6f}  last-layer max rel diff={rel:.1e}  "
              f"early-exit L{k} cos={early_cos:.6f}  {q}")

    after = [[np.array(x.astype(mx.float32)) for x in (c.cache if hasattr(c, "cache") else c.state)] for c in prefix.cache]
    unchanged = all(a.shape == b.shape and np.array_equal(a, b) for la, lb in zip(before, after) for a, b in zip(la, lb))
    ok &= unchanged
    print(f"[{'PASS' if unchanged else 'FAIL'}] shared prefix state unchanged after decide()")
    print("ALL PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
