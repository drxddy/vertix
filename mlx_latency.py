"""Latency of the MLX shared-prefix engine: per-stage, per query count, and per early-exit depth.

Measures on one image, with MLX work forced by mx.eval (MLX is lazy):
  vision        vision encoder only
  encode@k      vision + language prefill through the first k layers (k = all: full depth)
  decide@k(N)   N queries as one batch forked from the prefix, through k layers
  naive(N)      N independent full sequences (image re-encoded per query), the no-sharing baseline
Usage: venv-mlx/bin/python mlx_latency.py [--image image.png] [--budgets 224,448] [--exits 12,16,24]
"""
import argparse
import time

import mlx.core as mx
import numpy as np
from PIL import Image

from mlx_engine import MLXDecisionEngine

QUERIES = [
    "Is a human present in the image?",
    "Is the person to the left of the car?",
    "Is there a vehicle in the scene?",
    "Is the traffic light red?",
    "Is anyone holding an umbrella?",
    "Is the path in front of the camera blocked?",
    "Are there more people than cars?",
    "Is the lighting adequate?",
]


def timed(fn, warmup=3, iters=15):
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1e3)
    return np.percentile(t, 50), np.percentile(t, 95)


def fmt(t):
    return f"{t[0]:7.1f} / {t[1]:6.1f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen3.5-2B-bf16")
    parser.add_argument("--image", default="image.png")
    parser.add_argument("--budgets", default="224,448")
    parser.add_argument("--exits", default="12,16,24", help="layer counts to run (24 = full depth)")
    parser.add_argument("--sizes", default="1,6,32")
    args = parser.parse_args()

    engine = MLXDecisionEngine(args.model)
    image = Image.open(args.image).convert("RGB")
    exits = [int(k) for k in args.exits.split(",")]
    sizes = [int(n) for n in args.sizes.split(",")]
    print(f"{args.model}  layers={engine.total_layers}  (p50 / p95 ms)")

    for budget in (int(b) for b in args.budgets.split(",")):
        engine.pixel_budget = budget * budget
        inputs = engine.preprocess(image)
        grid = inputs["image_grid_thw"]
        vt = engine.model.vision_tower
        print(f"\n--- budget {budget}: image {engine.image_size(image)}  ---")
        print(f"vision encoder only        {fmt(timed(lambda: mx.eval(vt(inputs['pixel_values'].astype(vt.patch_embed.proj.weight.dtype), grid)[0])))}")
        for k in exits:
            prefix = engine.encode_frame(inputs=inputs, n_layers=k)
            enc = timed(lambda: engine.encode_frame(inputs=inputs, n_layers=k))
            line = f"encode@{k:<2d}                  {fmt(enc)}"
            for n in sizes:
                qs = (QUERIES * (n // len(QUERIES) + 1))[:n]
                dec = timed(lambda: mx.eval(engine.decision_hidden(prefix, qs)))
                line += f" | decide(N={n}) {fmt(dec)}"
            print(line + f"   [prefix_len={prefix.prefix_len}, visual={prefix.n_visual_tokens}]")
        n = 6
        qs = QUERIES[:n]
        naive = timed(lambda: mx.eval([engine.reference_hidden(image, q) for q in qs]), warmup=1, iters=5)
        full = timed(lambda: mx.eval(engine.decision_hidden(engine.encode_frame(inputs=inputs), qs)))
        print(f"naive, {n} independent full sequences {fmt(naive)}  vs shared prefix encode+decide {fmt(full)}"
              f"  -> {naive[0] / full[0]:.1f}x")


if __name__ == "__main__":
    main()
