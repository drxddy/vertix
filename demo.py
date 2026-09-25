"""Latency benchmark: shared-prefix batched decisions vs. naive per-query inference.

Usage: python demo.py [--image path] [--iters 20] [--warmup 3] [--sizes 1,8,32]
"""
import argparse
import time

import numpy as np
import torch
from PIL import Image

from engine import VisualDecisionEngine

QUERIES = [
    "Is a human present in the image?",
    "Is the path clear for navigation?",
    "Are there any obstacles on the left?",
    "Is the lighting adequate?",
    "Is there a vehicle in the scene?",
    "Is the scene outdoors?",
    "Are there any safety hazards?",
    "Is the camera lens obstructed?",
]


def noise_image(seed=0, size=(640, 480)):
    # Noise rather than a flat colour: a realistic camera frame, not an unrealistically compressible one
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def timed(device, fn, warmup, iters):
    """Run fn warmup+iters times; return (last result, list of ms) with device syncs around each call."""
    times, result = [], None
    for i in range(warmup + iters):
        sync(device)
        start = time.perf_counter()
        result = fn()
        sync(device)
        if i >= warmup:
            times.append((time.perf_counter() - start) * 1e3)
    return result, times


def fmt(times):
    return f"p50 {np.percentile(times, 50):8.1f} ms   p95 {np.percentile(times, 95):8.1f} ms"


def kv_bytes(config, seq_len, dtype):
    text = config.text_config
    head_dim = getattr(text, "head_dim", None) or text.hidden_size // text.num_attention_heads
    elem = torch.finfo(dtype).bits // 8
    return 2 * text.num_hidden_layers * seq_len * text.num_key_value_heads * head_dim * elem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="image file (default: generated noise)")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--sizes", default="1,8,32")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB") if args.image else noise_image()
    engine = VisualDecisionEngine("Qwen/Qwen2-VL-2B-Instruct", num_classes=3)
    dev, w, it = engine.device, args.warmup, args.iters

    prefix = engine.encode_frame(image)
    kv_mb = kv_bytes(engine.model.config, prefix.prefix_len, engine.dtype) / 2**20
    print(f"\ndevice={dev} dtype={engine.dtype} image={engine.image_size(image)} "
          f"visual_tokens={prefix.n_visual_tokens} prefix_len={prefix.prefix_len} prefix_kv≈{kv_mb:.1f} MB")

    # Per-frame fixed cost, independent of the number of queries
    inputs, t_pre = timed(dev, lambda: engine.preprocess(image), w, it)
    _, t_enc = timed(dev, lambda: engine.encode_frame(inputs=inputs), w, it)
    print(f"\npreprocess  (CPU)              {fmt(t_pre)}")
    print(f"encode_frame (ViT + prefill)   {fmt(t_enc)}")

    for n in (int(s) for s in args.sizes.split(",")):
        queries = (QUERIES * (n // len(QUERIES) + 1))[:n]
        print(f"\n--- N = {n} queries ---")

        probs, t_dec = timed(dev, lambda: engine.decide(prefix, queries), w, it)
        _, t_total = timed(dev, lambda: engine.decide(engine.encode_frame(image), queries), w, it)
        _, t_batched = timed(dev, lambda: engine.naive_forward(image, queries), w, it)
        # Sequential naive is slow, so run it fewer times
        _, t_seq = timed(dev, lambda: [engine.naive_forward(image, [q]) for q in queries], 1, max(3, it // 5))

        print(f"decide (suffix batch + head)   {fmt(t_dec)}")
        print(f"shared-prefix total            {fmt(t_total)}   "
              f"(~{1e3 / np.percentile(t_total, 50):.1f} Hz)")
        print(f"naive batched (image x N)      {fmt(t_batched)}")
        print(f"naive sequential               {fmt(t_seq)}")
        p50 = lambda t: np.percentile(t, 50)
        print(f"speedup vs batched {p50(t_batched) / p50(t_total):.1f}x   "
              f"vs sequential {p50(t_seq) / p50(t_total):.1f}x")

    print("\nZero-shot readout (LM head's Yes/No logits at the decision token, same single pass):")
    for q, p in zip(QUERIES, engine.zero_shot(prefix, QUERIES)):
        print(f"  {'Yes' if p[0] > p[1] else 'No':>9}  P(Yes)={p[0]:.3f}  {q}")

    print("\nNOTE: decision_head is untrained, so the probabilities below are meaningless until it is trained.")
    class_names = ["Yes", "No", "Uncertain"]
    for q, p in zip(QUERIES, engine.decide(prefix, QUERIES)):
        print(f"  {class_names[int(p.argmax())]:>9}  {[round(x, 3) for x in p.tolist()]}  {q}")


if __name__ == "__main__":
    main()
