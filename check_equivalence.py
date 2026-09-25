"""Checks that the shared-prefix batched path gives the same decision hidden states as full sequences.

Usage: python check_equivalence.py [--model id] [--device cpu|mps] [--image path] [--wrong-pos]
  --wrong-pos  deliberately uses prefix_len as the suffix start position (ignoring M-RoPE's
               compressed vision positions) to confirm this check catches position-id bugs.
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from engine import VisualDecisionEngine

QUERIES = [
    "Is a human present?",
    "Is the path clear for navigation, or is something blocking the way forward?",
    "Any hazards?",
    "Is the camera lens obstructed by dirt, water droplets, or a finger?",
]


def noise_image(seed=0, size=(640, 480)):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image", help="image file (default: generated noise)")
    parser.add_argument("--wrong-pos", action="store_true")
    args = parser.parse_args()

    # fp32 on CPU for a tight tolerance; bf16 elsewhere with a loose one
    fp32 = args.device == "cpu"
    dtype = torch.float32 if fp32 else torch.bfloat16
    max_diff_tol, cos_tol = (1e-3, 0.9999) if fp32 else (None, 0.99)

    image = Image.open(args.image).convert("RGB") if args.image else noise_image()
    engine = VisualDecisionEngine(args.model, device=args.device, dtype=dtype)

    prefix = engine.encode_frame(image)
    print(f"prefix_len={prefix.prefix_len} next_pos={prefix.next_pos} visual_tokens={prefix.n_visual_tokens}")
    if args.wrong_pos:
        prefix.next_pos = prefix.prefix_len

    cache_before = [(l.keys.clone(), l.values.clone()) for l in prefix.cache.layers]
    shared = engine.decision_hidden(prefix, QUERIES).float()

    ok = True
    for i, q in enumerate(QUERIES):
        ref = engine.reference_hidden(image, q).float()
        max_diff = (shared[i] - ref).abs().max().item()
        cos = F.cosine_similarity(shared[i], ref, dim=0).item()
        passed = cos > cos_tol and (max_diff_tol is None or max_diff < max_diff_tol)
        ok &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] max|diff|={max_diff:.2e} cos={cos:.6f}  {q}")

    unchanged = all(
        l.keys.shape == k.shape and torch.equal(l.keys, k) and torch.equal(l.values, v)
        for l, (k, v) in zip(prefix.cache.layers, cache_before)
    )
    ok &= unchanged
    print(f"[{'PASS' if unchanged else 'FAIL'}] shared prefix cache unchanged after decide()")

    print("ALL PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
