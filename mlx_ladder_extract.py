"""Decision-ladder feature extraction with the MLX engine (run in venv-mlx).

Writes data/cache/ladder_vlm__<model>_<budget>.npz in the same format as `ladder.py extract-vlm`,
so `venv/bin/python ladder.py report` compares it directly with the torch-extracted models.
Timings recorded here include copying every layer's state to CPU; use mlx_latency.py for latency.

Usage: venv-mlx/bin/python mlx_ladder_extract.py [--model mlx-community/Qwen3.5-2B-bf16] [--budget 224]
"""
import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from mlx_engine import MLXDecisionEngine

DATA = Path(__file__).parent / "data"
MAX_BATCH = 32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen3.5-2B-bf16")
    parser.add_argument("--budget", type=int, default=224)
    args = parser.parse_args()

    rows = [json.loads(l) for l in open(DATA / "ladder/manifest.jsonl")]
    frames = {}
    for i, r in enumerate(rows):
        frames.setdefault(r["group"], []).append(i)

    engine = MLXDecisionEngine(args.model, pixel_budget=args.budget * args.budget)
    n_states = engine.total_layers + 1
    hidden_size = engine.model.config.text_config.hidden_size
    hidden = np.zeros((len(rows), n_states, hidden_size), dtype=np.float16)
    pooled = np.zeros((len(frames), n_states, hidden_size), dtype=np.float16)
    zeroshot = np.zeros(len(rows), dtype=np.float32)
    frame_of_row = np.zeros(len(rows), dtype=np.int32)
    t_enc, t_dec = [], []

    for f, idx in enumerate(frames.values()):
        frame_of_row[idx] = f
        inputs = engine.preprocess(Image.open(DATA / rows[idx[0]]["image"]).convert("RGB"))
        t0 = time.perf_counter()
        prefix = engine.encode_frame(inputs=inputs, pool_layers=True)
        t1 = time.perf_counter()
        for start in range(0, len(idx), MAX_BATCH):
            chunk = idx[start:start + MAX_BATCH]
            h = engine.decision_hidden(prefix, [rows[i]["question"] for i in chunk], all_layers=True)
            p = mx.softmax(engine.yes_no_logits(h[:, -1]).astype(mx.float32), axis=-1)[:, 0]
            mx.eval(h, p)
            hidden[chunk] = np.array(h.astype(mx.float32))
            zeroshot[chunk] = np.array(p)
        t2 = time.perf_counter()
        pooled[f] = prefix.visual_pooled
        t_enc.append((t1 - t0) * 1e3)
        t_dec.append(((t2 - t1) * 1e3, len(idx)))
        if f % 250 == 0:
            print(f"  frame {f}/{len(frames)}  encode {np.median(t_enc):.0f} ms", flush=True)

    out = DATA / "cache" / f"ladder_vlm__{args.model.split('/')[-1]}_{args.budget}.npz"
    np.savez(out, hidden=hidden, zeroshot=zeroshot, pooled=pooled, frame_of_row=frame_of_row,
             t_encode=np.array(t_enc), t_decide=np.array(t_dec))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
