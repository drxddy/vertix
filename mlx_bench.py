"""MLX latency probe for the frame-encode bottleneck (vision encoder + prefill), run in venv-mlx.

Mirrors the torch engine's cost structure on one image: the vision tower alone, then a full
prefix+query prefill, reading P(Yes) at the decision token as a sanity check on the port.

Usage: venv-mlx/bin/python mlx_bench.py [--image image.png] [--models id1,id2] [--budgets 448,224]
"""
import argparse
import time

import mlx.core as mx
import numpy as np
from mlx_vlm import load
from mlx_vlm.utils import prepare_inputs
from PIL import Image

from prompts import PREFIX_INSTRUCTION, SUFFIX_TEMPLATE, fit_size

QUERY = "Is a human present in the image?"


def timed(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return np.percentile(times, 50), np.percentile(times, 95)


def bench(model_id, image, budget):
    model, processor = load(model_id)
    ip = processor.image_processor
    h, w = fit_size(*image.size, budget=budget * budget, multiple=ip.patch_size * ip.merge_size)
    img = image.resize((w, h), Image.BICUBIC)

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PREFIX_INSTRUCTION}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    text = text[: text.rfind("<|im_end|>")] + SUFFIX_TEMPLATE.format(q=QUERY)
    inputs = prepare_inputs(processor, images=[img], prompts=[text])
    input_ids, pixel_values = inputs["input_ids"], inputs["pixel_values"]
    grid = inputs["image_grid_thw"]
    extra = {k: v for k, v in inputs.items() if k not in ("input_ids", "pixel_values", "attention_mask")}

    def vision():
        mx.eval(model.vision_tower(pixel_values.astype(model.vision_tower.patch_embed.proj.weight.dtype), grid))

    def full():
        out = model(input_ids, pixel_values, mask=inputs.get("attention_mask"), **extra)
        logits = getattr(out, "logits", out)
        mx.eval(logits)
        return logits

    yes, no = (processor.tokenizer.encode(o, add_special_tokens=False)[0] for o in ("Yes", "No"))
    last = full()[0, -1]
    p_yes = mx.softmax(mx.stack([last[yes], last[no]]).astype(mx.float32))[0].item()

    n_tok = int(np.prod(np.array(grid)[0]) // ip.merge_size ** 2)
    (v50, v95), (f50, f95) = timed(vision), timed(full)
    print(f"{model_id:44s} {w}x{h} tokens={n_tok:4d}  vision {v50:6.1f}/{v95:6.1f} ms  "
          f"prefill+query {f50:6.1f}/{f95:6.1f} ms  P(Yes|human)={p_yes:.3f}", flush=True)
    del model
    mx.clear_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="image.png")
    parser.add_argument("--models", default="mlx-community/Qwen2-VL-2B-Instruct-bf16,mlx-community/Qwen2-VL-2B-Instruct-4bit")
    parser.add_argument("--budgets", default="448,224")
    args = parser.parse_args()
    image = Image.open(args.image).convert("RGB")
    for model_id in args.models.split(","):
        for budget in (int(b) for b in args.budgets.split(",")):
            try:
                bench(model_id, image, budget)
            except Exception as e:  # keep going across models; report what failed
                print(f"{model_id} @ {budget}: FAILED {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
