"""Check Tier 1 heads on the runtime's video-path features, against the cached image-path features.

For each test image of the presence tasks, the runtime computes pooled features with
encode_group([frame]) (the frame duplicated into one temporal group). This compares those
features with the cached image-path features and re-scores the exported heads on them.

With --dump, also writes the video-path pooled features of every presence frame (train and test)
to data/cache/video_pooled_presence.npz, which export_heads.py trains on (its default).

With --dump-train, also writes features for the extra COCO train2017 images listed in
data/head_train/manifest.jsonl (data/build_head_trainset.py) to data/cache/video_pooled_headtrain.npz.

Usage: venv-mlx/bin/python video_eval/validate_heads.py [--dump] [--dump-train]
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.heads import LinearHead  # noqa: E402
from video_engine import VideoDecisionEngine  # noqa: E402


def auroc(y, p):
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    return (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


def dump_video_features(engine, rows):
    images = sorted({r["image"] for r in rows if r["family"] == "presence"})
    groups = [engine.encode_group([np.asarray(Image.open(ROOT / "data" / im).convert("RGB"))]) for im in images]
    out = ROOT / "data/cache/video_pooled_presence.npz"
    np.savez(out, images=np.array(images), pooled=np.stack([g.pooled for g in groups]),
             pooled_max=np.stack([g.pooled_max for g in groups]))
    print(f"wrote {out} ({len(images)} frames)")


def dump_train_features(engine):
    rows = [json.loads(l) for l in open(ROOT / "data/head_train/manifest.jsonl")]
    groups = [engine.encode_group([np.asarray(Image.open(ROOT / "data" / r["image"]).convert("RGB"))]) for r in rows]
    out = ROOT / "data/cache/video_pooled_headtrain.npz"
    np.savez(out, images=np.array([r["image"] for r in rows]), pooled=np.stack([g.pooled for g in groups]),
             pooled_max=np.stack([g.pooled_max for g in groups]))
    print(f"wrote {out} ({len(rows)} frames)")


def main():
    rows = [json.loads(l) for l in open(ROOT / "data/ladder/manifest.jsonl")]
    cache = np.load(ROOT / "data/cache/ladder_vlm__Qwen3.5-2B-bf16_224.npz")
    image_feats, frame_of_row = cache["pooled"][:, 0].astype(np.float32), cache["frame_of_row"]
    engine = VideoDecisionEngine(pixel_budget=224 * 224)
    if "--dump" in sys.argv:
        dump_video_features(engine, rows)
    if "--dump-train" in sys.argv:
        dump_train_features(engine)

    video_feats, video_max = {}, {}
    for name in ("human", "vehicle", "animal"):
        head = LinearHead(ROOT / f"heads/{name}.npz")
        idx = [i for i, r in enumerate(rows) if r["family"] == "presence" and r["subtype"] == name and r["split"] == "test"]
        y = np.array([rows[i]["label"] for i in idx])
        for i in idx:
            if rows[i]["image"] not in video_feats:
                g = engine.encode_group([np.asarray(Image.open(ROOT / "data" / rows[i]["image"]).convert("RGB"))])
                video_feats[rows[i]["image"]] = g.pooled
                video_max[rows[i]["image"]] = np.concatenate([g.pooled, g.pooled_max])
                continue
                frame = np.asarray(Image.open(ROOT / "data" / rows[i]["image"]).convert("RGB"))
                video_feats[rows[i]["image"]] = engine.encode_group([frame]).pooled
        v = np.stack([video_feats[rows[i]["image"]] for i in idx])
        if head.features == "group_meanmax":
            v = np.stack([video_max[rows[i]["image"]] for i in idx])
        p_vid = np.array([head(x) for x in v])
        print(f"{name:8s} n={len(idx)}  [{head.features}]  runtime-path AUROC {auroc(y, p_vid):.3f}")


if __name__ == "__main__":
    main()
