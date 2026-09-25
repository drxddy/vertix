"""Diagnose Tier 1 heads on live webcam frames, through exactly the runtime's capture and vision path.

Grabs frames from the camera (skipping the first second while exposure settles), saves them as JPEGs
under data/webcam_probe/ (gitignored), and prints per frame: capture size, model input size, brightness,
P(human) from heads/human.npz, and P(human) with the colour channels swapped (an RGB/BGR mix-up shows
up as the swapped value being much higher).

Usage: venv-mlx/bin/python video_eval/webcam_probe.py [--camera 0] [--frames 6]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.heads import LinearHead  # noqa: E402
from runtime.sources import Clock, WebcamSource  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--frames", type=int, default=6)
    args = parser.parse_args()

    from video_engine import VideoDecisionEngine
    engine = VideoDecisionEngine(pixel_budget=224 * 224, fps=8.0)
    head = LinearHead(ROOT / "heads/human.npz")
    out = ROOT / "data/webcam_probe"
    out.mkdir(parents=True, exist_ok=True)

    source = WebcamSource(Clock(), args.camera)
    print(f"camera {args.camera}: reported {source.cap.get(3):.0f}x{source.cap.get(4):.0f} @ {source.fps:.0f} fps")
    frames, t_first = [], None
    for frame, t in source:
        t_first = t if t_first is None else t_first
        if t - t_first < 1.0:           # let auto-exposure settle
            continue
        frames.append(frame)
        if len(frames) == args.frames:
            break
        time.sleep(0.3)
    source.close()

    print(f"{'frame':8s}{'capture':>12s}{'model input':>13s}{'brightness':>12s}{'R/G/B mean':>18s}{'P(human)':>10s}{'swapped':>9s}")
    for i, f in enumerate(frames):
        Image.fromarray(f).save(out / f"frame_{i}.jpg", quality=90)
        p = head(engine.encode_group([f]).pooled)
        p_swapped = head(engine.encode_group([f[..., ::-1].copy()]).pooled)
        rgb = "/".join(f"{m:.0f}" for m in f.reshape(-1, 3).mean(0))
        print(f"{i:<8d}{f.shape[1]:>6d}x{f.shape[0]:<5d}{str(engine.frame_size(f)):>13s}{f.mean():12.1f}{rgb:>18s}{p:10.3f}{p_swapped:9.3f}")
    print(f"saved {len(frames)} frames to {out}")


if __name__ == "__main__":
    main()
