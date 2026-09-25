"""Video reasoning eval: does the model use time, and how much window does it need?

  synthetic  yes/no motion questions on data/video_eval/synthetic (known answers, reversed twins)
  mvbench    multiple-choice MVBench tasks (letter readout)
  report     tables from the saved results

Conditions per clip, each a separate window prefill with all of the clip's questions in one batch:
  full      frames in order (the real input)
  single    the middle frame only (one group): what appearance alone gives
  shuffled  the same groups in a random order: what the model gets without temporal order
temporal gain = full - single. The synthetic set also sweeps window length and sampling rate, and
scores reversal consistency: a clip and its exact reversal must get opposite motion answers.

Usage:
  venv-mlx/bin/python video_eval/eval_video.py synthetic [--limit N]
  venv-mlx/bin/python video_eval/eval_video.py mvbench [--tasks a,b] [--frames 16] [--limit N]
  venv-mlx/bin/python video_eval/eval_video.py report
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SYN = ROOT / "data/video_eval/synthetic"
MVB = ROOT / "data/mvbench"
RESULTS = ROOT / "data/video_eval/results"

MVBENCH_TASKS = ["moving_direction", "moving_count", "moving_attribute", "object_existence", "counterfactual_inference",
                 "action_sequence", "action_prediction", "object_interaction", "action_antonym",
                 "fine_grained_action", "egocentric_navigation"]
SYN_SETTINGS = [  # (condition, window seconds, sampling fps)
    ("full", 2.0, 8), ("full", 1.0, 8), ("full", 4.0, 8), ("full", 2.0, 4), ("single", 2.0, 8), ("shuffled", 2.0, 8)]


def read_frames(path, start=None, end=None):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if start is not None and end is not None and frames:
        a, b = int(start * fps), max(int(start * fps) + 1, int(end * fps))
        frames = frames[a:b] or frames
    return frames, fps


def groups_for(engine, frames, condition, rng):
    """Encode frames into temporal groups for one condition (frames already sampled, even count)."""
    if condition == "single":
        return [engine.encode_group([frames[len(frames) // 2]])]
    groups = [engine.encode_group(frames[i:i + 2]) for i in range(0, len(frames) - 1, 2)]
    if condition == "shuffled":
        order = list(range(len(groups)))
        while len(order) > 1 and order == sorted(order):
            rng.shuffle(order)
        groups = [groups[i] for i in order]
    return groups


# ---------------------------------------------------------------------------- runs

def run_synthetic(engine, limit):
    rows = [json.loads(l) for l in open(SYN / "manifest.jsonl")]
    by_clip = defaultdict(list)
    for r in rows:
        by_clip[r["clip"]].append(r)
    clips = sorted(by_clip)[:limit] if limit else sorted(by_clip)
    rng = random.Random(0)
    out = []
    for n, clip in enumerate(clips):
        frames, src_fps = read_frames(SYN / clip)
        for condition, window, fps in SYN_SETTINGS:
            step = max(1, round(src_fps / fps))
            sampled = frames[::step][-int(window * fps):]
            sampled = sampled[len(sampled) % 2:]                      # whole groups only
            engine.fps = fps
            prefix = engine.encode_window(groups_for(engine, sampled, condition, rng))
            p = np.array(engine.ask_yes_no(prefix, [r["question"] for r in by_clip[clip]]))
            out += [{**r, "condition": condition, "window": window, "fps": fps, "p": float(pi)}
                    for r, pi in zip(by_clip[clip], p)]
        if n % 50 == 0:
            print(f"  synthetic {n}/{len(clips)}", flush=True)
    return out


def index_videos():
    return {p.relative_to(MVB / "video").as_posix(): p for p in (MVB / "video").rglob("*")
            if p.suffix.lower() in (".mp4", ".webm", ".avi", ".mkv", ".gif", ".mov")}


def resolve(index, rel):
    hits = [p for k, p in index.items() if k == rel or k.endswith("/" + rel)]
    return hits[0] if hits else None


def run_mvbench(engine, tasks, n_frames, limit):
    index = index_videos()
    rng = random.Random(0)
    out = []
    for task in tasks:
        items = json.loads((MVB / "json" / f"{task}.json").read_text())[:limit] if limit else \
            json.loads((MVB / "json" / f"{task}.json").read_text())
        missing = 0
        for n, it in enumerate(items):
            path = resolve(index, it["video"])
            if path is None:
                missing += 1
                continue
            frames, src_fps = read_frames(path, it.get("start"), it.get("end"))
            if len(frames) < 2:
                missing += 1
                continue
            idx = np.linspace(0, len(frames) - 1, n_frames).round().astype(int)
            sampled = [frames[i] for i in idx]
            engine.fps = max(0.5, n_frames / (len(frames) / src_fps))   # timestamps follow real time
            answer = it["candidates"].index(it["answer"])
            for condition in ("full", "single", "shuffled"):
                prefix = engine.encode_window(groups_for(engine, sampled, condition, rng))
                probs = engine.ask_choice(prefix, [(it["question"], it["candidates"])])[0]
                out.append({"task": task, "video": it["video"], "condition": condition, "answer": answer,
                            "pred": int(np.argmax(probs)), "n_options": len(it["candidates"])})
            if n % 50 == 0:
                print(f"  {task} {n}/{len(items)}", flush=True)
        if missing:
            print(f"  {task}: {missing} videos missing or unreadable", flush=True)
    return out


# ---------------------------------------------------------------------------- report

def auroc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if y.min() == y.max():
        return float("nan")
    ranks = np.empty(len(p))
    ranks[np.argsort(p)] = np.arange(1, len(p) + 1)
    pos = y == 1
    return (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


def report():
    syn_path, mvb_path = RESULTS / "synthetic.jsonl", RESULTS / "mvbench.jsonl"
    if syn_path.exists():
        rows = [json.loads(l) for l in open(syn_path)]
        settings = sorted({(r["condition"], r["window"], r["fps"]) for r in rows}, key=lambda s: SYN_SETTINGS.index(s))
        qids = sorted({r["question_id"] for r in rows})
        print("\n=== Synthetic motion: AUROC per question (0.5 = chance) ===")
        print(f"{'setting':22s}" + "".join(f"{q:>14s}" for q in qids) + f"{'mean':>8s}")
        for s in settings:
            vals = []
            for q in qids:
                sel = [r for r in rows if (r["condition"], r["window"], r["fps"]) == s and r["question_id"] == q]
                vals.append(auroc([r["label"] for r in sel], [r["p"] for r in sel]))
            print(f"{s[0]:>8s} {s[1]:.0f}s @{s[2]}fps  " + "".join(f"{v:14.3f}" for v in vals) + f"{np.nanmean(vals):8.3f}")

        print("\nReversal consistency (full, 2 s @ 8 fps): share of reversed pairs whose P(yes) moves the right way")
        full = {(r["clip"], r["question_id"]): r for r in rows if (r["condition"], r["window"], r["fps"]) == ("full", 2.0, 8)}
        by_q = defaultdict(list)
        for (clip, q), r in full.items():
            if clip.endswith("_rev.mp4"):
                fwd = full.get((clip.replace("_rev.mp4", ".mp4"), q))
                if fwd and fwd["label"] != r["label"]:
                    by_q[q].append((fwd["p"] > r["p"]) == (fwd["label"] > r["label"]))
        print("  " + "  ".join(f"{q}={np.mean(v):.2f}(n={len(v)})" for q, v in sorted(by_q.items())))

    if mvb_path.exists():
        rows = [json.loads(l) for l in open(mvb_path)]
        print("\n=== MVBench (multiple choice accuracy) ===")
        print(f"{'task':26s}{'n':>5s}{'chance':>8s}{'full':>8s}{'single':>8s}{'shuffled':>9s}{'temporal gain':>15s}")
        for task in [t for t in MVBENCH_TASKS if any(r["task"] == t for r in rows)]:
            acc = {c: np.mean([r["pred"] == r["answer"] for r in rows if r["task"] == task and r["condition"] == c])
                   for c in ("full", "single", "shuffled")}
            sel = [r for r in rows if r["task"] == task and r["condition"] == "full"]
            chance = np.mean([1 / r["n_options"] for r in sel])
            print(f"{task:26s}{len(sel):5d}{chance:8.2f}{acc['full']:8.3f}{acc['single']:8.3f}{acc['shuffled']:9.3f}"
                  f"{acc['full'] - acc['single']:+15.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["synthetic", "mvbench", "report"])
    parser.add_argument("--tasks", default=",".join(MVBENCH_TASKS))
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--budget", type=int, default=224)
    args = parser.parse_args()
    if args.cmd == "report":
        return report()

    from video_engine import VideoDecisionEngine
    engine = VideoDecisionEngine(pixel_budget=args.budget * args.budget)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.cmd == "synthetic":
        rows = run_synthetic(engine, args.limit)
    else:
        rows = run_mvbench(engine, args.tasks.split(","), args.frames, args.limit)
    with open(RESULTS / f"{args.cmd}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    report()


if __name__ == "__main__":
    main()
