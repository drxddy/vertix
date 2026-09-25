"""End-to-end runtime benchmark.

  make-replay  concatenate synthetic motion clips (motion + static segments) into one video, with a
               sidecar of segment labels; replayed through the runtime in real time
  analyze      read a runtime JSONL log: per-stage latency, decision age, Tier 2 rate, drops,
               GPU busy fraction, stability (answer flips on static segments), and end-to-end
               accuracy of the smoothed answers against segment labels (including lag)

  venv-mlx/bin/python video_eval/runtime_bench.py make-replay --out data/video_eval/replay.mp4
  venv-mlx/bin/python -m runtime.run --source data/video_eval/replay.mp4 --questions video_eval/replay_questions.yaml \
      --no-view --log results/replay_run.jsonl
  venv-mlx/bin/python video_eval/runtime_bench.py analyze results/replay_run.jsonl --segments data/video_eval/replay.json
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SYN = ROOT / "data/video_eval/synthetic"


def make_replay(out, n_segments, seed, min_source):
    rows = [json.loads(l) for l in open(SYN / "manifest.jsonl")]
    # Only held-out sources, so heads trained on synthetic sources < min_source are scored fairly
    rows = [r for r in rows if int(r["clip"].split("_")[0][3:]) >= min_source]
    labels = defaultdict(dict)
    for r in rows:
        labels[r["clip"]][r["question_id"]] = r["label"]
    clips = sorted(labels)
    rng = random.Random(seed)
    # Stratified: every motion kind appears equally often, alternating with static segments, so
    # each question has positives and both responsiveness and stability are exercised
    kinds = defaultdict(list)
    for c in clips:
        kinds[c.split("_", 1)[1].removesuffix(".mp4")].append(c)
    motion_kinds = sorted(k for k in kinds if "static" not in k)
    static = [c for k, cs in kinds.items() if "static" in k for c in cs]
    order = [k for _ in range(max(1, n_segments // (2 * len(motion_kinds)))) for k in motion_kinds]
    rng.shuffle(order)
    chosen = []
    for k in order:
        chosen += [rng.choice(kinds[k]), rng.choice(static)]

    writer, t, segments = None, 0.0, []
    for clip in chosen:
        cap = cv2.VideoCapture(str(SYN / clip))
        fps = cap.get(cv2.CAP_PROP_FPS) or 8.0
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if writer is None:
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
            writer.write(frame)
            n += 1
        segments.append({"clip": clip, "t0": t, "t1": t + n / fps, "labels": labels[clip]})
        t += n / fps
    writer.release()
    Path(out).with_suffix(".json").write_text(json.dumps(segments, indent=1))
    print(f"wrote {out} ({t:.0f} s, {len(segments)} segments) and {Path(out).with_suffix('.json')}")


def balanced_acc(pairs):
    """Mean per-class recall over (prediction, label) pairs: always answering "no" scores 0.5."""
    recalls = [np.mean([p == l for p, l in pairs if l == c]) for c in (True, False) if any(l == c for _, l in pairs)]
    return float(np.mean(recalls)) if recalls else float("nan")


def pct(v, q):
    return float(np.percentile(v, q)) if len(v) else float("nan")


def analyze(log, segments_path, settle):
    records = [json.loads(l) for l in open(log)]
    config = next((r for r in records if r["type"] == "config"), {})
    ticks = [r for r in records if r["type"] == "tick"]
    decisions = [r for r in records if r["type"] == "decision"]
    events = [r for r in records if r["type"] == "event"]
    duration = ticks[-1]["t_done"] - ticks[0]["t_group_end"] if len(ticks) > 1 else float("nan")

    print(f"mode {config.get('mode', 'window')}  source {config.get('source')}  fps {config.get('fps')}  window {config.get('window')} s "
          f"({config.get('window_groups')} groups)  realtime {not config.get('offline', False)}")
    print(f"\n{len(ticks)} ticks over {duration:.1f} s  |  dropped groups "
          f"{sum(e['event'] == 'drop_group' for e in events)}  |  GPU busy ≈ "
          f"{sum(t['tick_ms'] for t in ticks) / 1e3 / duration:.0%}")
    t2 = [t for t in ticks if t["questions"]]
    print(f"Tier 2 ran on {len(t2)}/{len(ticks)} ticks  ({len(t2) / duration:.2f} Hz)")
    print(f"{'stage':12s}{'p50 ms':>9s}{'p95 ms':>9s}")
    for k in ("vision_ms", "append_ms", "prefill_ms", "decide_ms", "tick_ms"):
        v = [t.get(k, 0.0) for t in (t2 if k in ("prefill_ms", "decide_ms") else ticks)]
        print(f"{k:12s}{pct(v, 50):9.1f}{pct(v, 95):9.1f}")

    print("\nDecision age (t_decided - newest frame used), ms")
    for tier in sorted({d["tier"] for d in decisions}):
        ages = [d["age_ms"] for d in decisions if d["tier"] == tier]
        print(f"  tier {tier}: p50 {pct(ages, 50):6.0f}  p95 {pct(ages, 95):6.0f}  (n={len(ages)})")

    if not segments_path:
        return
    segments = json.loads(Path(segments_path).read_text())
    by_q = defaultdict(list)
    for d in decisions:
        if d["tier"] in (1, 2):
            by_q[d["question_id"]].append(d)
    label_id = lambda q: q.removesuffix("_head")   # Tier 1 motion heads are scored on the same labels

    def segment_at(t):
        return next((s for s in segments if s["t0"] <= t < s["t1"]), None)

    print(f"\nEnd-to-end accuracy of Tier 1/2 answers vs the segment being shown when the answer was produced "
          f"(first {settle:.1f} s of each segment excluded while the window fills with it)")
    print(f"{'question':18s}{'n':>6s}{'pos':>5s}{'bal.acc':>9s}{'raw@0.5':>9s}{'uncertain':>11s}{'flips/min static':>18s}")
    for q, ds in sorted(by_q.items()):
        if not any(label_id(q) in s["labels"] for s in segments):
            continue
        scored, raw, unc, flips, static_time = [], [], 0, 0, 0.0
        for d in ds:
            s = segment_at(d["t_end"])
            if s is None or label_id(q) not in s["labels"] or d["t_end"] - s["t0"] < settle:
                continue
            label = bool(s["labels"][label_id(q)])
            raw.append((d["p_raw"] >= 0.5, label))      # unsmoothed model output: model quality alone
            if d["answer"] == "uncertain":
                unc += 1
                continue
            scored.append((d["answer"] == "yes", label))
        for s in segments:
            if "static" in s["clip"]:
                answers = [d["answer"] for d in ds if s["t0"] + settle <= d["t_end"] < s["t1"]]
                flips += sum(a != b for a, b in zip(answers, answers[1:]))
                static_time += max(0.0, s["t1"] - s["t0"] - settle)
        n = len(scored) + unc
        n_pos = sum(lab for _, lab in raw)
        print(f"{q:18s}{n:6d}{n_pos:5d}{balanced_acc(scored):9.2f}{balanced_acc(raw):9.2f}{unc / max(n, 1):11.2f}"
              f"{flips / max(static_time / 60, 1e-9):18.1f}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-replay")
    m.add_argument("--out", default=str(ROOT / "data/video_eval/replay.mp4"))
    m.add_argument("--segments", type=int, default=32, help="total segments (half motion, half static)")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--min-source", type=int, default=40, help="use only synthetic sources >= this (held out)")
    a = sub.add_parser("analyze")
    a.add_argument("log")
    a.add_argument("--segments")
    a.add_argument("--settle", type=float, default=2.0, help="seconds after a segment starts before scoring")
    args = parser.parse_args()
    if args.cmd == "make-replay":
        make_replay(args.out, args.segments, args.seed, args.min_source)
    else:
        analyze(args.log, args.segments, args.settle)


if __name__ == "__main__":
    main()
