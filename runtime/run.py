"""Run the live decision runtime.

  venv-mlx/bin/python -m runtime.run --source webcam
  venv-mlx/bin/python -m runtime.run --source clip.mp4 --no-view --log run.jsonl
  venv-mlx/bin/python -m runtime.run --source clip.mp4 --offline --no-view --log run.jsonl   # every group, no drops
  venv-mlx/bin/python -m runtime.run --source webcam --ask "Is someone waving at the camera?"
"""
import argparse
import re
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.decisions import DecisionBus, JsonlSink  # noqa: E402
from runtime.health import HealthMonitor  # noqa: E402
from runtime.questions import Question, QuestionRegistry  # noqa: E402
from runtime.scheduler import Runtime  # noqa: E402
from runtime.sources import Clock, open_source  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="webcam", help="webcam, webcam:<index>, or a video file")
    parser.add_argument("--offline", action="store_true", help="file source: process every frame as fast as possible")
    parser.add_argument("--loop", action="store_true", help="file source: loop forever")
    parser.add_argument("--model", default="mlx-community/Qwen3.5-2B-bf16")
    parser.add_argument("--budget", type=int, default=224, help="per-frame pixel budget (square-equivalent side)")
    parser.add_argument("--fps", type=float, default=8.0, help="model sampling rate; Tier 2 max rate = fps / 2")
    parser.add_argument("--window", type=float, default=2.0, help="seconds of video in each Tier 2 window")
    parser.add_argument("--mode", choices=["stream", "window"], default="stream",
                        help="stream: append each group once to staggered streams (window..2*window of history); "
                             "window: re-prefill the last --window seconds every tick")
    parser.add_argument("--questions", default=str(ROOT / "runtime/questions.yaml"))
    parser.add_argument("--ask", action="append", default=[], help="extra ad-hoc yes/no question (repeatable)")
    parser.add_argument("--log", help="JSONL file for decisions, ticks and events")
    parser.add_argument("--no-view", action="store_true")
    parser.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=None,
                        help="mirror the preview like a selfie camera (default: on for webcams, off for files; "
                             "press m in the window to toggle). The model always sees the unmirrored frame.")
    parser.add_argument("--max-seconds", type=float, help="stop after this much runtime")
    args = parser.parse_args()

    from video_engine import VideoDecisionEngine  # heavy import after arg parsing

    registry = QuestionRegistry.from_yaml(args.questions)
    for q in registry.tier(1):
        q.head = str(ROOT / q.head) if not Path(q.head).is_absolute() else q.head
    for i, text in enumerate(args.ask):
        registry.add(Question(id=f"ask_{i}_" + re.sub(r"\W+", "_", text.lower()).strip("_")[:24], text=text))

    print(f"loading {args.model} ...", flush=True)
    engine = VideoDecisionEngine(args.model, pixel_budget=args.budget * args.budget, fps=args.fps)
    clock, bus, stop = Clock(), DecisionBus(), threading.Event()
    sink = JsonlSink(args.log) if args.log else None
    source = open_source(args.source, clock, realtime=not args.offline, loop=args.loop)
    runtime = Runtime(engine, registry, bus, clock, fps=args.fps, window_s=args.window,
                      realtime=not args.offline, health=HealthMonitor(), sink=sink, mode=args.mode)
    if sink:
        sink.write({"type": "config", **{k: v for k, v in vars(args).items()},
                    "window_groups": runtime.window_groups, "tier1": list(runtime.heads)})
    threads = runtime.start(source, stop)
    print(f"running [{args.mode}]: window {runtime.window_groups} groups ({args.window}s @ {args.fps} fps), "
          f"tier1 heads {list(runtime.heads) or 'none'}, tier2 questions {[q.id for q in registry.tier(2)]}", flush=True)

    t_start = time.monotonic()
    try:
        if args.no_view:
            last_print = 0.0
            while any(th.is_alive() for th in threads):
                time.sleep(0.1)
                if args.max_seconds and time.monotonic() - t_start > args.max_seconds:
                    stop.set()
                if time.monotonic() - last_print > 1.0:
                    last_print = time.monotonic()
                    tier2 = [d for d in bus.latest().values() if d.tier == 2]
                    if tier2:
                        print(" | ".join(f"{d.question_id}={d.answer}({d.p:.2f})" for d in sorted(tier2, key=lambda d: d.question_id)), flush=True)
        else:
            from runtime.viewer import show
            if args.max_seconds:
                threading.Timer(args.max_seconds, stop.set).start()
            is_webcam = args.source.startswith("webcam") or args.source.isdigit()
            show(runtime, bus, stop, threads, mirror=is_webcam if args.mirror is None else args.mirror)
    except KeyboardInterrupt:
        stop.set()
    finally:
        stop.set()
        for th in threads:
            th.join(timeout=5)
        source.close()
        if sink:
            sink.close()


if __name__ == "__main__":
    main()
