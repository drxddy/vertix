"""The runtime loop.

  capture thread: source -> Tier 0 health (every frame) -> sample at `fps` -> 2-frame groups -> queue
  worker thread:  group -> vision encoder (cached per group) -> Tier 1 heads on pooled features
                  -> Tier 2: LM prefill over the last `window_s` of groups, all due questions in one
                     forked batch -> smoothing -> decisions on the bus

Tier 2 prefix, two modes:
  window  re-prefill the last `window_s` of groups every Tier 2 tick (cost grows with the window)
  stream  append each new group once to a running stream (VideoDecisionEngine.stream_*); when it holds
          2 windows' worth of groups it is rebuilt from the last window's cached groups in one forward,
          so questions always see between window_s and 2 * window_s of history

Tier 2 can only change when a new group arrives, so its maximum rate is fps / 2.
Latency is bounded: in realtime mode, when the worker falls behind, the oldest queued groups are
dropped and the window restarts (a window must be contiguous so its timestamps stay truthful),
and Tier 2 is skipped while a backlog exists.
"""
import queue
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from runtime.decisions import Decision
from runtime.heads import ChoiceSmoother, Hysteresis, LinearHead, window_delta


class Runtime:
    def __init__(self, engine, registry, bus, clock, fps=8.0, window_s=2.0, realtime=True,
                 health=None, sink=None, max_backlog=2, scene_change=0.15, min_window_groups=2, mode="stream"):
        self.engine, self.registry, self.bus, self.clock = engine, registry, bus, clock
        self.fps, self.realtime, self.health, self.sink = fps, realtime, health, sink
        self.window_groups = max(1, round(window_s * fps / 2))
        self.min_window_groups = min(min_window_groups, self.window_groups)
        self.scene_change = scene_change
        self.mode = mode
        self.streams = []
        engine.fps, engine._token_cache = fps, {}   # timestamps in the window prompt follow the sampling rate
        self.groups = queue.Queue(maxsize=0 if realtime else max_backlog)
        self.max_backlog = max_backlog
        self.latest_frame = None
        self.stats = {"tick_ms": deque(maxlen=30), "tier2_times": deque(maxlen=30), "dropped_groups": 0, "backlog": 0}
        self._reset_window = False
        self.heads, self.smoothers = {}, {}
        for q in registry.tier(1):
            if Path(q.head).exists():
                self.heads[q.id] = LinearHead(q.head)
            else:
                registry.remove(q.id)
                self._event("skip_question", question_id=q.id, reason=f"missing head {q.head}")

    # ------------------------------------------------------------------ helpers

    def _event(self, kind, **fields):
        if self.sink:
            self.sink.write({"type": "event", "event": kind, "t": self.clock.now(), **fields})

    def _publish(self, d: Decision):
        self.bus.publish(d)
        if self.sink:
            self.sink.decision(d)

    def _smooth(self, q, p):
        s = self.smoothers.get(q.id)
        if s is None:
            s = self.smoothers[q.id] = (ChoiceSmoother(q.options, q.enter, q.ema) if q.kind == "choice"
                                        else Hysteresis(q.enter, q.exit, q.ema))
        return s.update(p)

    # ------------------------------------------------------------------ capture thread

    def capture(self, source, stop):
        next_sample, pending, last_health = None, [], -1.0
        try:
            for frame, t in source:
                if stop.is_set():
                    break
                self.latest_frame = (frame, t)
                if self.health and t - last_health >= 0.1:
                    last_health = t
                    self._publish_health(self.health.check(frame, t), t)
                if next_sample is None or t >= next_sample - 1e-6:
                    # If capture itself lagged, re-anchor instead of sampling a burst
                    next_sample = t + 1.0 / self.fps if next_sample is None or t - next_sample > 1.0 / self.fps \
                        else next_sample + 1.0 / self.fps
                    pending.append((frame, t))
                    if len(pending) == 2:
                        self._enqueue(pending)
                        pending = []
        finally:
            self.groups.put(None)

    def _enqueue(self, group):
        if not self.realtime:
            self.groups.put(group)   # offline: block, process every group deterministically
            return
        while self.groups.qsize() >= self.max_backlog:
            try:
                self.groups.get_nowait()
                self.stats["dropped_groups"] += 1
                self._reset_window = True
                self._event("drop_group")
            except queue.Empty:
                break
        self.groups.put(group)

    def _publish_health(self, flags, t):
        for name, flag in flags.items():
            if not name.startswith("_"):
                self._publish(Decision(f"camera_{name}", "yes" if flag else "no", float(flag), 0, t, t, self.clock.now()
                                       if self.realtime else t, extra={} if name != "occluded" else flags["_stats"]))

    # ------------------------------------------------------------------ worker thread

    def work(self, stop):
        window = deque(maxlen=self.window_groups)
        prev_pooled = None
        while not stop.is_set():
            item = self.groups.get()
            if item is None:
                break
            if self._reset_window:
                window.clear()
                self.streams = []
                self._reset_window = False
            (f0, t0), (f1, t1) = item
            wall0 = time.perf_counter()
            stamp = (lambda: self.clock.now()) if self.realtime else (lambda: t1 + time.perf_counter() - wall0)

            group = self.engine.encode_group([f0, f1], t_start=t0, t_end=t1)
            window.append(group)
            vision_ms = (time.perf_counter() - wall0) * 1e3

            delta = window_delta(window) if len(window) == self.window_groups else None
            for q in self.registry.tier(1):
                head = self.heads.get(q.id)
                if head is None:
                    continue
                if head.features == "window_delta":
                    if delta is None:      # motion heads were trained on full windows only
                        continue
                    x, t_from = delta, window[0].t_start
                elif head.features == "group_meanmax":
                    x, t_from = np.concatenate([group.pooled, group.pooled_max]), t0
                else:
                    x, t_from = group.pooled, t0
                p_raw = head(x)
                answer, p = self._smooth(q, p_raw)
                self._publish(Decision(q.id, answer, p, 1, t_from, t1, stamp(), p_raw=p_raw))

            # After Tier 1, so presence and motion heads never wait on the language model
            append_ms = self._stream_append(group, window) if self.mode == "stream" else 0.0

            changed = prev_pooled is not None and 1 - float(
                group.pooled @ prev_pooled / (np.linalg.norm(group.pooled) * np.linalg.norm(prev_pooled))) > self.scene_change
            prev_pooled = group.pooled

            prefill_ms = decide_ms = 0.0
            n_asked = 0
            backlog = self.groups.qsize()
            self.stats["backlog"] = backlog
            behind = self.realtime and backlog > 0
            history = max((st.n_groups for st in self.streams), default=0) if self.mode == "stream" else len(window)
            if history >= self.min_window_groups and not behind:
                due = self.registry.tier(2) if changed else self.registry.due(2, t1)
                if due:
                    prefill_ms, decide_ms = self._tier2(list(window), due, stamp)
                    n_asked = len(due)
                    self.stats["tier2_times"].append(stamp())

            tick_ms = (time.perf_counter() - wall0) * 1e3
            self.stats["tick_ms"].append(tick_ms)
            if self.sink:
                self.sink.write({"type": "tick", "t_group_end": t1, "t_done": stamp(), "vision_ms": vision_ms,
                                 "append_ms": append_ms, "prefill_ms": prefill_ms, "decide_ms": decide_ms,
                                 "tick_ms": tick_ms, "mode": self.mode, "history_groups": history,
                                 "questions": n_asked, "window_groups": len(window), "backlog": backlog,
                                 "scene_change": changed})

    def _stream_append(self, group, window):
        """Append the group to the stream; once it holds 2 windows of groups, rebuild it from the last
        window's cached groups in one forward (cheaper than feeding a second stream every tick)."""
        w0 = time.perf_counter()
        if self.streams and self.streams[0].n_groups + 1 > 2 * self.window_groups:
            self.streams = []
            stream = self.engine.new_stream()
            self.engine.stream_extend(stream, list(window))   # window already holds this group
            self.streams = [stream]
            self._event("stream_rebuild", groups=len(window))
        else:
            if not self.streams:
                self.streams = [self.engine.new_stream()]
            self.engine.stream_append(self.streams[0], group)
        return (time.perf_counter() - w0) * 1e3

    def _tier2(self, groups, due, stamp):
        w0 = time.perf_counter()
        if self.mode == "stream":
            stream = max(self.streams, key=lambda st: st.n_groups)
            prefix = self.engine.stream_prefix(stream)
            t_start, t_end = stream.t_start, stream.t_end
        else:
            prefix = self.engine.encode_window(groups)
            t_start, t_end = groups[0].t_start, groups[-1].t_end
        w1 = time.perf_counter()
        yesno = [q for q in due if q.kind == "yesno"]
        choice = [q for q in due if q.kind == "choice"]
        results = []
        if yesno:
            probs = np.array(self.engine.ask_yes_no(prefix, [q.text for q in yesno]))
            results += [(q, float(p)) for q, p in zip(yesno, probs)]
        if choice:
            dists = self.engine.ask_choice(prefix, [(q.text, q.options) for q in choice])
            results += list(zip(choice, dists))
        w2 = time.perf_counter()
        decided = stamp()
        for q, p_raw in results:
            answer, p = self._smooth(q, p_raw)
            q.last_asked = t_end
            raw = p_raw if q.kind == "yesno" else float(np.max(p_raw))
            extra = {} if q.kind == "yesno" else {"probs": dict(zip(q.options, map(float, p_raw)))}
            self._publish(Decision(q.id, answer, p, 2, t_start, t_end, decided, p_raw=raw,
                                   depth=prefix.n_layers, extra=extra))
        return (w1 - w0) * 1e3, (w2 - w1) * 1e3

    # ------------------------------------------------------------------ lifecycle

    def start(self, source, stop):
        threads = [threading.Thread(target=self.capture, args=(source, stop), name="capture", daemon=True),
                   threading.Thread(target=self.work, args=(stop,), name="worker", daemon=True)]
        for th in threads:
            th.start()
        return threads
