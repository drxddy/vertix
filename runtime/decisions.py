"""The decision contract, an in-process bus, and a JSONL sink.

Every answer says which frames it is about (t_start..t_end) and when it was produced (t_decided),
so consumers can reject stale answers: age_ms = t_decided - t_end.
"""
import json
import threading
from dataclasses import asdict, dataclass, field


@dataclass
class Decision:
    question_id: str
    answer: str             # "yes" | "no" | "uncertain" | an option string
    p: float                # smoothed P(yes), or the chosen option's probability
    tier: int               # 0 camera health, 1 presence heads, 2 VLM reasoning
    t_start: float          # earliest frame the answer is based on (runtime clock, s)
    t_end: float            # latest frame the answer is based on
    t_decided: float
    p_raw: float = None     # unsmoothed model output
    depth: int = None       # LM layers used (tier 2)
    extra: dict = field(default_factory=dict)

    @property
    def age_ms(self):
        return (self.t_decided - self.t_end) * 1e3


class DecisionBus:
    def __init__(self):
        self._lock = threading.Lock()
        self._latest = {}
        self._subscribers = []

    def subscribe(self, fn):
        self._subscribers.append(fn)

    def publish(self, decision: Decision):
        with self._lock:
            self._latest[decision.question_id] = decision
        for fn in self._subscribers:
            fn(decision)

    def latest(self):
        with self._lock:
            return dict(self._latest)


class JsonlSink:
    """Writes {"type": "decision"|"tick"|"event", ...} lines; thread-safe."""

    def __init__(self, path):
        self._f = open(path, "w")
        self._lock = threading.Lock()

    def decision(self, d: Decision):
        self.write({"type": "decision", **asdict(d), "age_ms": d.age_ms})

    def write(self, record):
        with self._lock:
            self._f.write(json.dumps(record, default=float) + "\n")

    def close(self):
        with self._lock:
            self._f.close()
