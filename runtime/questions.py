"""Question registry: what the runtime answers, how, and how often.

Questions come from a YAML file and can also be added while running (ad-hoc questions). Tier 1
questions need a trained head (heads/*.npz) over pooled vision features; Tier 2 questions are asked
of the VLM over the current window, batched per tick.
"""
import threading
from dataclasses import dataclass, field

import yaml


@dataclass
class Question:
    id: str
    text: str = ""
    kind: str = "yesno"         # "yesno" | "choice"
    options: list = field(default_factory=list)
    tier: int = 2
    rate_hz: float = 4.0        # max answer rate; the effective rate is also capped by new-group arrival
    head: str = "zeroshot"      # "zeroshot" (LM head) or a path to an exported linear head
    enter: float = 0.65         # smoothed p at or above which the state becomes "yes"
    exit: float = 0.35          # smoothed p at or below which the state becomes "no"
    ema: float = 0.5            # weight of the newest observation in the smoothed p
    last_asked: float = float("-inf")   # video time (newest frame) of the window last answered

    def due(self, t_video):
        # Measured on video time, not decision time: a decision lands ~one tick after its frames,
        # so comparing against decision time would skip every other group at full rate. The 50 ms
        # slack absorbs capture jitter (group gaps of 242-258 ms at a nominal 250 ms).
        return t_video - self.last_asked >= 1.0 / self.rate_hz - 0.05


class QuestionRegistry:
    def __init__(self, questions=()):
        self._lock = threading.Lock()
        self._questions = {q.id: q for q in questions}

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            spec = yaml.safe_load(f) or {}
        return cls(Question(**q) for q in spec.get("questions", []))

    def add(self, question: Question):
        with self._lock:
            self._questions[question.id] = question

    def remove(self, qid):
        with self._lock:
            self._questions.pop(qid, None)

    def tier(self, tier):
        with self._lock:
            return [q for q in self._questions.values() if q.tier == tier]

    def due(self, tier, now):
        return [q for q in self.tier(tier) if q.due(now)]
