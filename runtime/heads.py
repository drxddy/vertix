"""Tier 1 linear heads and per-question temporal smoothing.

A head is an exported logistic regression (see export_heads.py): standardize, dot, add bias,
divide by a fitted temperature, sigmoid. Smoothing is an EMA followed by hysteresis, so an answer
only flips when the evidence clearly crosses the other threshold; until the first crossing the
answer is "uncertain".
"""
import numpy as np


class LinearHead:
    def __init__(self, path):
        z = np.load(path)
        self.mean, self.scale = z["mean"], z["scale"]
        self.coef, self.intercept = z["coef"], float(z["intercept"])
        self.temperature = float(z["temperature"]) if "temperature" in z else 1.0
        # "group": mean-pooled features of the newest group; "group_meanmax": [mean, max] over its visual
        # tokens; "window_delta": [last - first, mean] pooled features over the window (motion heads)
        self.features = str(z["features"]) if "features" in z else "group"

    def __call__(self, x):
        logit = ((x - self.mean) / self.scale) @ self.coef + self.intercept
        return float(0.5 * (1.0 + np.tanh(logit / self.temperature / 2)))   # sigmoid, overflow-safe


def window_delta(groups):
    pooled = np.stack([g.pooled for g in groups])
    return np.concatenate([pooled[-1] - pooled[0], pooled.mean(0)])


class Hysteresis:
    def __init__(self, enter, exit, ema):
        self.enter, self.exit, self.ema = enter, exit, ema
        self.p, self.state = None, "uncertain"

    def update(self, p):
        self.p = p if self.p is None else self.ema * p + (1 - self.ema) * self.p
        if self.p >= self.enter:
            self.state = "yes"
        elif self.p <= self.exit:
            self.state = "no"
        return self.state, self.p


class ChoiceSmoother:
    """EMA over the option distribution; the answer is the argmax once it clears `enter`."""

    def __init__(self, options, enter, ema):
        self.options, self.enter, self.ema = options, enter, ema
        self.p = None

    def update(self, probs):
        probs = np.asarray(probs, dtype=np.float64)
        self.p = probs if self.p is None else self.ema * probs + (1 - self.ema) * self.p
        best = int(self.p.argmax())
        answer = self.options[best] if self.p[best] >= self.enter else "uncertain"
        return answer, float(self.p[best])
