"""Tier 0: camera health from cheap per-frame image statistics (CPU, every frame).

Thresholds are deliberately conservative starting points on a 160px-wide grayscale copy;
tune them on your camera. Signals:
  too_dark / too_bright  mean luminance outside [dark, bright]
  blurry                 variance of the Laplacian below `blur`
  frozen                 no pixel change for `frozen_s` seconds (stuck driver, paused stream)
  occluded               large fraction of flat 16px blocks (lens covered, smudge, finger)
"""
import cv2
import numpy as np


class HealthMonitor:
    def __init__(self, dark=40, bright=225, blur=30.0, frozen_s=1.0, flat_std=4.0, occluded_frac=0.45):
        self.dark, self.bright, self.blur = dark, bright, blur
        self.frozen_s, self.flat_std, self.occluded_frac = frozen_s, flat_std, occluded_frac
        self._prev, self._last_change = None, None

    def check(self, rgb, t):
        h, w = rgb.shape[:2]
        small = cv2.resize(rgb, (160, max(1, int(160 * h / w))), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        mean = float(gray.mean())
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if self._prev is None or np.abs(gray.astype(np.int16) - self._prev).mean() > 0.5:
            self._last_change = t
        self._prev = gray.astype(np.int16)

        bh, bw = gray.shape[0] // 16 * 16, gray.shape[1] // 16 * 16
        blocks = gray[:bh, :bw].reshape(bh // 16, 16, bw // 16, 16).std(axis=(1, 3))
        flat = float((blocks < self.flat_std).mean())

        return {
            "too_dark": mean < self.dark,
            "too_bright": mean > self.bright,
            "blurry": sharpness < self.blur,
            "frozen": t - self._last_change >= self.frozen_s,
            "occluded": flat >= self.occluded_frac,
            "_stats": {"mean": mean, "sharpness": sharpness, "flat_frac": flat},
        }
