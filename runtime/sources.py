"""Frame sources. Each yields (rgb_uint8_frame, t) where t is seconds on the runtime clock.

Realtime sources stamp frames with the wall clock at capture, so decision age is meaningful.
A non-realtime file source yields frames as fast as they are consumed and stamps media time,
which makes offline evaluation deterministic.
"""
import time

import cv2


class Clock:
    """Monotonic seconds since the runtime started; shared by every component."""

    def __init__(self):
        self.t0 = time.monotonic()

    def now(self):
        return time.monotonic() - self.t0


class WebcamSource:
    realtime = True

    def __init__(self, clock, index=0, width=None, height=None):
        self.clock, self.cap = clock, cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {index} (check macOS camera permission for the terminal)")
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    def __iter__(self):
        while True:
            ok, bgr = self.cap.read()
            if not ok:
                return
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), self.clock.now()

    def close(self):
        self.cap.release()


class FileSource:
    def __init__(self, clock, path, realtime=True, loop=False):
        self.clock, self.path, self.realtime, self.loop = clock, str(path), realtime, loop
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    def __iter__(self):
        index, start = 0, self.clock.now()
        while True:
            ok, bgr = self.cap.read()
            if not ok:
                if not self.loop:
                    return
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            media_t = index / self.fps
            index += 1
            if self.realtime:
                # Release each frame at its media time, like a camera would
                delay = start + media_t - self.clock.now()
                if delay > 0:
                    time.sleep(delay)
                t = self.clock.now()
            else:
                t = media_t
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), t

    def close(self):
        self.cap.release()


def open_source(spec, clock, realtime=True, loop=False):
    """'webcam' / 'webcam:1' / a camera index / a video path."""
    if spec == "webcam" or spec.startswith("webcam:"):
        return WebcamSource(clock, int(spec.split(":")[1]) if ":" in spec else 0)
    if spec.isdigit():
        return WebcamSource(clock, int(spec))
    return FileSource(clock, spec, realtime=realtime, loop=loop)
