"""Exactness of the video window engine against mlx-vlm's standard single-pass video path.

Runs in float32: the logic is exact there for every batch shape. In bf16, Metal picks different kernels
for a 1-query batch than for larger batches, which shifts results by ~1% relative (not a logic error).

Run: venv-mlx/bin/python -m pytest tests/test_video_equivalence.py -q
"""
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from video_engine import VideoDecisionEngine  # noqa: E402

QUERIES = ["Is the camera moving closer?", "Is a person visible in the video, and is she wearing a hat?", "Any hazards?"]


@pytest.fixture(scope="module")
def engine():
    return VideoDecisionEngine(dtype=mx.float32)


@pytest.fixture(scope="module")
def frames():
    """10 frames of a slow zoom into image.png: enough for overlapping 4-group windows."""
    base = np.asarray(Image.open(Path(__file__).resolve().parents[1] / "image.png").convert("RGB"))
    H, W = base.shape[:2]
    out = []
    for i in range(10):
        s = 1 + 0.06 * i
        ch, cw = int(H / s), int(W / s)
        y, x = (H - ch) // 2, (W - cw) // 2
        out.append(np.asarray(Image.fromarray(base[y:y + ch, x:x + cw]).resize((W, H))))
    return out


def as_np(x):
    return np.array(x.astype(mx.float32))


def assert_same(a, b, msg=""):
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-5 * np.abs(b).max(), err_msg=msg)


def groups_of(engine, frames):
    return [engine.encode_group(frames[i:i + 2]) for i in range(0, len(frames), 2)]


def test_window_matches_native_video_path(engine, frames):
    clip = frames[:8]
    prefix = engine.encode_window(groups_of(engine, clip))
    shared = as_np(engine.decision_hidden(prefix, QUERIES, all_layers=True))
    for i, q in enumerate(QUERIES):
        ref = as_np(engine.reference_video_hidden(clip, q, all_layers=True))
        assert_same(shared[i], ref, q)


def test_overlapping_windows_reuse_cached_groups(engine, frames):
    groups = groups_of(engine, frames)             # 5 groups, each encoded exactly once
    later = engine.encode_window(groups[1:5])      # window shifted by one group
    got = as_np(engine.decision_hidden(later, QUERIES[:1]))[0]
    ref = as_np(engine.reference_video_hidden(frames[2:10], QUERIES[0]))
    assert_same(got, ref)


def test_early_exit_equals_layer_k(engine, frames):
    clip, k = frames[:8], 16
    groups = groups_of(engine, clip)
    early = as_np(engine.decision_hidden(engine.encode_window(groups, n_layers=k), QUERIES[:1]))[0]
    ref = as_np(engine.reference_video_hidden(clip, QUERIES[0], all_layers=True))[k]
    assert_same(early, ref)


def test_prefix_state_unchanged_after_queries(engine, frames):
    prefix = engine.encode_window(groups_of(engine, frames[:8]))
    snap = lambda: [as_np(x) for c in prefix.cache for x in (c.cache if hasattr(c, "cache") else c.state)]
    before = snap()
    engine.decision_hidden(prefix, QUERIES)
    for a, b in zip(before, snap()):
        np.testing.assert_array_equal(a, b)


def test_controls_detect_order_and_timestamp_errors(engine, frames):
    """If the window ignored group order or timestamps, these would pass silently."""
    clip = frames[:8]
    groups = groups_of(engine, clip)
    ref = as_np(engine.reference_video_hidden(clip, QUERIES[0]))

    shuffled = as_np(engine.decision_hidden(engine.encode_window(groups[::-1]), QUERIES[:1]))[0]
    assert np.abs(shuffled - ref).max() > 1e-2

    fps = engine.fps
    try:
        engine.fps, engine._token_cache = fps / 2, {}   # wrong timestamps in the markers
        wrong_time = as_np(engine.decision_hidden(engine.encode_window(groups), QUERIES[:1]))[0]
    finally:
        engine.fps, engine._token_cache = fps, {}
    assert np.abs(wrong_time - ref).max() > 1e-3


def test_stream_matches_window(engine, frames):
    """Appending groups one at a time (instruction deferred into the query) == one window prefill."""
    groups = groups_of(engine, frames)
    stream = engine.new_stream()
    for g in groups:
        engine.stream_append(stream, g)
    got = as_np(engine.decision_hidden(engine.stream_prefix(stream), QUERIES, all_layers=True))
    ref = as_np(engine.decision_hidden(engine.encode_window(groups), QUERIES, all_layers=True))
    for i, q in enumerate(QUERIES):
        assert_same(got[i], ref[i], q)
    # and against the native single-pass path on the same frames
    native = as_np(engine.reference_video_hidden(frames, QUERIES[0]))
    assert_same(as_np(engine.decision_hidden(engine.stream_prefix(stream), QUERIES[:1]))[0], native)
