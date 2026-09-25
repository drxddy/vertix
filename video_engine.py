"""Video extension of the MLX shared-prefix engine: a sliding window of recent frames as the prefix.

Qwen3.5 consumes video as 2-frame temporal groups, each preceded by a "<t.t seconds>" marker, and its
vision encoder attends only within a group. So each group's vision features are computed once
(encode_group) and reused exactly by every window that contains it. A window prefill (encode_window)
then only re-runs the language model over the window's tokens, with timestamps restarting at 0.0 s
as in a normal clip. Queries fork from the window prefix exactly as in mlx_engine.

Streaming (new_stream / stream_append / stream_prefix): instead of re-prefilling the whole window
every tick, each new group is prefilled once into a running state (KV for the full-attention layers,
fixed-size recurrent state for the linear-attention layers). The instruction text, which the template
places after the video, is deferred into every query's suffix, so the result is exactly what a window
covering all groups since the stream started would give. Streams grow, so callers reset them
(see runtime.scheduler: two staggered streams).

Readouts: yes/no (P(Yes) vs No) and multiple choice (softmax over option letters).
"""
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import _timestamped_video_placeholder
from PIL import Image

from mlx_engine import QUERY_SLOT, FramePrefix, MLXDecisionEngine
from prompts import fit_size

VIDEO_INSTRUCTION = "Please answer the following question based on the video:\n"
LETTERS = "ABCDEFGH"


@dataclass
class GroupFeatures:
    """Vision-encoder output for one 2-frame temporal group."""
    features: mx.array      # (tokens, hidden) merged visual tokens, ready to scatter into the LM input
    pooled: np.ndarray      # (hidden,) mean visual feature: the query-free Tier-1 readout
    grid_hw: tuple          # (grid_h, grid_w) in patches
    pooled_max: np.ndarray = None  # (hidden,) max over visual tokens: keeps localized evidence the mean dilutes
    t_start: float = 0.0    # capture time of the first frame (seconds)
    t_end: float = 0.0      # capture time of the second frame


@dataclass
class VideoStream:
    """Running prefix for a growing clip: header + groups appended so far (no instruction yet)."""
    cache: list
    ids: list               # every token id prefilled so far
    n_groups: int = 0
    grid_hw: tuple = None
    next_pos: int = 0       # first M-RoPE text position after the prefilled tokens
    t_start: float = 0.0    # capture time of the first group's first frame
    t_end: float = 0.0      # capture time of the newest group's last frame


class VideoDecisionEngine(MLXDecisionEngine):
    def __init__(self, model_id="mlx-community/Qwen3.5-2B-bf16", pixel_budget=224 * 224, fps=4.0, dtype=None):
        super().__init__(model_id, pixel_budget=pixel_budget, dtype=dtype)
        self.fps = fps
        self.video_proc = self.processor.video_processor
        self.video_token_id = self.model.config.video_token_index
        self.tps = self.video_proc.temporal_patch_size
        self.merge = self.video_proc.merge_size
        self.video_prefix_text, self.video_suffix_template = self._split_video_template()
        self._token_cache = {}
        # Streaming splits the prefix around the video: header (before) and instruction (after)
        wrapped = f"{self.processor.vision_start_token}{self.processor.video_token}{self.processor.vision_end_token}"
        self.stream_header, self.stream_instruction = self.video_prefix_text.split(wrapped)

    # ------------------------------------------------------------------ templates

    def _split_video_template(self):
        messages = [{"role": "user", "content": [
            {"type": "video"}, {"type": "text", "text": VIDEO_INSTRUCTION + QUERY_SLOT}]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prefix, suffix = text.split(QUERY_SLOT)
        return prefix, "{q}" + suffix

    def choice_question(self, question, options):
        lines = [question, "Options:"] + [f"({LETTERS[i]}) {o}" for i, o in enumerate(options)]
        return "\n".join(lines + ["Answer with the option's letter from the given choices directly."])

    def _window_text(self, n_groups, tokens_per_group, suffix=""):
        """The window's prompt text, expanded exactly as the processor expands a video."""
        vs, ve = self.processor.vision_start_token, self.processor.vision_end_token
        wrapped = f"{vs}{self.processor.video_token}{ve}"
        expanded = _timestamped_video_placeholder(n_groups, tokens_per_group, self.tps, self.fps, vs, ve)
        return self.video_prefix_text.replace(wrapped, expanded, 1).replace(
            "<|placeholder|>", self.processor.video_token) + suffix

    def _window_ids(self, n_groups, tokens_per_group):
        key = (n_groups, tokens_per_group, self.fps)   # fps sets the "<t seconds>" markers
        if key not in self._token_cache:
            ids = self.tokenizer.encode(self._window_text(n_groups, tokens_per_group), add_special_tokens=False)
            self._token_cache[key] = mx.array([ids])
        return self._token_cache[key]

    # ------------------------------------------------------------------ vision

    def frame_size(self, frame):
        h, w = frame.shape[:2]
        return fit_size(w, h, budget=self.pixel_budget, multiple=self.grid)

    def resize(self, frame, size=None):
        """uint8 HxWx3 frame -> resized to the grid-aligned budget size (the processor then leaves it as is)."""
        h, w = size or self.frame_size(frame)
        if frame.shape[:2] == (h, w):
            return frame
        return np.asarray(Image.fromarray(frame).resize((w, h), Image.BICUBIC))

    def encode_group(self, frames, t_start=0.0, t_end=0.0) -> GroupFeatures:
        """Vision encoder on one temporal group (2 frames; 1 frame is duplicated, like a still image)."""
        if len(frames) == 1:
            frames = [frames[0], frames[0]]
        size = self.frame_size(frames[0])
        video = np.stack([self.resize(f, size) for f in frames])
        out = self.video_proc(video)
        pixels = mx.array(out["pixel_values_videos"])
        grid = mx.array(out["video_grid_thw"])
        vt = self.model.vision_tower
        feats, _ = vt(pixels.astype(vt.patch_embed.proj.weight.dtype), grid)
        pooled, pooled_max = feats.mean(axis=0), feats.max(axis=0)
        mx.eval(feats, pooled, pooled_max)
        _, gh, gw = out["video_grid_thw"][0].tolist()
        return GroupFeatures(features=feats, pooled=np.array(pooled.astype(mx.float32)),
                             pooled_max=np.array(pooled_max.astype(mx.float32)),
                             grid_hw=(gh, gw), t_start=t_start, t_end=t_end)

    # ------------------------------------------------------------------ window prefix

    def encode_window(self, groups, n_layers=None) -> FramePrefix:
        """LM prefill over a window of cached groups (oldest first), timestamps restarting at 0.0 s."""
        n_layers = n_layers or self.total_layers
        gh, gw = groups[0].grid_hw
        if any(g.grid_hw != (gh, gw) for g in groups):
            raise ValueError("all groups in a window must share one frame size")
        tokens_per_group = gh * gw // self.merge ** 2
        input_ids = self._window_ids(len(groups), tokens_per_group)
        grid = mx.array([[len(groups), gh, gw]])
        position_ids, _ = self.lm.get_rope_index(input_ids, None, grid, None)
        embeds = self.lm.model.embed_tokens(input_ids)
        feats = mx.concatenate([g.features for g in groups], axis=0).astype(embeds.dtype)
        embeds, _ = self.model.merge_input_ids_with_image_features(
            feats, embeds, input_ids, self.model.config.image_token_index, self.video_token_id)
        cache = self.lm.make_cache()
        h, _ = self._run(embeds, cache, position_ids, n_layers, collect=False)
        mx.eval([c.state for c in cache[:n_layers]] + [h])
        return FramePrefix(cache=cache, next_pos=int(position_ids.max().item()) + 1,
                           prefix_len=input_ids.shape[1], n_visual_tokens=len(groups) * tokens_per_group,
                           n_layers=n_layers)

    # ------------------------------------------------------------------ streaming prefix

    def new_stream(self) -> VideoStream:
        return VideoStream(cache=self.lm.make_cache(), ids=[])

    def _group_text(self, k, tokens_per_group):
        """Timestamp marker + vision block for group k of a clip, exactly as the processor renders it."""
        vs, ve = self.processor.vision_start_token, self.processor.vision_end_token
        blocks = _timestamped_video_placeholder(k + 1, tokens_per_group, self.tps, self.fps, vs, ve)
        last = blocks[blocks.rfind(ve, 0, len(blocks) - len(ve)) + len(ve):] if k else blocks
        return last.replace("<|placeholder|>", self.processor.video_token)

    def _stream_positions(self, n_groups, gh, gw):
        """M-RoPE positions of a stream holding n_groups groups. They depend only on the token layout
        (header, markers, grid), never on pixel content, so they are computed once and sliced."""
        key = (n_groups, gh, gw, self.fps)
        if key not in self._token_cache:
            tokens_per_group = gh * gw // self.merge ** 2
            text = self.stream_header + "".join(self._group_text(k, tokens_per_group) for k in range(n_groups))
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            position_ids, _ = self.lm.get_rope_index(mx.array([ids]), None, mx.array([[n_groups, gh, gw]]), None)
            mx.eval(position_ids)
            self._token_cache[key] = (ids, position_ids)
        return self._token_cache[key]

    def stream_extend(self, stream: VideoStream, groups):
        """Prefill one or more groups into the stream in a single forward (header too, if first)."""
        gh, gw = groups[0].grid_hw
        if stream.grid_hw not in (None, (gh, gw)) or any(g.grid_hw != (gh, gw) for g in groups):
            raise ValueError("all groups in a stream must share one frame size")
        ids_all, pos_all = self._stream_positions(stream.n_groups + len(groups), gh, gw)
        start = len(stream.ids)
        new_ids = ids_all[start:]
        ids = mx.array([new_ids])
        embeds = self.lm.model.embed_tokens(ids)
        feats = mx.concatenate([g.features for g in groups], axis=0).astype(embeds.dtype)
        embeds, _ = self.model.merge_input_ids_with_image_features(
            feats, embeds, ids, self.model.config.image_token_index, self.video_token_id)
        h, _ = self._run(embeds, stream.cache, pos_all[:, :, start:], self.total_layers, collect=False)
        mx.eval([c.state for c in stream.cache] + [h])
        if stream.n_groups == 0:
            stream.t_start = groups[0].t_start
        stream.ids = list(ids_all)
        stream.n_groups += len(groups)
        stream.grid_hw = (gh, gw)
        stream.next_pos = int(pos_all.max().item()) + 1
        stream.t_end = groups[-1].t_end

    def stream_append(self, stream: VideoStream, group: GroupFeatures):
        self.stream_extend(stream, [group])

    def stream_prefix(self, stream: VideoStream) -> FramePrefix:
        """Query-ready view of the stream: queries start with the deferred instruction."""
        gh, gw = stream.grid_hw
        return FramePrefix(cache=stream.cache, next_pos=stream.next_pos, prefix_len=len(stream.ids),
                           n_visual_tokens=stream.n_groups * gh * gw // self.merge ** 2,
                           n_layers=self.total_layers, suffix_lead=self.stream_instruction)

    # ------------------------------------------------------------------ queries

    def _tokenize_suffixes(self, queries, lead=""):
        ids = [self.tokenizer.encode(lead + self.video_suffix_template.format(q=q), add_special_tokens=False)
               for q in queries]
        width = max(map(len, ids))
        return mx.array([r + [self.pad_id] * (width - len(r)) for r in ids]), mx.array([len(r) - 1 for r in ids])

    def ask_yes_no(self, prefix, questions):
        """P(Yes) per question, one forked batch. Needs a full-depth prefix (LM-head readout)."""
        return mx.softmax(self.yes_no_logits(self.decision_hidden(prefix, questions)).astype(mx.float32), axis=-1)[:, 0]

    def ask_choice(self, prefix, items):
        """items: [(question, options)]. Returns a list of per-option probability arrays."""
        prompts = [self.choice_question(q, opts) for q, opts in items]
        hidden = self.decision_hidden(prefix, prompts)
        head = self.lm.lm_head.weight if hasattr(self.lm, "lm_head") else self.lm.model.embed_tokens.weight
        letter_ids = mx.array([self.tokenizer.encode(l, add_special_tokens=False)[0] for l in LETTERS])
        logits = (hidden @ head[letter_ids].T).astype(mx.float32)
        return [np.array(mx.softmax(logits[i, :len(opts)])) for i, (_, opts) in enumerate(items)]

    # ------------------------------------------------------------------ reference (for tests)

    def reference_video_hidden(self, frames, query, all_layers=False):
        """Standard single-pass path: the processor patchifies the whole clip, the model embeds it,
        one sequence with the query. Ground truth for window equivalence."""
        size = self.frame_size(frames[0])
        video = np.stack([self.resize(f, size) for f in frames])
        text = self.video_prefix_text + self.video_suffix_template.format(q=query)
        inputs = self.processor(text=[text], videos=[video], fps=self.fps)
        emb = self.model.get_input_embeddings(
            mx.array(inputs["input_ids"]), None,
            pixel_values_videos=mx.array(inputs["pixel_values_videos"]),
            video_grid_thw=mx.array(inputs["video_grid_thw"]))
        h, states = self._run(emb.inputs_embeds, self.lm.make_cache(), emb.position_ids,
                              self.total_layers, collect=all_layers)
        return mx.stack([s[0, -1] for s in states]) if all_layers else h[0, -1]
