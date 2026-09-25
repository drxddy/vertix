"""Shared-prefix decision engine on MLX for hybrid Qwen3.5 VLMs (Gated DeltaNet + full attention).

Same contract as the torch engine: encode a frame once into a prefix state, then answer N queries in
one batch that forks from it. The prefix state per layer is either
  - a KV cache (the 6 full-attention layers), or
  - a fixed-size recurrent + conv-window state (the 18 linear-attention layers).
Forking uses broadcast views, not copies: the linear-attention state and conv window are replaced
(never written in place) when the suffix runs, and the KV cache reallocates on append, so the shared
prefix is never mutated.

The decoder layers are driven directly rather than through mlx-vlm's forward, which gives
per-layer hidden states and early exit (run only the first `n_layers` layers) for free.
"""
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from mlx_vlm import load
from mlx_vlm.models.cache import ArraysCache, KVCache
from mlx_vlm.models.qwen3_5.language import _create_qwen3_5_attention_mask, _create_qwen3_5_ssm_mask
from mlx_vlm.utils import prepare_inputs
from PIL import Image

from prompts import PREFIX_INSTRUCTION, fit_size

QUERY_SLOT = "\x00QUERY\x00"


@dataclass
class FramePrefix:
    cache: list            # per layer: KVCache or ArraysCache, batch 1
    next_pos: int          # first M-RoPE text position after the prefix
    prefix_len: int
    n_visual_tokens: int
    n_layers: int          # layers the prefix was run through (early exit if < total)
    visual_pooled: np.ndarray | None = None  # (n_layers+1, hidden) mean visual-token state, if requested
    suffix_lead: str = ""  # text every query must start with (a streamed prefix defers its instruction here)


class MLXDecisionEngine:
    def __init__(self, model_id="mlx-community/Qwen3.5-2B-bf16", pixel_budget=224 * 224, dtype=None):
        self.model, self.processor = load(model_id)
        if dtype is not None:
            self.model.set_dtype(dtype)
        self.tokenizer = self.processor.tokenizer
        self.pixel_budget = pixel_budget
        ip = self.processor.image_processor
        self.grid = ip.patch_size * ip.merge_size
        self.lm = self.model.language_model
        self.layers = self.lm.model.layers
        self.total_layers = len(self.layers)
        self.image_token_id = self.model.config.image_token_index
        self.prefix_text, self.suffix_template = self._split_template()
        self.pad_id = self.tokenizer.pad_token_id or 0

    def _split_template(self):
        """Render the real chat template (generation prompt, thinking disabled) around a placeholder
        query, so the prefix/suffix split matches exactly what the model sees in a normal turn."""
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": PREFIX_INSTRUCTION + QUERY_SLOT}]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prefix, suffix = text.split(QUERY_SLOT)
        return prefix, "{q}" + suffix

    # ------------------------------------------------------------------ inputs

    def image_size(self, image):
        return fit_size(*image.size, budget=self.pixel_budget, multiple=self.grid)

    def preprocess(self, image, text=None):
        h, w = self.image_size(image)
        return prepare_inputs(self.processor, images=[image.resize((w, h), Image.BICUBIC)],
                              prompts=[text if text is not None else self.prefix_text])

    def _tokenize_suffixes(self, queries, lead=""):
        ids = [self.tokenizer.encode(lead + self.suffix_template.format(q=q), add_special_tokens=False) for q in queries]
        width = max(map(len, ids))
        padded = [row + [self.pad_id] * (width - len(row)) for row in ids]  # right padding: causal, so harmless
        return mx.array(padded), mx.array([len(row) - 1 for row in ids])

    # ------------------------------------------------------------------ core

    def _run(self, h, cache, position_ids, n_layers, collect):
        """Run the first n_layers decoder layers. Returns the final state, plus every layer's state if
        collect (index 0 = embeddings; the last entry is normed when running the full depth, matching
        the HF hidden_states convention)."""
        fa_mask = _create_qwen3_5_attention_mask(h, cache[self.lm.model.fa_idx])
        ssm_mask = _create_qwen3_5_ssm_mask(h, cache[self.lm.model.ssm_idx])
        position_embeddings = None
        for layer in self.layers:
            if not layer.is_linear:
                if not layer.self_attn.rotary_emb.fused_apply:
                    position_embeddings = layer.self_attn.rotary_emb(h, position_ids)
                break
        states = [h] if collect else None
        for i in range(n_layers):
            layer = self.layers[i]
            h = layer(h, mask=ssm_mask if layer.is_linear else fa_mask, cache=cache[i],
                      position_ids=position_ids, position_embeddings=position_embeddings)
            if collect:
                states.append(h)
        if n_layers == self.total_layers:
            h = self.lm.model.norm(h)
            if collect:
                states[-1] = h
        return h, states

    def encode_frame(self, image=None, inputs=None, n_layers=None, pool_layers=False) -> FramePrefix:
        """Vision encoder + language prefill through n_layers (default all), run once per frame."""
        n_layers = n_layers or self.total_layers
        if inputs is None:
            inputs = self.preprocess(image)
        input_ids = inputs["input_ids"]
        emb = self.model.get_input_embeddings(input_ids, inputs["pixel_values"],
                                              image_grid_thw=inputs["image_grid_thw"],
                                              mask=inputs.get("attention_mask"))
        cache = self.lm.make_cache()
        h, states = self._run(emb.inputs_embeds, cache, emb.position_ids, n_layers, collect=pool_layers)
        visual = np.array(input_ids[0] == self.image_token_id)
        pooled = None
        if pool_layers:
            vis_idx = mx.array(np.nonzero(visual)[0])
            pooled = mx.stack([s[0, vis_idx].mean(axis=0) for s in states])
            mx.eval(pooled)
            pooled = np.array(pooled.astype(mx.float32))
        mx.eval([c.state for c in cache[:n_layers]] + [h])
        return FramePrefix(cache=cache, next_pos=int(emb.position_ids.max().item()) + 1,
                           prefix_len=input_ids.shape[1], n_visual_tokens=int(visual.sum()),
                           n_layers=n_layers, visual_pooled=pooled)

    @staticmethod
    def _fork(cache, n, n_layers):
        """Per-query view of the prefix state: broadcast along batch, no copy."""
        forked = []
        for c in cache[:n_layers]:
            if isinstance(c, KVCache):
                keys, values = c.state
                f = KVCache()
                f.keys = mx.broadcast_to(keys, (n,) + keys.shape[1:])
                f.values = mx.broadcast_to(values, (n,) + values.shape[1:])
                f.offset = c.offset
            else:
                f = ArraysCache(size=len(c.cache))
                f.cache = [None if x is None else mx.broadcast_to(x, (n,) + x.shape[1:]) for x in c.cache]
            forked.append(f)
        return forked + [None] * (len(cache) - n_layers)

    def decision_hidden(self, prefix: FramePrefix, queries, all_layers=False, n_layers=None):
        """All queries as one batch forked from the shared prefix. Returns (n, hidden), or
        (n, n_layers+1, hidden) with all_layers=True."""
        n_layers = min(n_layers or prefix.n_layers, prefix.n_layers)
        n = len(queries)
        ids, last = self._tokenize_suffixes(queries, prefix.suffix_lead)
        seq = ids.shape[1]
        position_ids = mx.broadcast_to((prefix.next_pos + mx.arange(seq))[None, None, :], (3, n, seq))
        h = self.lm.model.embed_tokens(ids)
        h, states = self._run(h, self._fork(prefix.cache, n, n_layers), position_ids, n_layers, collect=all_layers)
        rows = mx.arange(n)
        if all_layers:
            return mx.stack([s[rows, last] for s in states], axis=1)
        return h[rows, last]

    def yes_no_logits(self, hidden):
        """Zero-shot readout: LM-head logits for 'Yes' vs 'No' on full-depth (normed) decision states."""
        ids = [self.tokenizer.encode(o, add_special_tokens=False)[0] for o in ("Yes", "No")]
        head = self.lm.lm_head.weight if hasattr(self.lm, "lm_head") else self.lm.model.embed_tokens.weight
        return hidden @ head[mx.array(ids)].T

    def zero_shot(self, prefix, queries):
        return mx.softmax(self.yes_no_logits(self.decision_hidden(prefix, queries)).astype(mx.float32), axis=-1)[:, 0]

    # ------------------------------------------------------------------ reference

    def reference_hidden(self, image, query, all_layers=False):
        """One ordinary prefix+query sequence (batch 1), no forking: ground truth for equivalence."""
        text = self.prefix_text + self.suffix_template.format(q=query)
        inputs = self.preprocess(image, text=text)
        emb = self.model.get_input_embeddings(inputs["input_ids"], inputs["pixel_values"],
                                              image_grid_thw=inputs["image_grid_thw"],
                                              mask=inputs.get("attention_mask"))
        h, states = self._run(emb.inputs_embeds, self.lm.make_cache(), emb.position_ids,
                              self.total_layers, collect=all_layers)
        return mx.stack([s[0, -1] for s in states]) if all_layers else h[0, -1]
