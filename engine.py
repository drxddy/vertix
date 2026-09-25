from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.cache_utils import DynamicCache
from qwen_vl_utils import process_vision_info

from prompts import PREFIX_INSTRUCTION, SUFFIX_TEMPLATE, fit_size  # noqa: F401 (re-exported)


@dataclass
class FramePrefix:
    """Shared visual prefix for one frame: its KV cache plus where the suffix positions start."""
    cache: DynamicCache
    prefix_len: int        # number of tokens in the cache
    next_pos: int          # first M-RoPE text position after the prefix (not equal to prefix_len)
    n_visual_tokens: int
    visual_pooled: torch.Tensor | None = None  # (layers+1, hidden) mean over visual tokens, if requested


def default_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class VisualDecisionEngine(nn.Module):
    def __init__(self, model_id="Qwen/Qwen2-VL-2B-Instruct", num_classes=3,
                 device=None, dtype=torch.bfloat16, pixel_budget=448 * 448):
        super().__init__()
        print(f"Loading base model {model_id}...")
        self.device = torch.device(device) if device is not None else default_device()
        self.dtype = dtype
        # A fixed pixel budget keeps the visual token count (and so the prefix length) roughly constant
        self.pixel_budget = pixel_budget
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        ip = self.processor.image_processor
        self.grid = ip.patch_size * ip.merge_size  # resize dims must be multiples of this
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=dtype,
            attn_implementation="sdpa",
        ).to(self.device).eval()

        hidden_size = self.model.config.text_config.hidden_size
        # Untrained classification head (decision hidden state -> num_classes)
        self.decision_head = nn.Linear(hidden_size, num_classes, dtype=dtype).to(self.device)
        print("VisualDecisionEngine initialized successfully.")

    # ------------------------------------------------------------------ inputs

    def image_size(self, image):
        return fit_size(*image.size, budget=self.pixel_budget, multiple=self.grid)

    def _messages(self, image, text):
        h, w = self.image_size(image)
        return [{
            "role": "user",
            "content": [
                {"type": "image", "image": image, "resized_height": h, "resized_width": w},
                {"type": "text", "text": text},
            ],
        }]

    def _prefix_text(self, messages):
        # Render the full user turn, then cut the closing <|im_end|> so queries continue the same turn
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        end = text.rfind("<|im_end|>")
        return text[:end]

    def preprocess(self, image, text=PREFIX_INSTRUCTION):
        """CPU-side preprocessing: chat template, resize, patchify. Returns processor outputs on device."""
        messages = self._messages(image, text)
        image_inputs, _ = process_vision_info(messages)
        return self.processor(
            text=[self._prefix_text(messages)],
            images=image_inputs,
            return_tensors="pt",
        ).to(self.device)

    def _rope_positions(self, inputs):
        # Compute M-RoPE positions explicitly so nothing depends on the model's cached rope_deltas state
        position_ids, _ = self.model.model.get_rope_index(
            inputs["input_ids"],
            mm_token_type_ids=inputs["mm_token_type_ids"],
            image_grid_thw=inputs["image_grid_thw"],
            attention_mask=inputs["attention_mask"],
        )
        return position_ids

    def _tokenize_suffixes(self, queries):
        self.tokenizer.padding_side = "right"
        return self.tokenizer(
            [SUFFIX_TEMPLATE.format(q=q) for q in queries],
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

    # ------------------------------------------------------------------ stages

    @torch.no_grad()
    def encode_frame(self, image=None, inputs=None, pool_layers=False) -> FramePrefix:
        """ViT + language prefill for one frame, run once and shared by every query.

        pool_layers=True also returns the mean visual-token hidden state at every layer: a
        query-free readout of the frame state, used to probe how deep each decision needs to go."""
        if inputs is None:
            inputs = self.preprocess(image)
        position_ids = self._rope_positions(inputs)
        cache = DynamicCache(config=self.model.config)
        out = self.model.model(
            **inputs,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=pool_layers,
        )
        visual = inputs["input_ids"][0] == self.model.config.image_token_id
        pooled = torch.stack([h[0, visual].mean(0) for h in out.hidden_states]) if pool_layers else None
        return FramePrefix(
            cache=cache,
            prefix_len=inputs["input_ids"].shape[1],
            next_pos=int(position_ids.max()) + 1,
            n_visual_tokens=int(visual.sum()),
            visual_pooled=pooled,
        )

    def _expanded_cache(self, prefix: FramePrefix, n: int) -> DynamicCache:
        # A fresh cache object per call so the shared prefix is never mutated and can be reused.
        # expand() is a view, but DynamicLayer.update() concatenates, which materialises n copies of
        # the prefix KV. A prefix-aware attention kernel (Hydragen/DeFT style) would avoid that copy.
        return DynamicCache(
            ddp_cache_data=[
                (layer.keys.expand(n, -1, -1, -1), layer.values.expand(n, -1, -1, -1))
                for layer in prefix.cache.layers
            ],
            config=self.model.config,
        )

    @torch.no_grad()
    def decision_hidden(self, prefix: FramePrefix, queries: list[str], all_layers=False) -> torch.Tensor:
        """Run all query suffixes as one batch against the shared prefix; return decision hidden states.

        Shape (n, hidden), or (n, layers+1, hidden) with all_layers=True (index 0 = embeddings)."""
        n = len(queries)
        suffix = self._tokenize_suffixes(queries)
        seq_len = suffix.input_ids.shape[1]

        # Suffix tokens are text, so all three M-RoPE axes share the same 1D position
        text_pos = torch.arange(prefix.next_pos, prefix.next_pos + seq_len, device=self.device)
        position_ids = text_pos.view(1, 1, -1).expand(3, n, -1)
        attention_mask = torch.cat(
            [torch.ones(n, prefix.prefix_len, dtype=suffix.attention_mask.dtype, device=self.device),
             suffix.attention_mask],
            dim=1,
        )

        out = self.model.model(
            input_ids=suffix.input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=self._expanded_cache(prefix, n),
            use_cache=True,
            output_hidden_states=all_layers,
        )
        last_idx = suffix.attention_mask.sum(dim=1) - 1
        rows = torch.arange(n, device=self.device)
        if all_layers:
            return torch.stack([h[rows, last_idx] for h in out.hidden_states], dim=1)
        return out.last_hidden_state[rows, last_idx]

    @torch.no_grad()
    def decide(self, prefix: FramePrefix, queries: list[str]) -> torch.Tensor:
        logits = self.decision_head(self.decision_hidden(prefix, queries))
        return torch.softmax(logits.float(), dim=-1)

    @torch.no_grad()
    def zero_shot(self, prefix: FramePrefix, queries: list[str], options=("Yes", "No")) -> torch.Tensor:
        """Untrained readout: the LM's own next-token logits for each option's first token at the
        decision position, softmaxed over just those options. Still one forward pass, no decoding."""
        option_ids = [self.tokenizer.encode(o, add_special_tokens=False)[0] for o in options]
        hidden = self.decision_hidden(prefix, queries)
        logits = hidden @ self.model.lm_head.weight[option_ids].T
        return torch.softmax(logits.float(), dim=-1)

    def forward_with_kv_cache(self, image, queries):
        return self.decide(self.encode_frame(image), queries)

    # ------------------------------------------------------------------ baselines / reference

    def _full_inputs(self, image, queries):
        texts, images = [], []
        for q in queries:
            messages = self._messages(image, PREFIX_INSTRUCTION)
            texts.append(self._prefix_text(messages) + SUFFIX_TEMPLATE.format(q=q))
            images.extend(process_vision_info(messages)[0])
        self.tokenizer.padding_side = "right"
        return self.processor(text=texts, images=images, padding=True, return_tensors="pt").to(self.device)

    @torch.no_grad()
    def _full_hidden(self, inputs):
        out = self.model.model(**inputs, position_ids=self._rope_positions(inputs))
        last_idx = inputs["attention_mask"].sum(dim=1) - 1
        return out.last_hidden_state[torch.arange(len(last_idx), device=self.device), last_idx]

    @torch.no_grad()
    def naive_forward(self, image, queries):
        """Baseline: repeat the image once per query and run full sequences, no prefix sharing."""
        hidden = self._full_hidden(self._full_inputs(image, queries))
        return torch.softmax(self.decision_head(hidden).float(), dim=-1)

    @torch.no_grad()
    def reference_hidden(self, image, query):
        """One ordinary prefix+suffix forward pass (batch 1): ground truth for the shared-prefix path."""
        return self._full_hidden(self._full_inputs(image, [query]))[0]
