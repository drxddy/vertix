"""Prompt format and resize rule shared by the torch engine and the MLX bench (no torch imports)."""

# Each query continues the user turn opened by the prefix, then hands over to the assistant.
# The last token of this suffix is the decision token whose hidden state feeds the head.
SUFFIX_TEMPLATE = "{q}<|im_end|>\n<|im_start|>assistant\n"
# Must end on a token boundary: a trailing space would BPE-merge with the query's first word
# in a full sequence (" Is") but not in the split prefix/suffix path, breaking equivalence.
PREFIX_INSTRUCTION = "Please answer the following question based on the image:\n"


def fit_size(width, height, budget=448 * 448, multiple=28):
    """(height, width) keeping the aspect ratio at roughly `budget` pixels, snapped to Qwen's 28px grid."""
    scale = (budget / (width * height)) ** 0.5
    snap = lambda x: max(multiple, round(x * scale / multiple) * multiple)
    return snap(height), snap(width)
