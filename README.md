# vertix

A shared-prefix VLM decision engine for images and live video on Apple Silicon.

A frame (or the last few seconds of video) is encoded once into a prefix state; many yes/no or
multiple-choice questions are then answered in one batch that forks from it, with no text decoding.
The runtime combines three tiers: camera-health checks, linear heads on vision features
(presence and motion), and Qwen3.5-2B reasoning over a streaming video window at about 4 Hz.

Read the article at **https://drxddy.github.io/vertix/**: every measurement, from the first
exactness bug to the live runtime and a real-webcam field test. Its source is `blog/index.html`,
published by `.github/workflows/pages.yml` on every push to `main`.

## Setup

Two environments: PyTorch for training probes and heads, MLX for everything that runs.

```bash
python3.11 -m venv venv && venv/bin/pip install -r requirements-torch.txt
python3.11 -m venv venv-mlx && venv-mlx/bin/pip install -r requirements-mlx.txt
```

## Run

```bash
venv-mlx/bin/python -m runtime.run --source webcam            # live overlay, press q to quit
venv-mlx/bin/python -m runtime.run --source clip.mp4 --no-view --log run.jsonl
venv-mlx/bin/python -m runtime.run --source webcam --ask "Is someone waving at the camera?"
```

Questions and thresholds live in `runtime/questions.yaml`. On macOS, give your terminal camera access
(System Settings → Privacy & Security → Camera) before using the webcam.

## Verify

```bash
venv-mlx/bin/python -m pytest tests/test_video_equivalence.py -q   # video window + streaming exactness
venv-mlx/bin/python mlx_check_equivalence.py --fp32                # image engine exactness
venv-mlx/bin/python mlx_check_equivalence.py --fp32 --wrong-pos    # control: must fail
```

## Layout

| Path | Role |
|---|---|
| `mlx_engine.py` | MLX shared-prefix engine for hybrid Qwen3.5: zero-copy fork, per-layer states, early exit |
| `video_engine.py` | per-group vision cache, window prefill, streaming prefix, yes/no and multiple-choice readouts |
| `runtime/` | sources, scheduler, tiers, heads, decision bus, viewer, CLI |
| `heads/` | exported linear heads (presence and motion) used by the runtime |
| `engine.py`, `check_equivalence.py`, `demo.py` | the original PyTorch engine (Qwen2/3-VL on MPS) |
| `eval.py`, `ladder.py`, `export_heads.py` | presence eval, layer-wise ladder probes, head export |
| `data/` | dataset builders (COCO presence set, decision ladder); the data itself is not committed |
| `video_eval/` | synthetic motion clips, MVBench eval, motion probes, runtime benchmark |
| `results/` | reports quoted in the blog |
