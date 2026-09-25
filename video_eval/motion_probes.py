"""Is motion direction linearly readable from the model even when its zero-shot answer is wrong?

  extract (venv-mlx): for every synthetic clip (full 2 s window @ 8 fps) cache the decision-token
                      hidden state of each of its questions at every layer, plus the per-group pooled
                      vision features (a no-language-model baseline)
  fit     (venv):     per question, logistic probes per layer; train on sources 0-39, test on 40-59
                      (a source's forward and reversed clips always stay on the same side)

Baselines: zero-shot P(yes) from the same run, and a vision-only probe on
[last group - first group, mean group] pooled features.

  export  (venv):     write runtime heads (heads/motion_<question>.npz) for the vision-only probe,
                      trained on sources 0-29 with a temperature fitted on sources 30-39, so the
                      runtime can answer motion questions from window features with no language model

Usage:
  venv-mlx/bin/python video_eval/motion_probes.py extract
  venv/bin/python video_eval/motion_probes.py fit
  venv/bin/python video_eval/motion_probes.py export
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SYN = ROOT / "data/video_eval/synthetic"
CACHE = ROOT / "data/cache/motion_probe_features.npz"
TRAIN_SOURCES = 40


def source_index(clip):
    return int(re.match(r"(?:cam|obj)(\d+)_", clip).group(1))


def extract():
    import mlx.core as mx
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "video_eval"))
    from eval_video import read_frames
    from video_engine import VideoDecisionEngine

    rows = [json.loads(l) for l in open(SYN / "manifest.jsonl")]
    by_clip = defaultdict(list)
    for i, r in enumerate(rows):
        by_clip[r["clip"]].append(i)
    engine = VideoDecisionEngine(pixel_budget=224 * 224, fps=8.0)
    n_states = engine.total_layers + 1
    hidden = np.zeros((len(rows), n_states, engine.model.config.text_config.hidden_size), dtype=np.float16)
    zeroshot = np.zeros(len(rows), dtype=np.float32)
    vision = {}
    for n, (clip, idx) in enumerate(sorted(by_clip.items())):
        frames, src_fps = read_frames(SYN / clip)
        sampled = frames[::max(1, round(src_fps / 8))][-16:]
        groups = [engine.encode_group(sampled[i:i + 2]) for i in range(0, len(sampled) - 1, 2)]
        prefix = engine.encode_window(groups)
        h = engine.decision_hidden(prefix, [rows[i]["question"] for i in idx], all_layers=True)
        p = mx.softmax(engine.yes_no_logits(h[:, -1]).astype(mx.float32), axis=-1)[:, 0]
        mx.eval(h, p)
        hidden[idx] = np.array(h.astype(mx.float32))
        zeroshot[idx] = np.array(p)
        pooled = np.stack([g.pooled for g in groups])
        vision[clip] = np.concatenate([pooled[-1] - pooled[0], pooled.mean(0)])
        if n % 100 == 0:
            print(f"  {n}/{len(by_clip)}", flush=True)
    vis = np.stack([vision[r["clip"]] for r in rows]).astype(np.float32)
    np.savez(CACHE, hidden=hidden, zeroshot=zeroshot, vision=vis)
    print(f"wrote {CACHE}")


def auroc(y, p):
    ranks = np.empty(len(p))
    ranks[np.argsort(p)] = np.arange(1, len(p) + 1)
    pos = y == 1
    return (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


def fit():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows = [json.loads(l) for l in open(SYN / "manifest.jsonl")]
    z = np.load(CACHE)
    hidden, zeroshot, vision = z["hidden"], z["zeroshot"], z["vision"]
    y = np.array([r["label"] for r in rows])
    train_side = np.array([source_index(r["clip"]) < TRAIN_SOURCES for r in rows])
    qids = sorted({r["question_id"] for r in rows})
    layers = list(range(0, hidden.shape[1], 2)) + [hidden.shape[1] - 1]

    def probe(x, tr, te):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000)).fit(x[tr], y[tr])
        return clf.predict_proba(x[te])[:, 1]

    print(f"test = sources {TRAIN_SOURCES}+ (unseen photos / people); AUROC, 0.5 = chance")
    print(f"{'question':15s}{'zero-shot':>10s}{'vision-only':>12s}{'best LM layer':>15s}   by layer " +
          " ".join(f"L{L}" for L in layers))
    results = {}
    for q in qids:
        m = np.array([r["question_id"] == q for r in rows])
        tr, te = m & train_side, m & ~train_side
        per_layer = {L: auroc(y[te], probe(hidden[:, L].astype(np.float32), tr, te)) for L in layers}
        best = max(per_layer, key=per_layer.get)
        vis = auroc(y[te], probe(vision, tr, te))
        zs = auroc(y[te], zeroshot[te])
        results[q] = {"zeroshot": zs, "vision_only": vis, "layers": per_layer}
        print(f"{q:15s}{zs:10.3f}{vis:12.3f}{per_layer[best]:10.3f} (L{best:<2d})   " +
              " ".join(f"{per_layer[L]:.2f}" for L in layers))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/motion_probes.json").write_text(json.dumps(results, indent=1, default=float))


def export():
    from scipy.optimize import minimize_scalar
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rows = [json.loads(l) for l in open(SYN / "manifest.jsonl")]
    vision = np.load(CACHE)["vision"]
    y = np.array([r["label"] for r in rows])
    src = np.array([source_index(r["clip"]) for r in rows])
    out = ROOT / "heads"
    out.mkdir(exist_ok=True)
    for q in sorted({r["question_id"] for r in rows}):
        m = np.array([r["question_id"] == q for r in rows])
        tr, cal, te = m & (src < 30), m & (src >= 30) & (src < TRAIN_SOURCES), m & (src >= TRAIN_SOURCES)
        scaler = StandardScaler().fit(vision[tr])
        clf = LogisticRegression(C=0.05, max_iter=5000).fit(scaler.transform(vision[tr]), y[tr])
        logit = lambda k: scaler.transform(vision[k]) @ clf.coef_[0] + clf.intercept_[0]

        def nll(t, k=cal):
            pr = np.clip(1 / (1 + np.exp(-logit(k) / t)), 1e-7, 1 - 1e-7)
            return -np.mean(y[k] * np.log(pr) + (1 - y[k]) * np.log(1 - pr))
        temp = minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x
        np.savez(out / f"motion_{q}.npz", mean=scaler.mean_.astype(np.float32), scale=scaler.scale_.astype(np.float32),
                 coef=clf.coef_[0].astype(np.float32), intercept=np.float32(clf.intercept_[0]),
                 temperature=np.float32(temp), features=np.array("window_delta"))
        p = 1 / (1 + np.exp(-logit(te) / temp))
        print(f"{q:15s} test AUROC {auroc(y[te], p):.3f}  accuracy@0.5 {((p >= 0.5) == y[te]).mean():.3f}  T={temp:.2f}")


if __name__ == "__main__":
    {"extract": extract, "fit": fit, "export": export}[sys.argv[1]]()
