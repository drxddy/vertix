"""Decision-task eval on data/eval: per-task quality plus latency, for VLM and non-VLM baselines.

Feature extraction is cached per config in data/cache/, so reports are cheap to regenerate.

  python eval.py --vlm Qwen/Qwen2-VL-2B-Instruct --budgets 448,336,224    # VLM configs
  python eval.py --siglip google/siglip-base-patch16-224                   # image-classifier baseline
  python eval.py --report                                                  # table over everything cached

Scoring methods per config:
  vlm/zeroshot  P(Yes) from the LM head at the decision token (no training)
  vlm/probe     logistic regression on the decision-token hidden state (frozen backbone)
  siglip/probe  logistic regression on the SigLIP image embedding (no language model)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).parent / "data"
IMAGES = DATA / "eval/images"
CACHE = DATA / "cache"
RESULTS = Path(__file__).parent / "results"

TASKS = {
    "human": "Is a human present in the image?",
    "vehicle": "Is there a car, truck, bus, motorcycle, bicycle or train in the image?",
    "animal": "Is there an animal in the image?",
    "traffic_light": "Is there a traffic light in the image?",
    "obstructed": "Is the camera lens obstructed?",
    "lighting_ok": "Is the lighting adequate?",
}
SAFETY = {"human"}          # tasks where a missed positive is the costly error
SMALL = 0.01                # object area fraction below which a positive counts as "small"


def load_manifest():
    rows = [json.loads(l) for l in open(DATA / "eval/manifest.jsonl")]
    labels = np.array([[np.nan if r["labels"][t] is None else r["labels"][t] for t in TASKS] for r in rows])
    areas = np.array([[r["max_area"].get(t, np.nan) for t in TASKS] for r in rows])
    train = np.array([r["split"] == "train" for r in rows])
    return rows, labels, areas, train


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------- feature extraction

def extract_vlm(model_id, budget, rows):
    from engine import VisualDecisionEngine
    engine = VisualDecisionEngine(model_id, pixel_budget=budget * budget)
    queries = list(TASKS.values())
    option_ids = [engine.tokenizer.encode(o, add_special_tokens=False)[0] for o in ("Yes", "No")]
    yes_no = engine.model.lm_head.weight[option_ids].T.detach()

    hidden, zeroshot, t_enc, t_dec, n_tok = [], [], [], [], []
    for i, r in enumerate(rows):
        image = Image.open(IMAGES / r["file"]).convert("RGB")
        inputs = engine.preprocess(image)
        sync(engine.device); t0 = time.perf_counter()
        prefix = engine.encode_frame(inputs=inputs)
        sync(engine.device); t1 = time.perf_counter()
        h = engine.decision_hidden(prefix, queries)
        p = torch.softmax((h @ yes_no).float(), dim=-1)[:, 0]
        sync(engine.device); t2 = time.perf_counter()
        hidden.append(h.float().cpu().numpy().astype(np.float16))
        zeroshot.append(p.cpu().numpy())
        t_enc.append((t1 - t0) * 1e3); t_dec.append((t2 - t1) * 1e3); n_tok.append(prefix.n_visual_tokens)
        if i % 100 == 0:
            print(f"  {i}/{len(rows)}  encode {np.median(t_enc):.0f} ms  decide {np.median(t_dec):.0f} ms", flush=True)
    return dict(hidden=np.stack(hidden), zeroshot=np.stack(zeroshot), t_encode=np.array(t_enc),
                t_decide=np.array(t_dec), visual_tokens=np.array(n_tok))


def extract_siglip(model_id, rows):
    from transformers import AutoModel, AutoImageProcessor
    from engine import default_device
    device = default_device()
    model = AutoModel.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(model_id)
    feats, t_enc = [], []
    for r in rows:
        inputs = processor(images=Image.open(IMAGES / r["file"]).convert("RGB"), return_tensors="pt").to(device)
        sync(device); t0 = time.perf_counter()
        with torch.no_grad():
            f = model.get_image_features(pixel_values=inputs["pixel_values"].to(torch.bfloat16))
        f = getattr(f, "pooler_output", f)
        sync(device); t_enc.append((time.perf_counter() - t0) * 1e3)
        feats.append(f[0].float().cpu().numpy())
    return dict(embedding=np.stack(feats), t_encode=np.array(t_enc))


def cache_path(kind, model_id, budget=None):
    name = model_id.split("/")[-1] + (f"_{budget}" if budget else "")
    return CACHE / f"{kind}__{name}.npz"


# ---------------------------------------------------------------------------- metrics

def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())


def task_metrics(y, p, area):
    pred = p >= 0.5
    pos, neg = y == 1, y == 0
    m = {
        "auroc": roc_auc_score(y, p),
        "bal_acc": 0.5 * (pred[pos].mean() + (~pred[neg]).mean()),
        "fnr": 1 - pred[pos].mean(),
        "ece": ece(y, p),
    }
    small = pos & (area < SMALL)
    if small.sum() >= 10:  # AUROC of small-object positives against all negatives
        keep = small | neg
        m["auroc_small"] = roc_auc_score(y[keep], p[keep])
        m["n_small"] = int(small.sum())
    return m


def probe_scores(x_train, y_train, x_test):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000))
    clf.fit(x_train, y_train)
    return clf.predict_proba(x_test)[:, 1]


def evaluate(name, feats, labels, areas, train):
    """Returns {method: {task: metrics}} scored on the test split."""
    out = {}
    methods = {}
    if "zeroshot" in feats:
        methods["zeroshot"] = lambda t, tr, te: feats["zeroshot"][te, t]
    if "hidden" in feats:
        methods["probe"] = lambda t, tr, te: probe_scores(feats["hidden"][tr, t].astype(np.float32),
                                                          labels[tr, t], feats["hidden"][te, t].astype(np.float32))
    if "embedding" in feats:
        methods["probe"] = lambda t, tr, te: probe_scores(feats["embedding"][tr], labels[tr, t], feats["embedding"][te])
    for method, score in methods.items():
        res = {}
        for t, task in enumerate(TASKS):
            has = ~np.isnan(labels[:, t])
            tr, te = has & train, has & ~train
            res[task] = task_metrics(labels[te, t], score(t, tr, te), areas[te, t])
        out[f"{name}/{method}"] = res
    return out


def print_report(all_results, latency):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "eval.json").write_text(json.dumps({"results": all_results, "latency": latency}, indent=1, default=float))

    print("\nAUROC on test split (threshold-free; 0.5 = chance)")
    header = f"{'config':44s}" + "".join(f"{t[:13]:>14s}" for t in TASKS) + f"{'mean':>8s}"
    print(header)
    for cfg, res in all_results.items():
        vals = [res[t]["auroc"] for t in TASKS]
        print(f"{cfg:44s}" + "".join(f"{v:14.3f}" for v in vals) + f"{np.mean(vals):8.3f}")

    print("\nSmall-object AUROC (positives with max object area < 1% of frame vs. all negatives)")
    for cfg, res in all_results.items():
        cells = [f"{t}={res[t]['auroc_small']:.3f} (n={res[t]['n_small']})" for t in TASKS if "auroc_small" in res[t]]
        print(f"{cfg:44s}  " + "  ".join(cells))

    print("\nOperating point @0.5: balanced accuracy / false-negative rate / calibration error (ECE)")
    for cfg, res in all_results.items():
        cells = [f"{t[:8]} {res[t]['bal_acc']:.2f}/{res[t]['fnr']:.2f}/{res[t]['ece']:.2f}" for t in TASKS]
        print(f"{cfg:44s}  " + "  ".join(cells))

    print("\nLatency per frame (p50 / p95 ms; VLM decide = all 6 queries in one batch)")
    for cfg, lat in latency.items():
        print(f"{cfg:44s}  " + "  ".join(f"{k} {v[0]:.0f}/{v[1]:.0f}" for k, v in lat.items() if k != "tokens")
              + (f"  visual_tokens≈{lat['tokens']:.0f}" if "tokens" in lat else ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm", help="VLM model id to extract")
    parser.add_argument("--budgets", default="448", help="comma-separated square-equivalent pixel budgets")
    parser.add_argument("--siglip", help="SigLIP/CLIP model id to extract")
    parser.add_argument("--report", action="store_true", help="only report on cached features")
    args = parser.parse_args()

    rows, labels, areas, train = load_manifest()
    CACHE.mkdir(parents=True, exist_ok=True)

    todo = []
    if args.vlm:
        todo += [("vlm", args.vlm, int(b)) for b in args.budgets.split(",")]
    if args.siglip:
        todo.append(("siglip", args.siglip, None))
    for kind, model_id, budget in todo:
        path = cache_path(kind, model_id, budget)
        if path.exists():
            print(f"cached: {path.name}")
            continue
        print(f"extracting {path.name} over {len(rows)} images...")
        feats = extract_vlm(model_id, budget, rows) if kind == "vlm" else extract_siglip(model_id, rows)
        np.savez(path, **feats)
        if kind == "vlm":
            torch.mps.empty_cache() if torch.backends.mps.is_available() else None

    all_results, latency = {}, {}
    for path in sorted(CACHE.glob("*.npz")):
        feats = dict(np.load(path))
        name = path.stem.replace("__", ":")
        all_results.update(evaluate(name, feats, labels, areas, train))
        lat = {k.removeprefix("t_"): (np.percentile(v, 50), np.percentile(v, 95)) for k, v in feats.items() if k.startswith("t_")}
        if "visual_tokens" in feats:
            lat["tokens"] = feats["visual_tokens"].mean()
        latency[name] = lat
    print_report(all_results, latency)


if __name__ == "__main__":
    main()
