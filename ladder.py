"""Decision-ladder study: how deep into the VLM does each kind of decision become readable,
and does the VLM beat fast classifiers where it should (spatial / compositional reasoning)?

  python ladder.py extract-vlm --model Qwen/Qwen3-VL-2B-Instruct --budget 224
  python ladder.py extract-siglip --model google/siglip-so400m-patch14-384
  python ladder.py report

Readouts compared per family (presence subtasks are probed separately, as in eval.py):
  vlm zeroshot        P(Yes) from the LM head at the decision token, no training
  vlm L<k>            logistic probe on the decision-token hidden state after layer k
  vlm pooled L<k>     logistic probe on the mean visual-token state after layer k: no query pass at
                      all (fixed tasks only, since it cannot see the question)
  siglip image        probe on the image embedding alone (cannot see the question)
  siglip text         probe on the question embedding alone: the language-prior floor
  siglip fusion       probe on [image, text, image*text]: the fast dual-encoder competitor
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
CACHE = DATA / "cache"
RESULTS = Path(__file__).parent / "results"
MAX_BATCH = 32  # queries per suffix batch; GQA frames carry up to ~30 questions


def load_manifest():
    rows = [json.loads(l) for l in open(DATA / "ladder/manifest.jsonl")]
    frames = {}
    for i, r in enumerate(rows):
        frames.setdefault(r["group"], []).append(i)
    return rows, frames


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------- extraction

def extract_vlm(model_id, budget):
    from engine import VisualDecisionEngine
    rows, frames = load_manifest()
    engine = VisualDecisionEngine(model_id, pixel_budget=budget * budget)
    option_ids = [engine.tokenizer.encode(o, add_special_tokens=False)[0] for o in ("Yes", "No")]
    yes_no = engine.model.lm_head.weight[option_ids].T.detach()

    n_layers = engine.model.config.text_config.num_hidden_layers + 1
    hidden_size = engine.model.config.text_config.hidden_size
    hidden = np.zeros((len(rows), n_layers, hidden_size), dtype=np.float16)
    zeroshot = np.zeros(len(rows), dtype=np.float32)
    frame_ids = list(frames)
    pooled = np.zeros((len(frame_ids), n_layers, hidden_size), dtype=np.float16)
    t_enc, t_dec = [], []

    for f, (group, idx) in enumerate(frames.items()):
        image = Image.open(DATA / rows[idx[0]]["image"]).convert("RGB")
        inputs = engine.preprocess(image)
        sync(engine.device); t0 = time.perf_counter()
        prefix = engine.encode_frame(inputs=inputs, pool_layers=True)
        sync(engine.device); t1 = time.perf_counter()
        for start in range(0, len(idx), MAX_BATCH):
            chunk = idx[start:start + MAX_BATCH]
            h = engine.decision_hidden(prefix, [rows[i]["question"] for i in chunk], all_layers=True)
            zeroshot[chunk] = torch.softmax((h[:, -1] @ yes_no).float(), dim=-1)[:, 0].cpu().numpy()
            hidden[chunk] = h.float().cpu().numpy()
        sync(engine.device); t2 = time.perf_counter()
        pooled[f] = prefix.visual_pooled.float().cpu().numpy()
        t_enc.append((t1 - t0) * 1e3); t_dec.append(((t2 - t1) * 1e3, len(idx)))
        if f % 250 == 0:
            print(f"  frame {f}/{len(frames)}  encode {np.median(t_enc):.0f} ms", flush=True)

    frame_of_row = np.zeros(len(rows), dtype=np.int32)
    for f, idx in enumerate(frames.values()):
        frame_of_row[idx] = f
    np.savez(CACHE / f"ladder_vlm__{model_id.split('/')[-1]}_{budget}.npz", hidden=hidden, zeroshot=zeroshot,
             pooled=pooled, frame_of_row=frame_of_row, t_encode=np.array(t_enc), t_decide=np.array(t_dec))


def extract_siglip(model_id):
    from transformers import AutoModel, AutoProcessor
    from engine import default_device
    rows, frames = load_manifest()
    device = default_device()
    model = AutoModel.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id)

    def pooled(out):
        return getattr(out, "pooler_output", out).float().cpu().numpy()

    image_emb = np.zeros((len(rows), model.config.vision_config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for group, idx in frames.items():
            px = processor(images=Image.open(DATA / rows[idx[0]]["image"]).convert("RGB"), return_tensors="pt")
            image_emb[idx] = pooled(model.get_image_features(pixel_values=px["pixel_values"].to(device, torch.bfloat16)))[0]
        questions = [r["question"] for r in rows]
        text_emb = []
        for start in range(0, len(questions), 256):
            tok = processor(text=questions[start:start + 256], padding="max_length", max_length=64,
                            truncation=True, return_tensors="pt").to(device)
            text_emb.append(pooled(model.get_text_features(**tok)))
    np.savez(CACHE / f"ladder_siglip__{model_id.split('/')[-1]}.npz", image=image_emb, text=np.concatenate(text_emb))


# ---------------------------------------------------------------------------- probing

def probe(x, y, train, test, C=0.05):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
    clf.fit(x[train], y[train])
    return clf.predict_proba(x[test])[:, 1]


def score(y, p):
    return {"auroc": roc_auc_score(y, p), "acc": float(((p >= 0.5) == y).mean())}


def tasks(rows):
    """Probe units: each presence subtask on its own; spatial and gqa as whole families."""
    fam = np.array([r["family"] for r in rows])
    sub = np.array([r["subtype"] for r in rows])
    units = {f"presence/{s}": (fam == "presence") & (sub == s) for s in sorted(set(sub[fam == "presence"]))}
    units["spatial"] = fam == "spatial"
    units["gqa"] = fam == "gqa"
    return units


def normed(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def report(layer_step):
    rows, _ = load_manifest()
    y = np.array([r["label"] for r in rows])
    train = np.array([r["split"] == "train" for r in rows])
    units = tasks(rows)
    results = {}

    for path in sorted(CACHE.glob("ladder_*.npz")):
        z = np.load(path)
        name = path.stem.removeprefix("ladder_").replace("__", ":")
        res = results[name] = {}
        if "hidden" in z:
            hidden, pooled, frame = z["hidden"], z["pooled"], z["frame_of_row"]
            layers = list(range(0, hidden.shape[1], layer_step)) + ([hidden.shape[1] - 1] if (hidden.shape[1] - 1) % layer_step else [])
            for unit, mask in units.items():
                tr, te = mask & train, mask & ~train
                r = res[unit] = {"zeroshot": score(y[te], z["zeroshot"][te]), "layers": {}, "pooled": {}}
                for L in layers:
                    r["layers"][L] = score(y[te], probe(hidden[:, L].astype(np.float32), y, tr, te))
                    if unit.startswith("presence"):
                        r["pooled"][L] = score(y[te], probe(pooled[frame, L].astype(np.float32), y, tr, te))
                print(f"{name} {unit}: done", flush=True)
            dec = z["t_decide"]
            res["_latency"] = {"encode_p50": float(np.median(z["t_encode"])),
                               "decide_ms_per_query": float(np.median(dec[:, 0] / dec[:, 1]))}
        else:
            img, txt = normed(z["image"]), normed(z["text"])
            feats = {"image": img, "text": txt, "fusion": np.concatenate([img, txt, img * txt], axis=1)}
            for unit, mask in units.items():
                tr, te = mask & train, mask & ~train
                res[unit] = {k: score(y[te], probe(v, y, tr, te)) for k, v in feats.items()}

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "ladder.json").write_text(json.dumps(results, indent=1, default=float))
    print_tables(results, units)


def print_tables(results, units):
    unit_names = list(units)
    short = lambda u: u.replace("presence/", "")
    print("\n=== Best readout per task (test AUROC) ===")
    print(f"{'readout':44s}" + "".join(f"{short(u)[:12]:>13s}" for u in unit_names))
    for name, res in results.items():
        if "_latency" in res:
            lines = {"zeroshot": lambda r: r["zeroshot"]["auroc"],
                     "best layer": lambda r: max(v["auroc"] for v in r["layers"].values()),
                     "last layer": lambda r: r["layers"][max(r["layers"])]["auroc"],
                     "pooled (no query), best": lambda r: max((v["auroc"] for v in r["pooled"].values()), default=np.nan)}
        else:
            lines = {k: (lambda r, k=k: r[k]["auroc"]) for k in ("image", "text", "fusion")}
        for label, fn in lines.items():
            print(f"{(name + ' ' + label)[:44]:44s}" + "".join(f"{fn(res[u]):13.3f}" for u in unit_names))

    for name, res in results.items():
        if "_latency" not in res:
            continue
        print(f"\n=== {name}: test AUROC by layer (decision token | pooled visual, presence only) ===")
        layers = sorted(res[unit_names[0]]["layers"])
        print(f"{'layer':>6s}" + "".join(f"{short(u)[:12]:>13s}" for u in unit_names))
        for L in layers:
            cells = []
            for u in unit_names:
                d = res[u]["layers"][L]["auroc"]
                p = res[u]["pooled"].get(L, {}).get("auroc")
                cells.append(f"{d:.3f}|{p:.3f}" if p is not None else f"{d:.3f}")
            print(f"{L:6d}" + "".join(f"{c:>13s}" for c in cells))
        lat = res["_latency"]
        print(f"latency: encode p50 {lat['encode_p50']:.0f} ms, decide ≈{lat['decide_ms_per_query']:.1f} ms/query (batched)")

    print("\nAccuracy @0.5 on the reasoning families (chance = majority rate ~0.5)")
    for name, res in results.items():
        for u in ("spatial", "gqa"):
            r = res[u]
            if "_latency" in res:
                best = max(r["layers"], key=lambda L: r["layers"][L]["auroc"])
                print(f"  {name:40s} {u:8s} zeroshot {r['zeroshot']['acc']:.3f}  best probe (L{best}) {r['layers'][best]['acc']:.3f}")
            else:
                print(f"  {name:40s} {u:8s} text-only {r['text']['acc']:.3f}  fusion {r['fusion']['acc']:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["extract-vlm", "extract-siglip", "report"])
    parser.add_argument("--model")
    parser.add_argument("--budget", type=int, default=224)
    parser.add_argument("--layer-step", type=int, default=2)
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.cmd == "extract-vlm":
        extract_vlm(args.model, args.budget)
    elif args.cmd == "extract-siglip":
        extract_siglip(args.model)
    else:
        report(args.layer_step)


if __name__ == "__main__":
    main()
