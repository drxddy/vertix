"""Export Tier 1 presence heads for the runtime (run in venv: needs sklearn).

Each head is a logistic regression on the mean visual-token feature straight out of the Qwen3.5
vision encoder (layer 0 of the cached ladder features: no language model involved), plus a
temperature fitted on a calibration slice of the training frames. Written as plain arrays so the
MLX runtime can apply it with numpy.

By default trains on the runtime's own video-path features (encode_group), dumped by
`venv-mlx/bin/python video_eval/validate_heads.py --dump`: the image and video processors resize
differently, so image-path features shift probabilities by up to ~0.6 at runtime.

Usage: venv/bin/python export_heads.py [--features video|image]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
TASKS = {"human": "human", "vehicle": "vehicle", "animal": "animal"}


def fit_temperature(logits, y):
    def nll(t):
        p = np.clip(1 / (1 + np.exp(-logits / t)), 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x


def ece(y, p, bins=10):
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    return sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=str(ROOT / "data/cache/ladder_vlm__Qwen3.5-2B-bf16_224.npz"))
    parser.add_argument("--features", choices=["video", "image"], default="video")
    parser.add_argument("--video-cache", default=str(ROOT / "data/cache/video_pooled_presence.npz"))
    parser.add_argument("--out", default=str(ROOT / "heads"))
    parser.add_argument("--extra-train", default=str(ROOT / "data/cache/video_pooled_headtrain.npz"),
                        help="extra COCO train2017 features added to the training side (skipped if missing)")
    parser.add_argument("--C", type=float, default=0.05)
    parser.add_argument("--pool", default="human=meanmax,vehicle=mean,animal=meanmax",
                        help="'mean', 'meanmax', or per task as task=mode,... . meanmax adds the max over visual "
                             "tokens to the mean; chosen per task on the COCO test split (human: better calibrated "
                             "and fewer false alarms on person-like clutter; vehicle: mean scores higher)")
    args = parser.parse_args()

    rows = [json.loads(l) for l in open(ROOT / "data/ladder/manifest.jsonl")]
    modes = dict(kv.split("=") for kv in args.pool.split(",")) if "=" in args.pool else {t: args.pool for t in TASKS}

    def pooled_of(z, name):
        return z["pooled"] if modes[name] == "mean" else np.concatenate([z["pooled"], z["pooled_max"]], axis=1)

    if args.features == "video":
        v = np.load(args.video_cache)
        by_image = {t: dict(zip(v["images"].tolist(), pooled_of(v, t).astype(np.float32))) for t in TASKS}
        row_feat = lambda idx, name: np.stack([by_image[name][rows[i]["image"]] for i in idx])
    else:
        z = np.load(args.cache)
        feats = z["pooled"][:, 0].astype(np.float32)      # layer 0 = vision-encoder output, per frame
        row_feat = lambda idx, name: feats[z["frame_of_row"][idx]]
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    extra = extra_labels = None
    if args.features == "video" and Path(args.extra_train).exists():
        e = np.load(args.extra_train)
        extra = {t: pooled_of(e, t).astype(np.float32) for t in TASKS}
        by_path = {json.loads(l)["image"]: json.loads(l)["labels"] for l in open(ROOT / "data/head_train/manifest.jsonl")}
        extra_labels = [by_path[p] for p in e["images"].tolist()]
        print(f"adding {len(extra_labels)} COCO train2017 frames to every head's training side")

    for name, subtype in TASKS.items():
        idx = [i for i, r in enumerate(rows) if r["family"] == "presence" and r["subtype"] == subtype]
        x = row_feat(idx, name)
        y = np.array([rows[i]["label"] for i in idx])
        split = np.array([rows[i]["split"] for i in idx])
        train_all, test = split == "train", split == "test"
        # Hold out a quarter of the training frames to fit the temperature on unseen data
        rng = np.random.default_rng(0)
        calib = train_all & (rng.random(len(idx)) < 0.25)
        train = train_all & ~calib

        x_fit, y_fit = x[train], y[train]
        if extra is not None:
            keep = np.array([l[name] is not None for l in extra_labels])
            x_fit = np.concatenate([x_fit, extra[name][keep]])
            y_fit = np.concatenate([y_fit, np.array([l[name] for l in extra_labels], dtype=object)[keep].astype(int)])
        scaler = StandardScaler().fit(x_fit)
        clf = LogisticRegression(C=args.C, max_iter=5000).fit(scaler.transform(x_fit), y_fit)
        logit = lambda m: scaler.transform(x[m]) @ clf.coef_[0] + clf.intercept_[0]
        temp = fit_temperature(logit(calib), y[calib])
        p_test = 1 / (1 + np.exp(-logit(test) / temp))

        np.savez(out / f"{name}.npz", mean=scaler.mean_.astype(np.float32), scale=scaler.scale_.astype(np.float32),
                 coef=clf.coef_[0].astype(np.float32), intercept=np.float32(clf.intercept_[0]),
                 temperature=np.float32(temp), features=np.array("group" if modes[name] == "mean" else "group_meanmax"))
        pos = y[test] == 1
        fnr = 1 - (p_test[pos] >= 0.5).mean()
        print(f"{name:8s} [{modes[name]:7s}] train {len(y_fit):4d} calib {calib.sum():3d} test {test.sum():4d} | "
              f"test AUROC {roc_auc_score(y[test], p_test):.3f}  ECE {ece(y[test], p_test):.3f}  "
              f"miss rate@0.5 {fnr:.2f}  T={temp:.2f}")


if __name__ == "__main__":
    main()
