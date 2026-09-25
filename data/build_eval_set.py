"""Builds a labelled yes/no decision eval set from COCO val2017 plus synthetic camera faults.

Tasks with COCO ground truth (human-annotated boxes):
  human, vehicle (land vehicles), animal, traffic_light
Synthetic tasks (derived variants of a subset of the clean images):
  obstructed  - a blurred finger-like blob covering 25-55% of the frame
  lighting_ok - clean (1) vs. underexposed with sensor noise (0)

Writes data/eval/images/*.jpg and data/eval/manifest.jsonl, one row per image:
  {"file", "split", "coco_id", "variant", "labels": {task: 0|1|null}, "max_area": {task: frac}}
null means the task is not evaluated on that image (ambiguous or not applicable).

Usage: python data/build_eval_set.py [--n 700] [--n-train 400] [--n-synth 200]
"""
import argparse
import json
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).parent
ANN = ROOT / "coco/annotations/instances_val2017.json"
OUT = ROOT / "eval"
CACHE = ROOT / "coco/val2017"
LAND_VEHICLES = {"bicycle", "car", "motorcycle", "bus", "train", "truck"}


def coco_labels(anns, cats, img):
    area = img["width"] * img["height"]
    present = {}  # category name -> largest instance area fraction
    for a in anns:
        name = cats[a["category_id"]]["name"]
        present[name] = max(present.get(name, 0.0), a["area"] / area)
    supers = {cats[a["category_id"]]["supercategory"] for a in anns}

    def task(names):
        hits = [present[n] for n in names if n in present]
        return (1, max(hits)) if hits else (0, 0.0)

    labels, max_area = {}, {}
    labels["human"], max_area["human"] = task({"person"})
    labels["vehicle"], max_area["vehicle"] = task(LAND_VEHICLES)
    if labels["vehicle"] == 0 and "vehicle" in supers:
        labels["vehicle"] = None  # only airplanes/boats: ambiguous for a land-robot "vehicle" question
    animal_names = {c["name"] for c in cats.values() if c["supercategory"] == "animal"}
    labels["animal"], max_area["animal"] = task(animal_names)
    labels["traffic_light"], max_area["traffic_light"] = task({"traffic light"})
    return labels, max_area


def occlude(img, rng):
    """Finger/debris over the lens: a blurred, partly translucent, shaded blob entering from an edge."""
    w, h = img.size
    target = rng.uniform(0.25, 0.55)
    # The ellipse is centred on an edge, so roughly half of it lands in frame
    rw, rh = w * np.sqrt(target) * 0.85, h * np.sqrt(target) * 0.85
    edge = rng.integers(4)
    cx = [rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * w, 0, w][edge]
    cy = [0, h, rng.uniform(0.2, 0.8) * h, rng.uniform(0.2, 0.8) * h][edge]
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).ellipse([cx - rw, cy - rh, cx + rw, cy + rh], fill=int(255 * rng.uniform(0.7, 0.95)))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=w * 0.05))
    # Out-of-focus skin: tone plus a brightness falloff away from the edge it enters from
    tone = rng.uniform([90, 55, 40], [190, 130, 100])
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.hypot((xx - cx) / rw, (yy - cy) / rh)
    shade = np.clip(0.55 + 0.5 * dist, 0.4, 1.2)[..., None]
    blob = Image.fromarray(np.clip(tone * shade, 0, 255).astype(np.uint8))
    return Image.composite(blob, img, mask)


def underexpose(img, rng):
    """Low light: strong exposure drop, gamma crush and sensor noise."""
    x = np.asarray(img).astype(np.float32) / 255.0
    x = (x * rng.uniform(0.1, 0.3)) ** rng.uniform(1.0, 1.3)
    x = x + rng.normal(0, rng.uniform(0.01, 0.025), x.shape)
    return Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8))


def fetch(img, retries=4):
    """Download a COCO image, caching the original under data/coco/val2017."""
    cached = CACHE / img["file_name"]
    if not cached.exists():
        url = f"http://images.cocodataset.org/val2017/{img['file_name']}"
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    data = r.read()
                break
            except (TimeoutError, OSError):
                if attempt == retries - 1:
                    raise
        cached.write_bytes(data)
    return Image.open(cached).convert("RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=700)
    parser.add_argument("--n-train", type=int, default=400)
    parser.add_argument("--n-synth", type=int, default=200)
    parser.add_argument("--n-traffic", type=int, default=160,
                        help="images guaranteed to contain a traffic light (rare in random COCO samples)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    coco = json.loads(ANN.read_text())
    cats = {c["id"]: c for c in coco["categories"]}
    anns_by_img = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    random.seed(args.seed)
    tl_id = next(c["id"] for c in coco["categories"] if c["name"] == "traffic light")
    with_tl = [i for i in coco["images"] if any(a["category_id"] == tl_id for a in anns_by_img.get(i["id"], []))]
    enriched = random.sample(with_tl, args.n_traffic)
    chosen = {i["id"] for i in enriched}
    rest = random.sample([i for i in coco["images"] if i["id"] not in chosen], args.n - args.n_traffic)
    images = enriched + rest
    random.shuffle(images)
    splits = {img["id"]: ("train" if i < args.n_train else "test") for i, img in enumerate(images)}
    synth_ids = {img["id"] for img in random.sample(images, args.n_synth)}

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(16) as pool:
        pixels = list(pool.map(fetch, images))

    rows = []
    for img, pil in zip(images, pixels):
        labels, max_area = coco_labels(anns_by_img.get(img["id"], []), cats, img)
        synth = img["id"] in synth_ids
        base = {"split": splits[img["id"]], "coco_id": img["id"]}
        rng = np.random.default_rng(img["id"])

        clean = f"{img['id']:012d}.jpg"
        pil.save(OUT / "images" / clean, quality=95)
        rows.append({**base, "file": clean, "variant": "clean", "max_area": max_area,
                     "labels": {**labels, "obstructed": 0 if synth else None,
                                "lighting_ok": 1 if synth else None}})
        if not synth:
            continue
        # Corrupted variants only carry their own task label: the fault may hide the COCO objects
        none = {k: None for k in labels}
        for variant, fn, task_labels in [
            ("occluded", occlude, {"obstructed": 1, "lighting_ok": None}),
            ("dark", underexpose, {"obstructed": None, "lighting_ok": 0}),
        ]:
            name = f"{img['id']:012d}_{variant}.jpg"
            fn(pil, rng).save(OUT / "images" / name, quality=95)
            rows.append({**base, "file": name, "variant": variant, "max_area": {},
                         "labels": {**none, **task_labels}})

    with open(OUT / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"wrote {len(rows)} images to {OUT}")
    tasks = rows[0]["labels"].keys()
    for split in ("train", "test"):
        print(f"\n{split}:")
        for t in tasks:
            vals = [r["labels"][t] for r in rows if r["split"] == split and r["labels"][t] is not None]
            print(f"  {t:14s} n={len(vals):4d}  positives={sum(vals):4d}")


if __name__ == "__main__":
    main()
