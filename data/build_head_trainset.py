"""Extra training images for the Tier 1 presence heads, from COCO train2017.

The first heads saw ~300 COCO val images. That is enough to score well on more COCO photos, and too
few to transfer: a clearly visible person in front of a busy patterned background read as P(human)
0.2 on a webcam. This samples a larger, balanced set (half with people) from train2017, which never
overlaps the val-based test split used to report head accuracy.

Writes data/head_train/manifest.jsonl rows {"image": "coco/train2017/<file>", "labels": {...}} and
caches the images under data/coco/train2017/.

Usage: venv-mlx/bin/python data/build_head_trainset.py [--n 3000]
"""
import argparse
import io
import json
import random
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).parent
ANN = ROOT / "coco/annotations/instances_train2017.json"
IMAGES = ROOT / "coco/train2017"
OUT = ROOT / "head_train"
LAND_VEHICLES = {"bicycle", "car", "motorcycle", "bus", "train", "truck"}


def ensure_annotations():
    if ANN.exists():
        return
    print("downloading COCO 2017 annotations (241 MB)...", flush=True)
    with urllib.request.urlopen("http://images.cocodataset.org/annotations/annotations_trainval2017.zip") as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        ANN.parent.mkdir(parents=True, exist_ok=True)
        ANN.write_bytes(z.read("annotations/instances_train2017.json"))


def fetch(name):
    path = IMAGES / name
    if not path.exists():
        for attempt in range(4):
            try:
                with urllib.request.urlopen(f"http://images.cocodataset.org/train2017/{name}", timeout=60) as r:
                    path.write_bytes(r.read())
                break
            except OSError:
                if attempt == 3:
                    raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    ensure_annotations()
    coco = json.loads(ANN.read_text())
    cats = {c["id"]: c for c in coco["categories"]}
    animal = {c["name"] for c in cats.values() if c["supercategory"] == "animal"}
    names = {}
    for a in coco["annotations"]:
        names.setdefault(a["image_id"], set()).add(cats[a["category_id"]]["name"])
    images = {i["id"]: i for i in coco["images"]}

    def labels(image_id):
        n = names.get(image_id, set())
        vehicle = 1 if n & LAND_VEHICLES else (None if n & {"airplane", "boat"} else 0)
        return {"human": int("person" in n), "vehicle": vehicle, "animal": int(bool(n & animal))}

    rng = random.Random(args.seed)
    with_person = [i for i in images if "person" in names.get(i, set())]
    without = [i for i in images if "person" not in names.get(i, set())]
    chosen = rng.sample(with_person, args.n // 2) + rng.sample(without, args.n - args.n // 2)
    IMAGES.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(fetch, [images[i]["file_name"] for i in chosen]))

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "manifest.jsonl", "w") as f:
        for i in chosen:
            f.write(json.dumps({"image": f"coco/train2017/{images[i]['file_name']}", "labels": labels(i)}) + "\n")
    print(f"wrote {len(chosen)} images to {OUT / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
