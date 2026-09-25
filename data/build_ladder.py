"""Builds the "decision ladder": yes/no questions graded from perception to reasoning, one manifest.

Families:
  presence  - the fixed COCO/synthetic tasks from data/eval (what fast classifiers already do well)
  spatial   - VSR: true/false spatial-relation statements on COCO images ("the cat is on the laptop")
  gqa       - GQA testdev-balanced yes/no questions (relations, attributes, logic, comparisons),
              answer-balanced to suppress language priors

Splits are by image, so a probe never sees a test image during training.
Writes data/ladder/manifest.jsonl rows:
  {"family", "subtype", "image", "question", "label", "split", "group"}
where "image" is relative to data/ and "group" identifies the frame (all its questions share one prefix).

Usage: python data/build_ladder.py
"""
import json
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
OUT = ROOT / "ladder"
TRAIN_FRAC = 0.6

# Fixed-task questions: identical to eval.py's TASKS so the presence rows match earlier results
PRESENCE_TASKS = {
    "human": "Is a human present in the image?",
    "vehicle": "Is there a car, truck, bus, motorcycle, bicycle or train in the image?",
    "animal": "Is there an animal in the image?",
    "traffic_light": "Is there a traffic light in the image?",
    "obstructed": "Is the camera lens obstructed?",
    "lighting_ok": "Is the lighting adequate?",
}


def split_by_group(groups, seed):
    groups = sorted(set(groups))
    random.Random(seed).shuffle(groups)
    n_train = int(len(groups) * TRAIN_FRAC)
    return {g: ("train" if i < n_train else "test") for i, g in enumerate(groups)}


def presence_rows():
    rows = []
    for r in map(json.loads, open(ROOT / "eval/manifest.jsonl")):
        for task, label in r["labels"].items():
            if label is not None:
                rows.append({"family": "presence", "subtype": task, "image": f"eval/images/{r['file']}",
                             "question": PRESENCE_TASKS[task], "label": label, "split": r["split"],
                             "group": f"eval:{r['file']}"})
    return rows


def gqa_rows():
    q = pd.read_parquet(ROOT / "gqa/questions.parquet")
    q = q[q.answer.isin(["yes", "no"])]
    images = pd.read_parquet(ROOT / "gqa/images.parquet")
    (OUT / "gqa").mkdir(parents=True, exist_ok=True)
    for _, im in images.iterrows():
        path = OUT / "gqa" / f"{im['id']}.jpg"
        if not path.exists():
            path.write_bytes(im["image"]["bytes"])
    split = split_by_group(q.imageId, seed=1)
    return [{"family": "gqa", "subtype": f"{r.types['structural']}/{r.types['semantic']}",
             "image": f"ladder/gqa/{r.imageId}.jpg", "question": r.question,
             "label": int(r.answer == "yes"), "split": split[r.imageId], "group": f"gqa:{r.imageId}"}
            for r in q.itertuples()]


def coco_path(link):
    # ".../train2017/000000451431.jpg" -> data/coco/train2017/000000451431.jpg (VSR mixes train and val)
    split, name = link.split("/")[-2:]
    return ROOT / "coco" / split / name


def fetch_coco(link):
    path = coco_path(link)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        for attempt in range(4):
            try:
                with urllib.request.urlopen(link, timeout=60) as r:
                    path.write_bytes(r.read())
                break
            except (TimeoutError, OSError):
                if attempt == 3:
                    raise
    return path


def vsr_question(caption):
    # "The cat is on the laptop." -> "Is it true that the cat is on the laptop?"
    body = caption.strip().rstrip(".")
    return f"Is it true that {body[0].lower() + body[1:]}?"


def vsr_rows():
    items = [json.loads(l) for l in open(ROOT / "vsr/test.jsonl")]
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(fetch_coco, sorted({it["image_link"] for it in items})))
    split = split_by_group([it["image"] for it in items], seed=2)
    return [{"family": "spatial", "subtype": it["relation"], "image": str(coco_path(it["image_link"]).relative_to(ROOT)),
             "question": vsr_question(it["caption"]), "label": int(it["label"]),
             "split": split[it["image"]], "group": f"vsr:{it['image']}"}
            for it in items]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = presence_rows() + vsr_rows() + gqa_rows()
    with open(OUT / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    df = pd.DataFrame(rows)
    print(f"wrote {len(df)} questions over {df.group.nunique()} frames to {OUT / 'manifest.jsonl'}")
    print(df.groupby(["family", "split"]).agg(questions=("label", "size"), frames=("group", "nunique"),
                                              yes_rate=("label", "mean")).round(2).to_string())


if __name__ == "__main__":
    main()
