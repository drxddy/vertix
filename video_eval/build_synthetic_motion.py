"""Synthetic short-horizon motion clips with known answers, built from COCO val2017.

Each "source" produces a forward clip and its exact reversal (frames[::-1]), whose answers must flip:
  camera motion over a still photo:  zoom_in <-> zoom_out, pan_right <-> pan_left, plus static
  object motion: a COCO person (segmentation-mask cut-out) on a different background:
                 approach (grows) <-> recede (shrinks), move_right <-> move_left, plus static
Clips are 4 s at 8 fps (32 frames, 320x240) mp4s, so windows of 1, 2 and 4 s can be evaluated.

Writes data/video_eval/synthetic/*.mp4 and data/video_eval/synthetic/manifest.jsonl rows:
  {"clip", "kind", "pair", "question_id", "question", "label"}

Usage: venv-mlx/bin/python video_eval/build_synthetic_motion.py [--n-camera 60] [--n-object 60]
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
COCO = ROOT / "data/coco"
OUT = ROOT / "data/video_eval/synthetic"
W, H, FPS, N_FRAMES = 320, 240, 8, 32

CAMERA_QUESTIONS = {
    "cam_forward": "Is the camera moving forward, getting closer to the scene?",
    "cam_backward": "Is the camera moving backward, away from the scene?",
    "cam_pan_right": "Is the camera panning to the right?",
    "cam_pan_left": "Is the camera panning to the left?",
    "cam_moving": "Is the camera moving?",
}
OBJECT_QUESTIONS = {
    "obj_approach": "Is the person moving toward the camera?",
    "obj_recede": "Is the person moving away from the camera?",
    "obj_right": "Is the person moving to the right?",
    "obj_left": "Is the person moving to the left?",
    "obj_moving": "Is the person moving?",
}
# Which questions are true for each motion kind (everything else in that family is false)
TRUE = {
    "zoom_in": {"cam_forward", "cam_moving"}, "zoom_out": {"cam_backward", "cam_moving"},
    "pan_right": {"cam_pan_right", "cam_moving"}, "pan_left": {"cam_pan_left", "cam_moving"}, "cam_static": set(),
    "approach": {"obj_approach", "obj_moving"}, "recede": {"obj_recede", "obj_moving"},
    "move_right": {"obj_right", "obj_moving"}, "move_left": {"obj_left", "obj_moving"}, "obj_static": set(),
}
REVERSE = {"zoom_in": "zoom_out", "pan_right": "pan_left", "approach": "recede", "move_right": "move_left"}


def load_rgb(path, size=None):
    im = Image.open(path).convert("RGB")
    return np.asarray(im.resize(size, Image.BICUBIC) if size else im)


def crop_resize(img, cx, cy, scale):
    """View of `img` centred at (cx, cy) (fractions) covering 1/scale of it, resized to W x H."""
    h, w = img.shape[:2]
    ch, cw = int(h / scale), int(w / scale)
    y = int(np.clip(cy * h - ch / 2, 0, h - ch))
    x = int(np.clip(cx * w - cw / 2, 0, w - cw))
    return np.asarray(Image.fromarray(img[y:y + ch, x:x + cw]).resize((W, H), Image.BICUBIC))


def camera_clip(img, kind):
    frames = []
    for i in range(N_FRAMES):
        a = i / (N_FRAMES - 1)
        if kind == "zoom_in":
            frames.append(crop_resize(img, 0.5, 0.5, 1.0 + 0.8 * a))
        elif kind == "pan_right":   # camera turns right: the view window slides right across the photo
            frames.append(crop_resize(img, 0.3 + 0.4 * a, 0.5, 1.6))
        else:  # cam_static
            frames.append(crop_resize(img, 0.5, 0.5, 1.3))
    return frames


def person_cutout(ann, img_info):
    img = load_rgb(COCO / "val2017" / img_info["file_name"])
    mask = Image.new("L", (img_info["width"], img_info["height"]), 0)
    for poly in ann["segmentation"]:
        ImageDraw.Draw(mask).polygon(list(map(float, poly)), fill=255)
    x, y, w, h = map(int, ann["bbox"])
    return img[y:y + h, x:x + w], np.asarray(mask)[y:y + h, x:x + w]


def object_clip(background, sprite, mask, kind):
    frames = []
    for i in range(N_FRAMES):
        a = i / (N_FRAMES - 1)
        if kind == "approach":
            height, cx = 0.35 + 0.5 * a, 0.5
        elif kind == "move_right":
            height, cx = 0.6, 0.2 + 0.6 * a
        else:  # obj_static
            height, cx = 0.6, 0.5
        sh = max(8, int(H * height))
        sw = max(4, int(sprite.shape[1] * sh / sprite.shape[0]))
        spr = np.asarray(Image.fromarray(sprite).resize((sw, sh), Image.BICUBIC))
        msk = np.asarray(Image.fromarray(mask).resize((sw, sh), Image.BILINEAR))[..., None] / 255.0
        frame = background.copy().astype(np.float32)
        x0 = int(cx * W - sw / 2)
        y0 = H - sh - 4                        # feet near the bottom: growth reads as approaching
        xa, xb = max(0, x0), min(W, x0 + sw)
        if xb > xa:
            region = frame[y0:y0 + sh, xa:xb]
            m = msk[:, xa - x0:xb - x0]
            frame[y0:y0 + sh, xa:xb] = m * spr[:, xa - x0:xb - x0] + (1 - m) * region
        frames.append(frame.astype(np.uint8))
    return frames


def write_clip(frames, path):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()


def rows_for(clip, kind, pair, family):
    questions = CAMERA_QUESTIONS if family == "camera" else OBJECT_QUESTIONS
    return [{"clip": clip, "kind": kind, "pair": pair, "family": family, "question_id": qid, "question": text,
             "label": int(qid in TRUE[kind])} for qid, text in questions.items()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-camera", type=int, default=60)
    parser.add_argument("--n-object", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)

    coco = json.loads((COCO / "annotations/instances_val2017.json").read_text())
    have = {p.name for p in (COCO / "val2017").glob("*.jpg")}
    images = {i["id"]: i for i in coco["images"] if i["file_name"] in have}
    persons = [a for a in coco["annotations"]
               if a["category_id"] == 1 and a["image_id"] in images and not a["iscrowd"]
               and a["area"] / (images[a["image_id"]]["width"] * images[a["image_id"]]["height"]) > 0.08
               and a["bbox"][3] > 1.5 * a["bbox"][2]]          # upright, reasonably large people
    with_people = {a["image_id"] for a in coco["annotations"] if a["category_id"] == 1}
    backgrounds = [i for i in images.values() if i["id"] not in with_people]

    rows = []
    for n, img_info in enumerate(rng.sample(list(images.values()), args.n_camera)):
        img = load_rgb(COCO / "val2017" / img_info["file_name"])
        for kind in ("zoom_in", "pan_right", "cam_static"):
            frames = camera_clip(img, kind)
            pair = f"cam{n}_{kind}"
            write_clip(frames, OUT / f"{pair}.mp4")
            rows += rows_for(f"{pair}.mp4", kind, pair, "camera")
            if kind in REVERSE:
                write_clip(frames[::-1], OUT / f"{pair}_rev.mp4")
                rows += rows_for(f"{pair}_rev.mp4", REVERSE[kind], pair, "camera")

    for n, ann in enumerate(rng.sample(persons, min(args.n_object, len(persons)))):
        sprite, mask = person_cutout(ann, images[ann["image_id"]])
        bg = load_rgb(COCO / "val2017" / rng.choice(backgrounds)["file_name"], (W, H))
        for kind in ("approach", "move_right", "obj_static"):
            frames = object_clip(bg, sprite, mask, kind)
            pair = f"obj{n}_{kind}"
            write_clip(frames, OUT / f"{pair}.mp4")
            rows += rows_for(f"{pair}.mp4", kind, pair, "object")
            if kind in REVERSE:
                write_clip(frames[::-1], OUT / f"{pair}_rev.mp4")
                rows += rows_for(f"{pair}_rev.mp4", REVERSE[kind], pair, "object")

    with open(OUT / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    clips = {r["clip"] for r in rows}
    print(f"wrote {len(clips)} clips, {len(rows)} questions to {OUT} "
          f"({len(persons)} usable person cut-outs, {len(backgrounds)} person-free backgrounds)")


if __name__ == "__main__":
    main()
