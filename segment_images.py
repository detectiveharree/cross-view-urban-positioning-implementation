#!/usr/bin/env python3
"""
segment_images.py — Run YOLO segmentation and compute per-class coverage statistics.

For each image, computes what fraction of pixels each semantic class covers.
Useful for characterising environments and understanding why the positioning
algorithm performs differently across scenes.

YOLO classes in the fine-tuned model:
    Class  0: building     — used in positioning (obstacle / occlusion)
    Class  1: lamp post    — detected but NOT used in positioning
    Class  2: tree         — detected but NOT used in positioning
    Class  3: stop sign    — detected but NOT used in positioning
    Class  4: fire hydrant — detected but NOT used in positioning
    Class  5: road         — used in positioning (road surface)
    Class  6: sidewalk     — used in positioning (road surface)
    Class  7: crosswalk    — used in positioning (distinct feature)

Output: results/coverage.csv
    Columns: image_id, city, building, lamp_post, tree, stop_sign,
             fire_hydrant, road, sidewalk, crosswalk, total_detected

Usage:
    python segment_images.py
    python segment_images.py --city London --n-samples 200
    python segment_images.py --dataset gv_dataset
"""

import os
import json
import random
import argparse
import numpy as np
import pandas as pd
import cv2

CLASS_NAMES = ["building", "lamp_post", "tree", "stop_sign",
               "fire_hydrant", "road", "sidewalk", "crosswalk"]


def compute_coverage(image_path, model, conf=0.2):
    """Return per-class pixel fraction for one image."""
    results = model.predict(image_path, conf=conf, verbose=False)
    res     = results[0]
    h, w    = res.orig_shape
    total   = h * w

    coverage = {name: 0.0 for name in CLASS_NAMES}
    if res.masks:
        for i, mask in enumerate(res.masks.data):
            cls = int(res.boxes.cls[i])
            if cls < len(CLASS_NAMES):
                m = cv2.resize(mask.cpu().numpy(), (w, h)) > 0
                coverage[CLASS_NAMES[cls]] += float(m.sum()) / total

    # Clamp individual classes to [0, 1] (masks can overlap slightly)
    for k in coverage:
        coverage[k] = min(coverage[k], 1.0)

    coverage["total_detected"] = min(sum(v for k, v in coverage.items()
                                         if k != "total_detected"), 1.0)
    return coverage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",  default=None)
    parser.add_argument("--dataset",   default="gv_dataset",
                        choices=["gv_dataset", "uk_dataset"])
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Limit number of images (random sample)")
    parser.add_argument("--city",      default=None)
    parser.add_argument("--model",     default=None,
                        help="Path to YOLO .pt weights (auto-discovered if omitted)")
    parser.add_argument("--conf",      type=float, default=0.2)
    parser.add_argument("--output",    default="results/coverage.csv")
    args = parser.parse_args()

    from ultralytics import YOLO
    from engine import find_best_weights

    _repo_root = os.path.dirname(os.path.abspath(__file__))
    _data_root = args.data_dir or os.environ.get(
        "DATA_DIR", os.path.join(_repo_root, "..", "data")
    )

    model_path = args.model or find_best_weights(
        os.path.join(_repo_root, "..", "runs", "segment", "positioning_research")
    )
    if not model_path:
        raise FileNotFoundError(
            "No YOLO model weights found. Pass --model <path.pt> or check that "
            "runs/segment/positioning_research/*/weights/best.pt exists."
        )
    print(f"Model: {model_path}")
    model = YOLO(model_path)

    dataset_dir   = os.path.join(_data_root, args.dataset)
    metadata_path = os.path.join(dataset_dir, "metadata.json")
    with open(metadata_path) as f:
        metadata = json.load(f)

    if args.city:
        metadata = [m for m in metadata if m.get("city","").lower() == args.city.lower()]
        print(f"City filter: {args.city} — {len(metadata)} images")

    if args.n_samples and args.n_samples < len(metadata):
        metadata = random.sample(metadata, args.n_samples)

    print(f"Segmenting {len(metadata)} images...")

    os.makedirs(os.path.dirname(args.output) or "results", exist_ok=True)

    # Resume from existing output
    existing = set()
    rows     = []
    if os.path.exists(args.output):
        df_existing = pd.read_csv(args.output)
        existing    = set(df_existing["image_id"].astype(str))
        rows        = df_existing.to_dict("records")
        print(f"Resuming — {len(existing)} images already processed")

    for i, m in enumerate(metadata):
        img_id = str(m["id"])
        if img_id in existing:
            continue

        img_path = os.path.join(dataset_dir, m.get("local_path", ""))
        if not os.path.exists(img_path):
            continue

        try:
            cov = compute_coverage(img_path, model, args.conf)
            rows.append({"image_id": img_id, "city": m.get("city", "unknown"), **cov})
        except Exception as e:
            print(f"  Warning [{img_id}]: {e}")

        if (i + 1) % 50 == 0:
            pd.DataFrame(rows).to_csv(args.output, index=False)
            print(f"  {i+1}/{len(metadata)} — checkpoint saved")

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)

    print(f"\nSaved {len(df)} rows to {args.output}")
    print("\nMean coverage by class:")
    for cls in CLASS_NAMES + ["total_detected"]:
        if cls in df.columns:
            print(f"  {cls:20s}: {df[cls].mean()*100:5.1f}%")

    if "city" in df.columns:
        print("\nMean total coverage by city:")
        for city, grp in df.groupby("city"):
            print(f"  {city:15s}: {grp['total_detected'].mean()*100:5.1f}%")


if __name__ == "__main__":
    main()
