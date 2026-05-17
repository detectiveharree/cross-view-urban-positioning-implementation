#!/usr/bin/env python3
"""
classify_environment.py — Classify street-view images by environment type using CLIP.

Downloads openai/clip-vit-base-patch32 (~400 MB from HuggingFace) on first run
and caches it locally. Subsequent runs load from cache instantly.

How it works:
    CLIP is a vision-language model trained to match images with text descriptions.
    For each image we compute its similarity to a set of text prompts, one per
    environment label, and pick the closest match. No fine-tuning required.

Environment labels (edit ENVIRONMENT_PROMPTS to fit your study):
    URBAN       — Dense city centre, tall buildings, heavy traffic
    SUBURBAN    — Quiet residential streets, houses and gardens
    RURAL       — Countryside roads, villages, farmland
    INDUSTRIAL  — Warehouses, factories, service roads
    WATERFRONT  — Near rivers, canals, docks, or harbours

Output: results/environment_labels.json
    {
      "image_id": {
        "label": "URBAN",
        "confidence": 0.82,
        "scores": {"URBAN": 0.82, "SUBURBAN": 0.11, ...},
        "city": "London"
      }, ...
    }

Usage:
    python classify_environment.py
    python classify_environment.py --city Edinburgh --n-samples 200
    python classify_environment.py --dataset gv_dataset
"""

import os
import json
import random
import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# ---------------------------------------------------------------------------
# Edit these prompts to define the environment categories for your study.
# More specific prompts generally give better discrimination.
# ---------------------------------------------------------------------------
ENVIRONMENT_PROMPTS = {
    "URBAN":      "a dense city centre street with tall buildings, heavy traffic and many pedestrians",
    "SUBURBAN":   "a quiet suburban residential street with terraced or detached houses and gardens",
    "RURAL":      "a rural countryside road or village with fields, trees and very few buildings",
    "INDUSTRIAL": "an industrial area with warehouses, factories, loading bays and wide service roads",
    "WATERFRONT": "a street or path alongside a river, canal, dock, or harbour",
}


def load_clip(device):
    print("Loading CLIP model (openai/clip-vit-base-patch32) ...")
    print("First run downloads ~400 MB; subsequent runs use HuggingFace cache.")
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval().to(device)
    print(f"CLIP loaded on {device}")
    return model, processor


def classify_image(image_path, model, processor, prompts, device):
    image  = Image.open(image_path).convert("RGB")
    labels = list(prompts.keys())
    texts  = list(prompts.values())

    inputs = processor(text=texts, images=image, return_tensors="pt",
                       padding=True).to(device)
    with torch.no_grad():
        probs = model(**inputs).logits_per_image[0].softmax(dim=0)

    scores     = {label: float(p) for label, p in zip(labels, probs)}
    best_label = max(scores, key=scores.get)
    return {"label": best_label, "confidence": scores[best_label], "scores": scores}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",  default=None)
    parser.add_argument("--dataset",   default="uk_dataset",
                        choices=["uk_dataset", "gv_dataset"])
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Classify this many images (randomly sampled)")
    parser.add_argument("--city",      default=None)
    parser.add_argument("--output",    default="results/environment_labels.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_clip(device)

    _repo_root = os.path.dirname(os.path.abspath(__file__))
    _data_root = args.data_dir or os.environ.get(
        "DATA_DIR", os.path.join(_repo_root, "..", "data")
    )
    dataset_dir   = os.path.join(_data_root, args.dataset)
    metadata_path = os.path.join(dataset_dir, "metadata.json")

    with open(metadata_path) as f:
        metadata = json.load(f)

    if args.city:
        metadata = [m for m in metadata if m.get("city","").lower() == args.city.lower()]
        print(f"City filter: {args.city} — {len(metadata)} images")

    # Resume from existing output
    os.makedirs(os.path.dirname(args.output) or "results", exist_ok=True)
    results = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)
        print(f"Resuming — {len(results)} images already classified")

    remaining = [m for m in metadata if str(m["id"]) not in results]
    if args.n_samples and args.n_samples < len(remaining):
        remaining = random.sample(remaining, args.n_samples)

    print(f"Classifying {len(remaining)} images on {device} ...")
    print(f"Environment labels: {list(ENVIRONMENT_PROMPTS.keys())}")

    for i, m in enumerate(remaining):
        img_path = os.path.join(dataset_dir, m.get("local_path", ""))
        if not os.path.exists(img_path):
            continue

        try:
            result = classify_image(img_path, model, processor, ENVIRONMENT_PROMPTS, device)
            results[str(m["id"])] = {**result, "city": m.get("city")}
        except Exception as e:
            print(f"  Warning [{m['id']}]: {e}")

        if (i + 1) % 50 == 0:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  {i+1}/{len(remaining)} — checkpoint saved")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    label_counts: dict = {}
    for v in results.values():
        label_counts[v["label"]] = label_counts.get(v["label"], 0) + 1
    total = sum(label_counts.values())

    print(f"\nSaved {total} labels to {args.output}")
    print("\nLabel distribution:")
    for label in ENVIRONMENT_PROMPTS:
        count = label_counts.get(label, 0)
        bar   = "█" * int(count / max(total, 1) * 40)
        print(f"  {label:15s} {count:5d}  {bar}")


if __name__ == "__main__":
    main()
