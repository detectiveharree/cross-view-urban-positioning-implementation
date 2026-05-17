#!/usr/bin/env python3
"""
download_mapillary.py — Download street-view images and metadata from Mapillary.

Fetches images from the Mapillary Graph API for one or more cities.
Images are saved as JPEGs; metadata (GPS, heading, camera type, capture time, etc.)
is saved to metadata.json alongside them.

Requires a MAPILLARY_ACCESS_TOKEN in your .env file.

Usage:
    python download_mapillary.py                         # all UK cities, 5000 total
    python download_mapillary.py --cities London Oxford  # specific cities only
    python download_mapillary.py --n-images 500          # fewer images
    python download_mapillary.py --output-dir ./data/my_city --bbox -0.15 51.49 -0.08 51.52

Mapillary metadata saved per image:
    id              — unique image ID
    geometry        — noisy GPS coordinates (used as prior in positioning)
    computed_geometry — SfM-refined coordinates (ground truth for evaluation)
    compass_angle   — camera heading in degrees
    camera_parameters — normalised focal length (used to compute HFOV)
    camera_type     — PERSPECTIVE, FISHEYE, etc.
    captured_at     — Unix timestamp (ms) — convert to datetime for time-of-day analysis
    make / model    — camera make and model
    width / height  — image dimensions
"""

import os
import json
import time
import random
import argparse
import requests
from dotenv import load_dotenv

load_dotenv()

ACCESS_TOKEN = os.getenv("MAPILLARY_ACCESS_TOKEN")
BASE_URL     = "https://graph.mapillary.com"

# Bounding boxes: [west, south, east, north]
UK_CITIES = {
    "London":     [-0.150, 51.490, -0.080, 51.520],
    "Manchester": [-2.260, 53.460, -2.190, 53.500],
    "Birmingham": [-1.930, 52.450, -1.860, 52.490],
    "Edinburgh":  [-3.220, 55.930, -3.160, 55.960],
    "Bristol":    [-2.630, 51.430, -2.560, 51.470],
    "Leeds":      [-1.570, 53.780, -1.520, 53.810],
    "Glasgow":    [-4.280, 55.840, -4.220, 55.870],
    "Liverpool":  [-2.990, 53.390, -2.940, 53.420],
    "Newcastle":  [-1.640, 54.960, -1.590, 54.990],
    "Cardiff":    [-3.200, 51.470, -3.160, 51.500],
    "Oxford":     [-1.270, 51.740, -1.220, 51.770],
    "Cambridge":  [ 0.100, 52.190,  0.150, 52.220],
}

METADATA_FIELDS = [
    "id", "computed_geometry", "geometry", "compass_angle",
    "camera_parameters", "camera_type", "captured_at",
    "thumb_2048_url", "sequence", "width", "height", "make", "model",
]


def fetch_image_ids(bbox, limit=5000):
    bbox_str = ",".join(map(str, bbox))
    for current_limit in [limit, 2000, 1000, 500]:
        try:
            r = requests.get(f"{BASE_URL}/images", params={
                "access_token": ACCESS_TOKEN,
                "bbox": bbox_str,
                "is_pano": "false",
                "limit": current_limit,
            })
            if r.status_code == 500:
                continue
            r.raise_for_status()
            return [img["id"] for img in r.json().get("data", [])]
        except Exception as e:
            print(f"    Error fetching IDs (limit {current_limit}): {e}")
            time.sleep(1)
    return []


def fetch_metadata(image_id):
    for delay in [1, 2, 4]:
        try:
            r = requests.get(f"{BASE_URL}/{image_id}", params={
                "access_token": ACCESS_TOKEN,
                "fields": ",".join(METADATA_FIELDS),
            })
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(delay)
    return {}


def download_image(url, save_path):
    try:
        r = requests.get(url, stream=True, timeout=15)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"    Download failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities",    nargs="+", default=None,
                        help="City names to download (default: all UK cities)")
    parser.add_argument("--n-images",  type=int, default=5000,
                        help="Total images across all cities")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save images and metadata.json")
    parser.add_argument("--bbox",      nargs=4, type=float, default=None,
                        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
                        help="Custom bounding box for a single area")
    args = parser.parse_args()

    if not ACCESS_TOKEN:
        print("MAPILLARY_ACCESS_TOKEN not set. Copy .env.example to .env and add your token.")
        return

    # Resolve output directory
    _repo_root = os.path.dirname(os.path.abspath(__file__))
    _data_root = os.environ.get("DATA_DIR", os.path.join(_repo_root, "..", "data"))
    output_dir = args.output_dir or os.path.join(_data_root, "uk_dataset")
    os.makedirs(output_dir, exist_ok=True)

    # Build city list
    if args.bbox:
        cities = {"custom_area": args.bbox}
    elif args.cities:
        cities = {c: UK_CITIES[c] for c in args.cities if c in UK_CITIES}
        missing = [c for c in args.cities if c not in UK_CITIES]
        if missing:
            print(f"Unknown cities (check spelling): {missing}")
            print(f"Available: {list(UK_CITIES.keys())}")
    else:
        cities = UK_CITIES

    images_per_city = max(1, args.n_images // len(cities))

    # Load existing metadata
    metadata_path  = os.path.join(output_dir, "metadata.json")
    master_metadata = []
    existing_ids    = set()
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            master_metadata = json.load(f)
        existing_ids = {str(m["id"]) for m in master_metadata}
        print(f"Resuming — {len(existing_ids)} images already downloaded.")

    print(f"Target: {args.n_images} images across {len(cities)} cities")
    print(f"Output: {os.path.abspath(output_dir)}")

    for city_name, bbox in cities.items():
        print(f"\n{city_name}...")
        all_ids = fetch_image_ids(bbox)
        new_ids = [i for i in all_ids if str(i) not in existing_ids]

        if not new_ids:
            print(f"  No new images (all {len(all_ids)} already downloaded)")
            continue

        sampled = random.sample(new_ids, min(len(new_ids), images_per_city))
        print(f"  Fetching {len(sampled)} of {len(new_ids)} new images...")

        city_dir = os.path.join(output_dir, city_name.replace(" ", "_").lower())
        os.makedirs(city_dir, exist_ok=True)

        count = 0
        for img_id in sampled:
            meta = fetch_metadata(img_id)
            if not meta or "thumb_2048_url" not in meta:
                continue

            img_path = os.path.join(city_dir, f"{img_id}.jpg")
            if download_image(meta["thumb_2048_url"], img_path):
                master_metadata.append({
                    **meta,
                    "city":       city_name,
                    "local_path": os.path.join(city_name.replace(" ", "_").lower(), f"{img_id}.jpg"),
                })
                existing_ids.add(str(img_id))
                count += 1

            if count % 50 == 0 and count > 0:
                print(f"    {count}/{len(sampled)} saved")

        with open(metadata_path, "w") as f:
            json.dump(master_metadata, f, indent=2)
        print(f"  Done. {count} new images. Total: {len(master_metadata)}")

    print(f"\nFinished. {len(master_metadata)} images in {os.path.abspath(metadata_path)}")


if __name__ == "__main__":
    main()
