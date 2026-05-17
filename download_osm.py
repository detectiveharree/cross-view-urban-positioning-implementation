#!/usr/bin/env python3
"""
download_osm.py — Download OpenStreetMap tiles for each image in the dataset.

For each image in metadata.json, fetches a ~300×300 m patch of OSM vector data
centred on the image's GPS location. Saved as GeoJSON + a preview PNG.

The algorithm uses the noisy GPS position (geometry) as the tile centre — not
the SfM ground truth (computed_geometry) — because in deployment we only know
the noisy GPS prior.

OSM features fetched:
    building         — footprint polygons (class 1 in ray-casting)
    highway          — road and path centre-lines (class 2)
    crossing         — pedestrian crossings (class 3)
    amenity          — street_lamp, bench, fire_hydrant (not used in positioning
                       but may be useful for environment characterisation)
    building:levels, height, lit — structural metadata (informational)

Usage:
    python download_osm.py                        # all images in dataset
    python download_osm.py --city London          # one city at a time
    python download_osm.py --n-images 200         # limit for testing
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import osmnx as ox
from shapely.geometry import box

PRIOR_UNCERTAINTY = 50   # metres — expected GPS error radius
MAP_PADDING       = 100  # metres — extra buffer around prior
TOTAL_SPAN        = PRIOR_UNCERTAINTY + MAP_PADDING

ox.settings.use_cache   = True
ox.settings.log_console = False


def safe_project(gdf, to_crs=None):
    try:
        return ox.project_gdf(gdf, to_crs=to_crs)
    except Exception:
        try:
            return ox.projection.project_gdf(gdf, to_crs=to_crs)
        except Exception:
            if to_crs is None:
                lon = gdf.geometry.centroid.x.mean()
                lat = gdf.geometry.centroid.y.mean()
                zone = int((lon + 180) / 6) + 1
                hemi = "north" if lat >= 0 else "south"
                to_crs = (f"+proj=utm +zone={zone} +{hemi} "
                          "+ellps=WGS84 +datum=WGS84 +units=m +no_defs")
            return gdf.to_crs(to_crs)


def fetch_osm_features(lat, lon, dist):
    tags = {
        "building":         True,
        "highway":          True,
        "crossing":         True,
        "amenity":          ["street_lamp", "bench", "fire_hydrant"],
        "building:levels":  True,
        "height":           True,
        "lit":              True,
    }
    try:
        return ox.features_from_point((lat, lon), tags=tags, dist=dist)
    except Exception:
        return None


def save_preview(gdf, cx, cy, span, save_path):
    try:
        gdf_p = safe_project(gdf)
        fig, ax = plt.subplots(figsize=(6, 6), dpi=80)
        if "building" in gdf_p.columns:
            b = gdf_p[gdf_p["building"].notnull()]
            if not b.empty:
                b.plot(ax=ax, color="#e74c3c", alpha=0.7, zorder=2)
        if "highway" in gdf_p.columns:
            r = gdf_p[gdf_p["highway"].notnull()]
            if not r.empty:
                r.plot(ax=ax, color="#2c3e50", lw=1.5, alpha=0.8, zorder=1)
        p = gdf_p[gdf_p.geometry.type == "Point"]
        if not p.empty:
            p.plot(ax=ax, color="#3498db", markersize=30, zorder=3)
        ax.set_xlim(cx - span, cx + span)
        ax.set_ylim(cy - span, cy + span)
        ax.set_aspect("equal")
        plt.axis("off")
        plt.savefig(save_path, bbox_inches=None, pad_inches=0, facecolor="white")
        plt.close(fig)
    except Exception as e:
        print(f"    Preview failed: {e}")
        plt.close("all")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city",      default=None,   help="Process one city only")
    parser.add_argument("--n-images",  type=int, default=None, help="Limit images processed")
    parser.add_argument("--data-dir",  default=None)
    parser.add_argument("--dataset",   default="uk_dataset",
                        choices=["uk_dataset", "gv_dataset"])
    args = parser.parse_args()

    _repo_root = os.path.dirname(os.path.abspath(__file__))
    _data_root = args.data_dir or os.environ.get(
        "DATA_DIR", os.path.join(_repo_root, "..", "data")
    )
    dataset_dir  = os.path.join(_data_root, args.dataset)
    osm_tile_dir = os.path.join(_data_root, "osm_tiles")
    os.makedirs(osm_tile_dir, exist_ok=True)

    metadata_path = os.path.join(dataset_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        print(f"metadata.json not found at {metadata_path}")
        print("Run download_mapillary.py first.")
        return

    with open(metadata_path) as f:
        metadata = json.load(f)

    if args.city:
        metadata = [m for m in metadata if m.get("city", "").lower() == args.city.lower()]
        print(f"Filtered to {len(metadata)} images in {args.city}")

    if args.n_images:
        metadata = metadata[:args.n_images]

    print(f"Processing {len(metadata)} images...")
    processed = 0

    for i, entry in enumerate(metadata):
        img_id   = str(entry.get("id", ""))
        city     = entry.get("city", "unknown").replace(" ", "_").lower()
        city_dir = os.path.join(osm_tile_dir, city)
        os.makedirs(city_dir, exist_ok=True)

        vector_path  = os.path.join(city_dir, f"{img_id}_map.geojson")
        preview_path = os.path.join(city_dir, f"{img_id}_preview.png")

        if os.path.exists(vector_path) and os.path.exists(preview_path):
            continue

        coords = None
        if entry.get("geometry"):
            coords = entry["geometry"].get("coordinates")
        if not coords:
            continue

        lon, lat = coords
        gdf      = fetch_osm_features(lat, lon, TOTAL_SPAN)

        if gdf is not None and not gdf.empty:
            gdf.dropna(axis=1, how="all").to_file(vector_path, driver="GeoJSON")

            center_gdf = gpd.GeoDataFrame(
                geometry=[box(lon, lat, lon, lat).centroid], crs="EPSG:4326"
            )
            center_p = safe_project(center_gdf, to_crs=safe_project(gdf).crs)
            cx, cy   = center_p.geometry.iloc[0].x, center_p.geometry.iloc[0].y
            save_preview(gdf, cx, cy, TOTAL_SPAN, preview_path)

            processed += 1
            if processed % 10 == 0:
                print(f"  {processed} tiles saved")
        else:
            print(f"  No features for {img_id} ({city})")

    print(f"\nDone. {processed} new tiles in {os.path.abspath(osm_tile_dir)}")


if __name__ == "__main__":
    main()
