#!/usr/bin/env python3
"""
run_positioning.py — Vectorised semantic ray-casting positioning experiment.

Runs the planarity+occlusion positioning variant with a known heading assumption
on a set of Mapillary images and produces accuracy metrics (top-N% inclusion
rate, median error in metres).

Key metric — bracket percentile:
    For each image, the heatmap assigns a score to every candidate position.
    The GT location's percentile rank in that heatmap is recorded.
    "Top-10% inclusion" = fraction of images where GT is in the top 10% of scores.
    Lower percentile = better. This is more meaningful than raw error because it
    accounts for the size of the search area.

Usage examples:
    python run_positioning.py --n-samples 100
    python run_positioning.py --n-samples 200 --city London
    python run_positioning.py --n-samples 50 --env-labels results/environment_labels.json
"""

import gc
import os
import sys
import json
import math
import random
import argparse
import time
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import torch
import geopandas as gpd
from shapely.geometry import Point

from engine import (
    PositioningConfig,
    PositioningEngine,
    ExperimentMode,
    SampleData,
    find_best_weights,
    calculate_hfov,
    get_training_image_ids,
)


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

MEM_LIMIT_BYTES = 1 * 1024 ** 3   # 1 GB


# ---------------------------------------------------------------------------
# CACHED GRID COORDINATES
# ---------------------------------------------------------------------------

_coord_cache: Dict[Tuple[int, int, str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def get_eval_coords(H, W, pad, prior_px, device):
    key = (H, W, device)
    if key not in _coord_cache:
        gy, gx = torch.meshgrid(
            torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
        )
        gy, gx = gy.flatten().float(), gx.flatten().float()
        em = torch.zeros((H, W), dtype=torch.bool, device=device)
        em[pad:pad+prior_px, pad:pad+prior_px] = True
        ei = em.flatten().nonzero().squeeze()
        _coord_cache[key] = (gy[ei], gx[ei], ei)
    return _coord_cache[key]


# ---------------------------------------------------------------------------
# VECTORISED RAY-CASTING HELPERS
# ---------------------------------------------------------------------------

def _ray_directions(col_indices, n_cols_total, heading_deg, fov_deg, device):
    offsets = ((col_indices.float() / n_cols_total) - 0.5) * fov_deg
    angles  = torch.deg2rad(torch.tensor(heading_deg, device=device) + offsets)
    return torch.sin(angles), -torch.cos(angles)


def _grid_lookup(b_gy, b_gx, dx, dy, steps, grid, H, W):
    B, K, S = b_gy.shape[0], dx.shape[0], steps.shape[0]
    ry = b_gy.view(B,1,1) + steps.view(1,1,S) * dy.view(1,K,1)
    rx = b_gx.view(B,1,1) + steps.view(1,1,S) * dx.view(1,K,1)
    ry_idx = ry.to(torch.int32).clamp(0, H-1)
    rx_idx = rx.to(torch.int32).clamp(0, W-1)
    del ry, rx
    return grid[ry_idx, rx_idx]


def _apply_occlusion(vals):
    occ  = (vals == 1).cummax(dim=-1)[0]
    mask = torch.cat([
        torch.ones((*vals.shape[:-1], 1), dtype=torch.bool, device=vals.device),
        ~occ[..., :-1],
    ], dim=-1)
    return vals * mask


# ---------------------------------------------------------------------------
# FAST VECTORISED BASE HEATMAP
# ---------------------------------------------------------------------------

def compute_base_heatmap_fast(engine, grid, seg_query, heading, fov):
    config   = engine.config
    device   = engine.device
    H, W     = grid.shape
    pad      = int((config.tile_size - config.prior_size) / 2 / config.grid_res)
    n_cols   = seg_query.shape[1]
    n_steps  = int(80 / config.grid_res)

    gy_eval, gx_eval, eval_idx = get_eval_coords(H, W, pad, config.prior_px, device)
    n_eval = gy_eval.shape[0]

    col_indices = torch.arange(n_cols, device=device)
    dx, dy      = _ray_directions(col_indices, n_cols, heading, fov, device)
    steps       = torch.arange(0, 80, config.grid_res, device=device).float() / config.grid_res

    class_configs = [(1, 0.10, 1.0), (2, 0.10, 1.0), (3, 0.05, 2.0)]
    class_pcts, active = {}, {}
    for cv, thr, _ in class_configs:
        pct  = (seg_query == cv).float().mean(dim=0)
        mask = pct > thr
        if mask.any():
            class_pcts[cv] = pct
            active[cv]     = mask

    hm             = torch.zeros((H, W), device=device)
    bytes_per_pos  = n_cols * n_steps * 4 * 5
    pos_batch_size = max(1, min(n_eval, MEM_LIMIT_BYTES // bytes_per_pos))

    for i_start in range(0, n_eval, pos_batch_size):
        i_end = min(i_start + pos_batch_size, n_eval)
        b_idx = eval_idx[i_start:i_end]
        vals    = _grid_lookup(gy_eval[i_start:i_end], gx_eval[i_start:i_end],
                               dx, dy, steps, grid, H, W)
        visible = _apply_occlusion(vals)
        del vals
        score = torch.zeros(i_end - i_start, device=device)
        for cv, thr, rw in class_configs:
            if cv not in active:
                continue
            hit     = (visible == cv).any(dim=2)
            contrib = hit * active[cv] * (rw * class_pcts[cv])
            score  += contrib.sum(dim=1)
            del hit, contrib
        del visible
        hm.view(-1)[b_idx] = score

    return hm




# ---------------------------------------------------------------------------
# RESULT DATACLASS + METRICS
# ---------------------------------------------------------------------------

@dataclass
class HeadingResult:
    image_id:           str
    mode:               str
    heading_regime:     str
    gt_pixel:           Tuple[float, float]
    pred_pixel:         Tuple[int, int]
    heatmap_final:      torch.Tensor
    seg_time:           float = 0.0
    algo_time:          float = 0.0
    bracket_percentile: float = 0.0
    error_m:            float = 0.0

    @property
    def total_time(self):
        return self.seg_time + self.algo_time

    def compute_bracket_percentile(self, config):
        hm_flat  = self.heatmap_final.view(-1)
        total    = hm_flat.numel()
        gt_x     = max(0, min(int(self.gt_pixel[0]), config.tile_px - 1))
        gt_y     = max(0, min(int(self.gt_pixel[1]), config.tile_px - 1))
        gt_score = self.heatmap_final[gt_y, gt_x].item()
        n_above  = (hm_flat > gt_score).sum().item()
        n_tied   = (hm_flat == gt_score).sum().item()
        self.bracket_percentile = ((n_above + n_above + n_tied - 1) / 2.0 / total) * 100.0
        return self.bracket_percentile


def normalise(hm):
    mn, mx = hm.min(), hm.max()
    return (hm - mn) / (mx - mn + 1e-6)


def heatmap_to_result(hm_raw, gt_pixel, config, image_id, mode, heading_regime, seg_time, algo_time):
    hm_norm   = normalise(hm_raw)
    pred_flat = hm_norm.view(-1).argmax()
    pred_y    = (pred_flat // config.tile_px).item()
    pred_x    = (pred_flat  % config.tile_px).item()
    error_m   = float(math.sqrt((pred_x - gt_pixel[0])**2 + (pred_y - gt_pixel[1])**2) * config.grid_res)
    result    = HeadingResult(
        image_id=image_id, mode=mode, heading_regime=heading_regime,
        gt_pixel=gt_pixel, pred_pixel=(pred_x, pred_y),
        heatmap_final=hm_norm, seg_time=seg_time, algo_time=algo_time, error_m=error_m,
    )
    result.compute_bracket_percentile(config)
    return result


def compute_metrics(results: List[HeadingResult]) -> Dict[str, Any]:
    if not results:
        return {}
    percentiles = [r.bracket_percentile for r in results]
    errors_m    = [r.error_m            for r in results]
    metrics = {
        "n_samples":         len(results),
        "mean_seg_time_s":   float(np.mean([r.seg_time   for r in results])),
        "mean_algo_time_s":  float(np.mean([r.algo_time  for r in results])),
        "mean_total_time_s": float(np.mean([r.total_time for r in results])),
        "median_error_m":    float(np.median(errors_m)),
        "mean_error_m":      float(np.mean(errors_m)),
    }
    for t in [5, 10, 15, 20]:
        metrics[f"top_{t}_inclusion"] = float(np.mean([p <= t for p in percentiles]) * 100)
    for thresh in [1, 3, 5]:
        metrics[f"recall_{thresh}m"] = float(np.mean([e <= thresh for e in errors_m]) * 100)
    return metrics


def print_results_table(all_metrics, title="RESULTS"):
    regimes = list(all_metrics.keys())
    lw, cw  = 30, 20
    sep     = "=" * (lw + cw * len(regimes))
    print(f"\n{sep}\n{title}\n{sep}")
    print(f"{'Metric':<{lw}}", end="")
    for r in regimes:
        print(f"{r:<{cw}}", end="")
    print()
    print("-" * (lw + cw * len(regimes)))

    def row(label, key, fmt=".1f"):
        print(f"{label:<{lw}}", end="")
        for r in regimes:
            val = all_metrics[r].get(key, None)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                print(f"{'N/A':<{cw}}", end="")
            elif fmt == "d":
                print(f"{int(val):<{cw}}", end="")
            else:
                print(f"{val:<{cw}{fmt}}", end="")
        print()

    row("Samples",              "n_samples",         "d")
    print("-" * (lw + cw * len(regimes)))
    for t in [5, 10, 15, 20]:
        row(f"Top-{t}% inclusion (%)", f"top_{t}_inclusion")
    print("-" * (lw + cw * len(regimes)))
    for thresh in [1, 3, 5]:
        row(f"Recall @{thresh}m (%)",  f"recall_{thresh}m")
    print("-" * (lw + cw * len(regimes)))
    row("Median error (m)",     "median_error_m")
    row("Mean error (m)",       "mean_error_m")
    print("-" * (lw + cw * len(regimes)))
    row("Mean seg time (s)",    "mean_seg_time_s",  ".3f")
    row("Mean algo time (s)",   "mean_algo_time_s", ".3f")
    row("Mean total time (s)",  "mean_total_time_s",".3f")
    print(sep)


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def get_valid_samples(config, city_filter=None, env_labels=None, env_filter=None):
    print("\nLoading dataset...")
    with open(config.metadata_file) as f:
        metadata = json.load(f)

    seen = get_training_image_ids(config.yolo_labels_dir)
    valid = []
    for m in metadata:
        if str(m["id"]) in seen:
            continue
        if city_filter and m.get("city", "").lower() != city_filter.lower():
            continue
        if env_filter and env_labels:
            label = env_labels.get(str(m["id"]), {}).get("label", "")
            if label not in env_filter:
                continue

        city     = m["city"].replace(" ", "_").lower()
        map_path = os.path.join(config.osm_tile_dir, city, f"{m['id']}_map.geojson")
        if not os.path.exists(map_path):
            continue
        if not m.get("camera_parameters"):
            continue
        cg = m.get("computed_geometry")
        if not cg or not cg.get("coordinates"):
            continue
        valid.append(m)

    print(f"  Valid samples: {len(valid)}")
    return valid


# ---------------------------------------------------------------------------
# PROGRESS BAR
# ---------------------------------------------------------------------------

class ProgressBar:
    def __init__(self, total, desc="", width=40):
        self.total = total; self.desc = desc; self.width = width
        self.current = 0; self.start_time = time.time()

    def update(self, n=1):
        self.current += n
        pct     = self.current / self.total if self.total else 0
        filled  = int(self.width * pct)
        bar     = "█" * filled + "░" * (self.width - filled)
        elapsed = time.time() - self.start_time
        eta     = (elapsed / self.current) * (self.total - self.current) if self.current else 0
        sys.stdout.write(
            f"\r{self.desc}: |{bar}| {self.current}/{self.total} "
            f"({pct*100:.1f}%) ETA {int(eta//60):02d}:{int(eta%60):02d}  "
        )
        sys.stdout.flush()
        if self.current >= self.total:
            sys.stdout.write(
                f"\r{self.desc}: |{bar}| {self.current}/{self.total} "
                f"(100%) Done in {int(elapsed//60):02d}:{int(elapsed%60):02d}  \n"
            )


# ---------------------------------------------------------------------------
# MAIN EXPERIMENT LOOP
# ---------------------------------------------------------------------------

def run_experiments(engine, samples, config):
    results = {"planarity_occlusion": {"known": []}}

    print(f"\n{'='*65}")
    print(f"SEMANTIC RAY-CASTING POSITIONING  (planarity+occlusion, known heading)")
    print(f"  Samples: {len(samples)}")
    print(f"{'='*65}")

    progress = ProgressBar(len(samples), desc="Processing")
    first_sample = True

    for sample in samples:
        img_id   = str(sample["id"])
        city     = sample["city"].replace(" ", "_").lower()
        img_path = os.path.join(config.gv_dataset_dir, sample["local_path"])
        map_path = os.path.join(config.osm_tile_dir, city, f"{sample['id']}_map.geojson")
        fov      = calculate_hfov(sample.get("camera_parameters"))
        if fov is None:
            progress.update(); continue

        gps_lon, gps_lat   = sample["geometry"]["coordinates"]
        true_lon, true_lat = sample["computed_geometry"]["coordinates"]
        true_heading       = sample.get("compass_angle", 0.0)

        try:
            grid, (cx, cy), map_crs = engine.rasterize_vector_map(map_path, gps_lon, gps_lat)

            t_seg = time.time()
            seg_query, _, _, _, _, _ = engine.extract_query_features(
                img_path, fov,
                focal_length=sample.get("camera_parameters", [0.5])[0],
                compute_depth=False, compute_road_geometry=False,
            )
            seg_time = time.time() - t_seg

            true_pt  = gpd.GeoDataFrame(geometry=[Point(true_lon, true_lat)], crs="EPSG:4326").to_crs(map_crs)
            tx, ty   = true_pt.geometry.iloc[0].x, true_pt.geometry.iloc[0].y
            gt_x     = (tx - (cx - config.tile_size/2)) / config.grid_res
            gt_y     = config.tile_px - ((ty - (cy - config.tile_size/2)) / config.grid_res)
            gt_pixel = (gt_x, gt_y)

            t0      = time.time()
            hm      = compute_base_heatmap_fast(engine, grid, seg_query, true_heading, fov)
            algo_t  = time.time() - t0
            results["planarity_occlusion"]["known"].append(
                heatmap_to_result(hm, gt_pixel, config, img_id,
                                  "planarity_occlusion", "known", seg_time, algo_t)
            )

        except Exception as e:
            import traceback
            print(f"\n  Warning — {img_id}: {e}")
            if first_sample:
                traceback.print_exc()
        finally:
            if config.device == "cuda":
                torch.cuda.empty_cache()

        first_sample = False
        progress.update()

    return results


# ---------------------------------------------------------------------------
# ARGUMENT PARSING + MAIN
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Semantic ray-casting positioning (planarity+occlusion, known heading)")
    p.add_argument("--n-samples",  type=int, default=200)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--city",       type=str, default=None,
                   help="Filter to one city, e.g. 'London'")
    p.add_argument("--env-labels", type=str, default=None,
                   help="Path to environment_labels.json (from classify_environment.py)")
    p.add_argument("--env-filter", type=str, nargs="+", default=None,
                   help="Only run on images with these environment labels, e.g. URBAN SUBURBAN")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Override results output directory")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("\n" + "=" * 65)
    print("SEMANTIC RAY-CASTING POSITIONING")
    print("=" * 65)
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
          f"seed={args.seed}  samples={args.n_samples}")

    config = PositioningConfig()
    if args.output_dir:
        config.output_dir = args.output_dir
    os.makedirs(config.output_dir, exist_ok=True)

    model_path = find_best_weights(config.runs_dir)
    if not model_path:
        print("No trained YOLO weights found. Check that runs/ contains a best.pt file.")
        print(f"  Searched: {config.runs_dir}")
        return
    print(f"  Model: {model_path}")
    print(f"  Device: {config.device}")

    env_labels = None
    if args.env_labels and os.path.exists(args.env_labels):
        with open(args.env_labels) as f:
            env_labels = json.load(f)
        print(f"  Environment labels loaded: {len(env_labels)} images")

    engine        = PositioningEngine(model_path, config, load_depth_model=False)
    valid_samples = get_valid_samples(config, city_filter=args.city,
                                      env_labels=env_labels, env_filter=args.env_filter)
    n             = min(args.n_samples, len(valid_samples))
    test_samples  = random.sample(valid_samples, n)
    print(f"\nSelected {n} samples\n")

    all_results = run_experiments(engine, test_samples, config)

    po_metrics = {"planarity_occlusion (known)":
                  compute_metrics(all_results["planarity_occlusion"]["known"])}
    print_results_table(po_metrics, "PLANARITY+OCCLUSION — KNOWN HEADING")

    # Save JSON results
    metrics_out = {"planarity_occlusion": {"known": po_metrics["planarity_occlusion (known)"]}}
    with open(os.path.join(config.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    detail_out = {"planarity_occlusion": {"known": [
        {"image_id":           r.image_id,
         "bracket_percentile": r.bracket_percentile,
         "error_m":            r.error_m,
         "seg_time_s":         r.seg_time,
         "algo_time_s":        r.algo_time,
         "total_time_s":       r.total_time}
        for r in all_results["planarity_occlusion"]["known"]
    ]}}
    with open(os.path.join(config.output_dir, "results_detail.json"), "w") as f:
        json.dump(detail_out, f, indent=2)

    print(f"\nResults saved to {config.output_dir}/")
    print("  metrics.json        — aggregate metrics per regime")
    print("  results_detail.json — per-image results (feed into analyse_results.py)")
    print("Done.")


if __name__ == "__main__":
    main()
