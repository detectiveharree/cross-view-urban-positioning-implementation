"""
engine.py — Core positioning engine for semantic ray-casting.

Contains:
  - PositioningConfig   dataclass for all tuneable parameters
  - PositioningEngine   YOLO segmentation + ray-casting localisation
  - Evaluation metrics  (top-k inclusion, positioning error)
  - Helper utilities    (find_best_weights, calculate_hfov, …)

Adapted from the UCL PhD codebase (v1_utils.py).
"""

import os
import math
import torch
import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Tuple, Any


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.environ.get("DATA_DIR", os.path.join(_REPO_ROOT, "..", "data"))


@dataclass
class PositioningConfig:
    """Configuration for positioning experiments.

    Paths are resolved relative to DATA_DIR (env var) or ../data by default.
    """

    # Directories
    osm_tile_dir:    str = os.path.join(_DATA_ROOT, "osm_tiles")
    gv_dataset_dir:  str = os.path.join(_DATA_ROOT, "gv_dataset")
    metadata_file:   str = os.path.join(_DATA_ROOT, "gv_dataset", "metadata.json")
    output_dir:      str = "results/positioning"
    runs_dir:        str = os.path.join(_REPO_ROOT, "..", "runs", "segment", "positioning_research")
    yolo_labels_dir: str = os.path.join(_DATA_ROOT, "yolo_dataset_v1", "labels")

    # Grid parameters (physical metres)
    grid_res:   float = 0.5    # metres per pixel
    tile_size:  float = 300.0  # 300×300 m total map tile
    prior_size: float = 128.0  # 128×128 m evaluation window (centred)

    # Computation
    batch_size: int = 200_000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def tile_px(self) -> int:
        return int(self.tile_size / self.grid_res)

    @property
    def prior_px(self) -> int:
        return int(self.prior_size / self.grid_res)


class ExperimentMode(Enum):
    BASELINE            = "baseline"             # ray-casting, no occlusion
    PLANARITY_OCCLUSION = "planarity_occlusion"  # ray-casting + occlusion
    EXTENDED_CLASSES    = "extended_classes"     # + crosswalks
    INSTANCE_BEARING    = "instance_bearing"     # + building bearing histogram
    DEPTH_ORDERING      = "depth_ordering"       # + ZoeDepth rank correlation
    ROAD_GEOMETRY       = "road_geometry"        # + geometric road depth


# ---------------------------------------------------------------------------
# SEMANTIC CLASS MAPPINGS
# ---------------------------------------------------------------------------

# Grid values:   0 = background  |  1 = building  |  2 = road/sidewalk  |  3 = crosswalk
#
# YOLO classes (from fine-tuned model):
#   0: building    1: lamp post   2: tree        3: stop sign
#   4: fire hydrant               5: road        6: sidewalk   7: crosswalk
#
# Note: lamp post / tree / stop sign / fire hydrant are detected but not used
#       in ray-casting — they're interesting for environment characterisation.

YOLO_CLASS_NAMES = [
    "building", "lamp_post", "tree", "stop_sign",
    "fire_hydrant", "road", "sidewalk", "crosswalk",
]

YOLO_TO_GRID = {
    0: 1,  # building  → obstacle (causes occlusion)
    5: 2,  # road      → road surface
    6: 2,  # sidewalk  → road surface
    7: 3,  # crosswalk → distinct feature
}


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def find_best_weights(runs_dir: str) -> Optional[str]:
    """Return path to the most-recently-trained best.pt weights file."""
    if not os.path.exists(runs_dir):
        return None
    runs = [
        os.path.join(runs_dir, d)
        for d in os.listdir(runs_dir)
        if os.path.isdir(os.path.join(runs_dir, d))
    ]
    if not runs:
        return None
    runs.sort(key=os.path.getmtime, reverse=True)
    for run in runs:
        w = os.path.join(run, "weights", "best.pt")
        if os.path.exists(w):
            return w
    return None


def calculate_hfov(camera_parameters: Optional[List[float]]) -> Optional[float]:
    """Horizontal FOV (degrees) from Mapillary normalised focal length."""
    if not camera_parameters or len(camera_parameters) < 1:
        return None
    focal_length = camera_parameters[0]
    return math.degrees(2 * math.atan(0.5 / focal_length))


def get_training_image_ids(yolo_labels_dir: str) -> set:
    """IDs of images used during YOLO fine-tuning (exclude from eval)."""
    seen = set()
    for split in ["train", "val"]:
        d = os.path.join(yolo_labels_dir, split)
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith(".txt"):
                    seen.add(f.replace(".txt", ""))
    return seen


# ---------------------------------------------------------------------------
# EVALUATION METRICS
# ---------------------------------------------------------------------------

@dataclass
class PositioningResult:
    image_id:      str
    error_m:       float
    gt_pixel:      Tuple[float, float]
    pred_pixel:    Tuple[int, int]
    heatmap_final: torch.Tensor
    mode:          ExperimentMode
    gt_rank_percentile: float = 0.0

    def compute_inclusion_metrics(self, config: PositioningConfig):
        hm      = self.heatmap_final
        hm_flat = hm.view(-1)
        gt_x    = max(0, min(int(self.gt_pixel[0]), config.tile_px - 1))
        gt_y    = max(0, min(int(self.gt_pixel[1]), config.tile_px - 1))
        gt_val  = hm[gt_y, gt_x].item()

        higher_count          = (hm_flat > gt_val).sum().item()
        self.gt_rank_percentile = (higher_count / hm_flat.numel()) * 100

        self.distinguishable = {}
        for threshold in [5, 10, 15, 20]:
            top_n      = int(hm_flat.numel() * threshold / 100)
            top_scores, _ = hm_flat.topk(top_n)
            thresh_score  = top_scores[-1].item()
            self.distinguishable[threshold] = thresh_score > hm_flat.min().item() + 1e-6

        return self.gt_rank_percentile


@dataclass
class SampleData:
    image_id:          str
    img_path:          str
    grid:              torch.Tensor
    seg_query:         torch.Tensor
    full_mask:         np.ndarray
    gt_pixel:          Tuple[float, float]
    heading:           float
    fov:               float
    building_bearings: List[float]       = None
    depth_map:         np.ndarray        = None
    column_depths:     np.ndarray        = None
    road_distances:    np.ndarray        = None


def compute_aggregate_metrics(results: List[PositioningResult]) -> Dict[str, Any]:
    if not results:
        return {}
    errors      = [r.error_m            for r in results]
    percentiles = [r.gt_rank_percentile for r in results]
    metrics = {
        "mean_error_m":   float(np.mean(errors)),
        "median_error_m": float(np.median(errors)),
        "std_error_m":    float(np.std(errors)),
        "n_samples":      len(results),
    }
    for threshold in [5, 10, 15, 20]:
        metrics[f"top_{threshold}_inclusion"] = (
            len([r for r in results if r.gt_rank_percentile <= threshold]) / len(results) * 100
        )
    return metrics


# ---------------------------------------------------------------------------
# POSITIONING ENGINE
# ---------------------------------------------------------------------------

class PositioningEngine:
    """
    Semantic ray-casting localisation engine.

    Loads a fine-tuned YOLO segmentation model and, for each query image,
    extracts semantic features that are matched against a rasterised OSM map
    via ray-casting to produce a probability heatmap over candidate positions.
    """

    def __init__(self, model_path: str, config: PositioningConfig,
                 load_depth_model: bool = False):
        from ultralytics import YOLO
        print(f"Loading YOLO model: {model_path}")
        self.model  = YOLO(model_path)
        self.config = config
        self.device = config.device
        print(f"Model loaded on {self.device}")

        self.depth_model = None
        if load_depth_model:
            self._load_zoedepth()

    def _load_zoedepth(self):
        try:
            print("Loading ZoeDepth...")
            self.depth_model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)
            self.depth_model = self.depth_model.to(self.device).eval()
            print("ZoeDepth loaded (torch hub)")
        except Exception as e:
            try:
                from transformers import pipeline
                self.depth_model = pipeline(
                    task="depth-estimation", model="Intel/zoedepth-nyu-kitti",
                    device=0 if self.device == "cuda" else -1,
                )
                print("ZoeDepth loaded (transformers fallback)")
            except Exception as e2:
                print(f"Could not load ZoeDepth: {e2}")
                self.depth_model = None

    def estimate_depth(self, img_path: str) -> Optional[np.ndarray]:
        if self.depth_model is None:
            return None
        try:
            from PIL import Image
            image = Image.open(img_path)
            if hasattr(self.depth_model, "__call__") and not hasattr(self.depth_model, "infer"):
                depth_map = np.array(self.depth_model(image)["depth"])
            else:
                import torchvision.transforms as T
                img_t = T.ToTensor()(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    depth_map = self.depth_model.infer(img_t).squeeze().cpu().numpy()
            return depth_map
        except Exception as e:
            print(f"Depth estimation failed: {e}")
            return None

    def compute_road_geometry_depth(self, seg_mask: np.ndarray, focal_length: float,
                                    camera_height: float = 1.5) -> np.ndarray:
        h, w       = seg_mask.shape
        seg_small  = cv2.resize(seg_mask, (160, 90), interpolation=cv2.INTER_NEAREST)
        h_s, _     = seg_small.shape
        cy         = h_s / 2.0
        f_px       = focal_length * 160
        road_dist  = np.full(160, np.nan)
        for col in range(160):
            rows = np.where(seg_small[:, col] == 2)[0]
            if not len(rows):
                continue
            farthest = rows.min()
            if farthest <= cy:
                road_dist[col] = 100.0
                continue
            angle = np.arctan((farthest - cy) / f_px)
            road_dist[col] = min(camera_height / np.tan(angle), 100.0) if angle > 0.01 else 100.0
        return road_dist

    def rasterize_vector_map(self, geojson_path: str, center_lon: float,
                             center_lat: float) -> Tuple[torch.Tensor, Tuple[float, float], Any]:
        gdf = gpd.read_file(geojson_path)
        gdf = gdf.to_crs(gdf.estimate_utm_crs())

        center_pt = gpd.GeoDataFrame(
            geometry=[Point(center_lon, center_lat)], crs="EPSG:4326"
        ).to_crs(gdf.crs)
        cx, cy = center_pt.geometry.iloc[0].x, center_pt.geometry.iloc[0].y

        minx  = cx - self.config.tile_size / 2
        miny  = cy - self.config.tile_size / 2
        grid  = np.zeros((self.config.tile_px, self.config.tile_px), dtype=np.uint8)

        if "highway" in gdf.columns:
            primary    = ["motorway","motorway_link","trunk","trunk_link","primary","primary_link"]
            secondary  = ["secondary","secondary_link"]
            local      = ["tertiary","tertiary_link","unclassified","residential","service",
                          "road","living_street"]
            roads = gdf[gdf["highway"].isin(primary + secondary + local)]
            for _, row in roads.iterrows():
                if row.geometry is None:
                    continue
                w_m = (14 if row["highway"] in ["motorway","trunk"] else
                       10 if row["highway"] in primary[2:] else
                        8 if row["highway"] in secondary else 5)
                self._draw_poly(grid, row.geometry.buffer(w_m / 2), minx, miny, 2)
            for _, row in gdf[gdf["highway"] == "footway"].iterrows():
                if row.geometry is not None:
                    self._draw_poly(grid, row.geometry.buffer(2), minx, miny, 2)

        if "building" in gdf.columns:
            for _, row in gdf[gdf["building"].notnull()].iterrows():
                if row.geometry is not None:
                    self._draw_poly(grid, row.geometry, minx, miny, 1)

        if "highway" in gdf.columns:
            for _, row in gdf[gdf["highway"] == "crossing"].iterrows():
                if row.geometry is not None:
                    self._draw_poly(grid, row.geometry.buffer(3), minx, miny, 3)
        if "crossing" in gdf.columns:
            for _, row in gdf[gdf["crossing"].notnull()].iterrows():
                if row.geometry is not None:
                    self._draw_poly(grid, row.geometry.buffer(3), minx, miny, 3)

        return torch.from_numpy(grid).to(self.device), (cx, cy), gdf.crs

    def _draw_poly(self, grid: np.ndarray, poly, minx: float, miny: float, val: int):
        if poly.is_empty:
            return
        try:
            if not hasattr(poly, "exterior"):
                return
            coords = np.array(poly.exterior.coords)
            coords[:, 0] = (coords[:, 0] - minx) / self.config.grid_res
            coords[:, 1] = self.config.tile_px - (coords[:, 1] - miny) / self.config.grid_res
            cv2.fillPoly(grid, [coords.astype(np.int32).reshape(-1, 1, 2)], val)
        except Exception:
            pass

    def extract_query_features(self, img_path: str, fov: float,
                               focal_length: float = 0.5,
                               compute_depth: bool = False,
                               compute_road_geometry: bool = False):
        results = self.model.predict(img_path, conf=0.2, verbose=False)
        res     = results[0]
        h, w    = res.orig_shape

        class_mask        = np.zeros((h, w), dtype=np.uint8)
        building_bearings = []

        if res.masks:
            for i, mask in enumerate(res.masks.data):
                cls = int(res.boxes.cls[i])
                if cls in YOLO_TO_GRID:
                    m = cv2.resize(mask.cpu().numpy(), (w, h)) > 0
                    class_mask[m] = YOLO_TO_GRID[cls]
                if cls == 0:
                    box     = res.boxes.xywh[i].cpu().numpy()
                    bearing = ((box[0] / w) - 0.5) * fov
                    building_bearings.append(bearing)

        seg_small    = cv2.resize(class_mask, (160, 90), interpolation=cv2.INTER_NEAREST)
        depth_map    = None
        column_depths = None
        road_distances = None

        if compute_depth and self.depth_model is not None:
            depth_map = self.estimate_depth(img_path)
            if depth_map is not None:
                if depth_map.shape[:2] != (h, w):
                    depth_map = cv2.resize(depth_map, (w, h))
                depth_small   = cv2.resize(depth_map, (160, 90))
                seg_for_depth = cv2.resize(class_mask, (160, 90), interpolation=cv2.INTER_NEAREST)
                column_depths = np.full(160, np.nan)
                for col in range(160):
                    bm = seg_for_depth[:, col] == 1
                    if bm.any():
                        column_depths[col] = np.median(depth_small[bm, col])

        if compute_road_geometry:
            road_distances = self.compute_road_geometry_depth(class_mask, focal_length)

        return (
            torch.from_numpy(seg_small).to(self.device),
            class_mask,
            building_bearings,
            depth_map,
            column_depths,
            road_distances,
        )

    def compute_heatmap(self, grid: torch.Tensor, seg_query: torch.Tensor,
                        heading: float, fov: float, mode: ExperimentMode,
                        building_bearings=None, column_depths=None,
                        road_distances=None) -> torch.Tensor:
        H, W = grid.shape
        hm   = torch.zeros((H, W), device=self.device)

        total_cols = seg_query.shape[1]
        steps = (torch.arange(0, 80, self.config.grid_res, device=self.device).float()
                 / self.config.grid_res)
        pad   = int((self.config.tile_size - self.config.prior_size) / 2 / self.config.grid_res)

        gy, gx = torch.meshgrid(
            torch.arange(H, device=self.device),
            torch.arange(W, device=self.device),
            indexing="ij",
        )
        gy, gx = gy.flatten().float(), gx.flatten().float()

        eval_mask = torch.zeros((H, W), dtype=torch.bool, device=self.device)
        eval_mask[pad:pad + self.config.prior_px, pad:pad + self.config.prior_px] = True
        eval_idx = eval_mask.flatten().nonzero().squeeze()

        use_occlusion = mode != ExperimentMode.BASELINE
        extended = mode in [ExperimentMode.EXTENDED_CLASSES, ExperimentMode.INSTANCE_BEARING,
                            ExperimentMode.DEPTH_ORDERING, ExperimentMode.ROAD_GEOMETRY]
        class_configs = (
            [(1, 0.1, 1.0, 0.3, True), (2, 0.1, 1.0, 0.0, False), (3, 0.05, 2.0, 0.5, True)]
            if extended else
            [(1, 0.1, 1.0, 0.3, True), (2, 0.1, 1.0, 0.0, False)]
        )

        for col in range(total_cols):
            seg_col = seg_query[:, col]
            angle   = math.radians(heading + ((col / total_cols) - 0.5) * fov)
            dx, dy  = math.sin(angle), -math.cos(angle)

            for i in range(0, eval_idx.shape[0], self.config.batch_size):
                b_idx = eval_idx[i:i + self.config.batch_size]
                ry = gy[b_idx].unsqueeze(1) + steps.unsqueeze(0) * dy
                rx = gx[b_idx].unsqueeze(1) + steps.unsqueeze(0) * dx
                vals = grid[ry.long().clamp(0, H-1), rx.long().clamp(0, W-1)]

                if use_occlusion:
                    occ     = (vals == 1).cummax(dim=1)[0]
                    visible = vals * torch.cat(
                        [torch.ones((vals.shape[0], 1), device=self.device, dtype=torch.bool),
                         ~occ[:, :-1]], dim=1
                    ).long()
                else:
                    visible = vals

                for cls_val, threshold, reward, penalty, apply_penalty in class_configs:
                    cls_pct = (seg_col == cls_val).float().mean()
                    if cls_pct > threshold:
                        hit = (visible == cls_val).any(dim=1)
                        hm.view(-1)[b_idx[hit]] += reward * cls_pct.item()
                        if apply_penalty and use_occlusion and cls_pct > 0.5:
                            hm.view(-1)[b_idx[~hit]] -= penalty * cls_pct.item()

        if mode == ExperimentMode.INSTANCE_BEARING and building_bearings:
            hm = hm * (1.0 + self._compute_bearing_score(grid, heading, fov, building_bearings))

        if mode == ExperimentMode.DEPTH_ORDERING and column_depths is not None:
            if (~np.isnan(column_depths)).sum() >= 5:
                hm = hm * (1.0 + self._compute_depth_score(grid, heading, fov, column_depths))

        if mode == ExperimentMode.ROAD_GEOMETRY and road_distances is not None:
            if (~np.isnan(road_distances)).sum() >= 5:
                hm = hm * (1.0 + self._compute_road_score(grid, heading, fov, road_distances))

        return hm

    def _compute_bearing_score(self, grid, heading, fov, detected_bearings,
                               bearing_tolerance=10.0):
        H, W = grid.shape
        hm   = torch.zeros((H, W), device=self.device)
        pad  = int((self.config.tile_size - self.config.prior_size) / 2 / self.config.grid_res)
        steps = (torch.arange(0, 80, self.config.grid_res, device=self.device).float()
                 / self.config.grid_res)
        gy, gx = torch.meshgrid(torch.arange(H, device=self.device),
                                 torch.arange(W, device=self.device), indexing="ij")
        gy, gx = gy.flatten().float(), gx.flatten().float()
        em = torch.zeros((H, W), dtype=torch.bool, device=self.device)
        em[pad:pad+self.config.prior_px, pad:pad+self.config.prior_px] = True
        eval_idx = em.flatten().nonzero().squeeze()

        bearing_samples = torch.arange(-fov/2, fov/2+1, 2.0, device=self.device)
        detected        = torch.tensor(detected_bearings, device=self.device)

        for i in range(0, eval_idx.shape[0], self.config.batch_size):
            b_idx = eval_idx[i:i+self.config.batch_size]
            hits  = torch.zeros((len(b_idx), len(bearing_samples)), device=self.device)
            for bi, bo in enumerate(bearing_samples):
                angle   = math.radians(heading + bo.item())
                dx, dy  = math.sin(angle), -math.cos(angle)
                ry = gy[b_idx].unsqueeze(1) + steps.unsqueeze(0) * dy
                rx = gx[b_idx].unsqueeze(1) + steps.unsqueeze(0) * dx
                vals = grid[ry.long().clamp(0,H-1), rx.long().clamp(0,W-1)]
                occ  = (vals==1).cummax(dim=1)[0]
                vis  = vals * torch.cat(
                    [torch.ones((vals.shape[0],1), device=self.device, dtype=torch.bool),
                     ~occ[:,:-1]], dim=1).long()
                hits[:, bi] = (vis==1).any(dim=1).float()
            score = torch.zeros(len(b_idx), device=self.device)
            for db in detected:
                close   = (bearing_samples - db).abs() < bearing_tolerance
                matched = (hits[:, close].sum(dim=1) > 0).float()
                score  += matched
            if len(detected) > 0:
                score /= len(detected)
            hm.view(-1)[b_idx] = score
        return hm

    def _compute_depth_score(self, grid, heading, fov, column_depths):
        H, W = grid.shape
        hm   = torch.zeros((H, W), device=self.device)
        valid_cols   = np.where(~np.isnan(column_depths))[0]
        image_depths = column_depths[valid_cols]
        image_ranks  = np.argsort(np.argsort(image_depths)).astype(np.float32)
        image_ranks_t = torch.tensor(image_ranks, device=self.device)

        pad   = int((self.config.tile_size - self.config.prior_size) / 2 / self.config.grid_res)
        steps = (torch.arange(0,80,self.config.grid_res,device=self.device).float()
                 / self.config.grid_res)
        gy, gx = torch.meshgrid(torch.arange(H,device=self.device),
                                 torch.arange(W,device=self.device), indexing="ij")
        gy, gx = gy.flatten().float(), gx.flatten().float()
        em = torch.zeros((H,W), dtype=torch.bool, device=self.device)
        em[pad:pad+self.config.prior_px, pad:pad+self.config.prior_px] = True
        eval_idx = em.flatten().nonzero().squeeze()
        n_valid  = len(valid_cols)

        for i in range(0, eval_idx.shape[0], self.config.batch_size):
            b_idx    = eval_idx[i:i+self.config.batch_size]
            map_dist = torch.full((len(b_idx), n_valid), float("inf"), device=self.device)
            for vi, col in enumerate(valid_cols):
                angle  = math.radians(heading + ((col/160)-0.5)*fov)
                dx, dy = math.sin(angle), -math.cos(angle)
                ry = gy[b_idx].unsqueeze(1) + steps.unsqueeze(0)*dy
                rx = gx[b_idx].unsqueeze(1) + steps.unsqueeze(0)*dx
                vals = grid[ry.long().clamp(0,H-1), rx.long().clamp(0,W-1)]
                ib   = vals==1
                fh   = ib.float().argmax(dim=1)
                has  = ib.any(dim=1)
                d    = fh.float() * self.config.grid_res
                d[~has] = 9999.0
                map_dist[:, vi] = d
            map_ranks  = torch.argsort(torch.argsort(map_dist, dim=1), dim=1).float()
            diff       = map_ranks - image_ranks_t.unsqueeze(0)
            n          = float(n_valid)
            spearman   = 1.0 - 6.0*(diff**2).sum(dim=1)/(n*(n**2-1))
            hm.view(-1)[b_idx] = (spearman + 1.0) / 2.0
        return hm

    def _compute_road_score(self, grid, heading, fov, road_distances):
        H, W = grid.shape
        hm   = torch.zeros((H, W), device=self.device)
        valid_cols = np.where(~np.isnan(road_distances))[0]
        n_valid    = len(valid_cols)
        if n_valid < 5:
            return hm
        pad   = int((self.config.tile_size - self.config.prior_size) / 2 / self.config.grid_res)
        steps = (torch.arange(0,80,self.config.grid_res,device=self.device).float()
                 / self.config.grid_res)
        gy, gx = torch.meshgrid(torch.arange(H,device=self.device),
                                 torch.arange(W,device=self.device), indexing="ij")
        gy, gx = gy.flatten().float(), gx.flatten().float()
        em = torch.zeros((H,W), dtype=torch.bool, device=self.device)
        em[pad:pad+self.config.prior_px, pad:pad+self.config.prior_px] = True
        eval_idx = em.flatten().nonzero().squeeze()
        for i in range(0, eval_idx.shape[0], self.config.batch_size):
            b_idx      = eval_idx[i:i+self.config.batch_size]
            violations = torch.zeros(len(b_idx), device=self.device)
            for col in valid_cols:
                rd     = road_distances[col]
                angle  = math.radians(heading + ((col/160)-0.5)*fov)
                dx, dy = math.sin(angle), -math.cos(angle)
                ry = gy[b_idx].unsqueeze(1) + steps.unsqueeze(0)*dy
                rx = gx[b_idx].unsqueeze(1) + steps.unsqueeze(0)*dx
                vals = grid[ry.long().clamp(0,H-1), rx.long().clamp(0,W-1)]
                ib   = vals==1
                fh   = ib.float().argmax(dim=1)
                has  = ib.any(dim=1)
                md   = fh.float() * self.config.grid_res
                md[~has] = 9999.0
                violations += ((md < rd - 5.0) & has).float()
            hm.view(-1)[b_idx] = torch.clamp(1.0 - violations/n_valid, 0.0, 1.0)
        return hm

    def localize_all_modes(self, sample_data: SampleData,
                           modes: List[ExperimentMode]) -> Dict[ExperimentMode, PositioningResult]:
        results = {}
        for mode in modes:
            hm      = self.compute_heatmap(
                sample_data.grid, sample_data.seg_query,
                sample_data.heading, sample_data.fov, mode,
                building_bearings=sample_data.building_bearings,
                column_depths=sample_data.column_depths,
                road_distances=sample_data.road_distances,
            )
            hm_norm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-6)
            pf      = hm_norm.view(-1).argmax()
            py, px  = (pf // self.config.tile_px).item(), (pf % self.config.tile_px).item()
            error   = math.sqrt((px - sample_data.gt_pixel[0])**2 +
                                (py - sample_data.gt_pixel[1])**2) * self.config.grid_res
            result  = PositioningResult(
                image_id=sample_data.image_id, error_m=error,
                gt_pixel=sample_data.gt_pixel, pred_pixel=(px, py),
                heatmap_final=hm_norm, mode=mode,
            )
            result.compute_inclusion_metrics(self.config)
            results[mode] = result
        return results
