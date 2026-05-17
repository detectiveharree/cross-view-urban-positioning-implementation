# MSc Cross-View Positioning — Research Toolkit

This repository gives you everything you need to run, evaluate, and analyse the semantic ray-casting cross-view positioning algorithm across different environments.

**Supervisor:** Harrison Reeves (UCL)

---

## What this does

The algorithm takes a street-level photograph and a rough GPS position, and figures out more precisely where the camera is — without any GPS refinement or machine learning at inference time. It works by:

1. Segmenting the photograph with a fine-tuned YOLO model to identify buildings, roads, crosswalks, etc.
2. Downloading a small patch of OpenStreetMap data around the GPS prior.
3. Casting rays from every candidate position in the map and scoring how well they predict the observed segmentation.

The output is a probability heatmap over candidate positions. Your job is to understand **when and why this works, and when it doesn't** — which depends heavily on the environment.

---

## Setup

### 1. Prerequisites

- Python 3.10+
- A GPU is strongly recommended for the positioning algorithm (CPU works but is slow)
- ~5 GB free disk space for models and data

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API access

```bash
cp .env.example .env
# Edit .env and add your Mapillary API token (ask your supervisor)
```

### 4. Get the YOLO model weights

The fine-tuned segmentation model is not included in this repository (it's too large for Git). Ask your supervisor for the `best.pt` file and place it at:

```
../runs/segment/positioning_research/mapillary_benchmark_v12/weights/best.pt
```

Or pass `--model /path/to/best.pt` to any script that needs it.

---

## Quick start

The datasets (images + OSM tiles) are already downloaded and live at `../data/`. You can run the positioning algorithm immediately:

```bash
# Run on 50 images from the UK dataset (fast test)
python run_positioning.py --n-samples 50

# Run on 200 images, only in London
python run_positioning.py --n-samples 200 --city London

# Check results
cat results/positioning/metrics.json
```

---

## Scripts

### `run_positioning.py` — Main experiment

Runs the positioning algorithm and reports accuracy metrics.

```bash
python run_positioning.py --n-samples 200 --windows 5 10 20 45
```

**Key metric — bracket percentile:**
For each image, the algorithm scores every candidate position on the map. The ground-truth location's rank in that score distribution is the bracket percentile. If the GT is in the top 5% of scores, that counts as a "top-5% inclusion". This is the main metric — lower is better, and you'll see it reported as "Top-5/10/15/20% inclusion rate (%)".

**Heading modes:**
The algorithm can be given a perfect heading (`known`), a heading within ±N degrees (`window_Ndeg`), or no heading at all (`free`). The `--windows` argument controls which uncertainty levels are tested.

**Filtering by environment:**
Once you've run `classify_environment.py`, you can filter the experiment to specific environments:

```bash
python run_positioning.py --n-samples 200 \
    --env-labels results/environment_labels.json \
    --env-filter URBAN SUBURBAN
```

---

### `classify_environment.py` — Label images by environment type

Uses [CLIP](https://openai.com/research/clip) to classify each image as URBAN, SUBURBAN, RURAL, INDUSTRIAL, or WATERFRONT. CLIP is a vision-language model — no training required, it just scores image–text similarity.

```bash
python classify_environment.py                    # all images
python classify_environment.py --city Edinburgh   # one city
python classify_environment.py --n-samples 500    # subset
```

**Customising labels:** Open `classify_environment.py` and edit `ENVIRONMENT_PROMPTS`. You can add, remove, or reword labels — the more specific the text prompt, the better the discrimination. For example, you might add `"PARK": "a street next to a large public park or green space"`.

---

### `segment_images.py` — Per-class coverage statistics

Runs YOLO on each image and records what fraction of the image each semantic class covers. Useful for understanding what makes an environment "easy" or "hard" for the positioning algorithm.

```bash
python segment_images.py --n-samples 500
python segment_images.py --city Bristol
```

Output: `results/coverage.csv`

**YOLO classes detected (8 total):**

| Class | Name | Used in positioning? |
|-------|------|---------------------|
| 0 | building | Yes — primary cue (occlusion) |
| 1 | lamp post | No — detected but ignored |
| 2 | tree | No — detected but ignored |
| 3 | stop sign | No — detected but ignored |
| 4 | fire hydrant | No — detected but ignored |
| 5 | road | Yes — road surface cue |
| 6 | sidewalk | Yes — road surface cue |
| 7 | crosswalk | Yes — distinctive feature |

Classes 1–4 are interesting for environment characterisation even though they're not used in positioning. A scene full of trees and no buildings gives the algorithm almost nothing to work with.

---

### `analyse_results.py` — Analyse by environment

Joins positioning results with environment labels and coverage stats, produces plots and a merged CSV.

```bash
python analyse_results.py
python analyse_results.py --mode depth_weighted --regime window_10deg
```

Outputs:
- `results/plots/accuracy_by_environment.png` — bar charts
- `results/plots/inclusion_curves.png` — top-N% curves per environment
- `results/plots/coverage_by_environment.png` — YOLO class breakdown
- `results/environment_analysis.csv` — full merged table (open in Excel or R)

---

### `download_mapillary.py` — Download new images

Download Mapillary images for new cities or regions.

```bash
python download_mapillary.py --cities "Leeds" "Newcastle"
python download_mapillary.py --bbox -3.19 51.47 -3.16 51.50 --output-dir data/cardiff_extra
```

---

### `download_osm.py` — Download OSM map tiles

After downloading new images, download the corresponding OSM tiles.

```bash
python download_osm.py --city Leeds
python download_osm.py --n-images 200
```

---

## Ideas for testing in different environments

Here are several axes along which you can characterise and compare environments. Mix and match these for your analysis.

### 1. Geographic location

The simplest axis — compare different UK cities, or compare UK (mostly temperate, brick-heavy) with the European cities in the `gv_dataset`. Cities differ in:
- Building density and height
- Street grid regularity (London vs Edinburgh)
- Road width and type

```bash
python run_positioning.py --city London --n-samples 100
python run_positioning.py --city Edinburgh --n-samples 100
```

### 2. Mapillary metadata (camera / capture conditions)

The metadata saved per image includes several interesting fields you can slice on. Load `data/uk_dataset/metadata.json` and inspect:

| Field | What it tells you |
|-------|------------------|
| `camera_type` | `PERSPECTIVE`, `FISHEYE`, `SPHERICAL` — different FOV assumptions |
| `captured_at` | Unix timestamp (ms) — convert to time of day / season |
| `make` / `model` | Camera manufacturer and model |
| `compass_angle` | Reported heading — compare with algorithm's heading sensitivity |
| `camera_parameters[0]` | Normalised focal length — determines horizontal FOV |

Example: does the algorithm perform better at certain times of day? With certain camera types?

```python
import json, datetime
with open("../data/uk_dataset/metadata.json") as f:
    meta = json.load(f)

for m in meta[:5]:
    ts = m.get("captured_at", 0) / 1000
    dt = datetime.datetime.utcfromtimestamp(ts)
    print(m["id"], m.get("camera_type"), dt.strftime("%H:%M"), m.get("make"))
```

### 3. OpenStreetMap feature availability

Not all areas have equally complete OSM data. In dense urban centres OSM is very accurate; in rural areas building footprints may be missing. You can measure this by counting features per tile:

```python
import geopandas as gpd, glob, os

tile_dir = "../data/osm_tiles/london"
for path in glob.glob(f"{tile_dir}/*_map.geojson")[:5]:
    gdf = gpd.read_file(path)
    n_buildings = gdf["building"].notna().sum() if "building" in gdf.columns else 0
    n_roads = gdf["highway"].notna().sum() if "highway" in gdf.columns else 0
    print(os.path.basename(path), f"buildings={n_buildings} roads={n_roads}")
```

### 4. YOLO class coverage (from `segment_images.py`)

After running `segment_images.py`, you can group images by their visual content:
- **Building-heavy** scenes (urban, industrial): the algorithm has strong cues
- **Tree-heavy** scenes (parks, suburban avenues): trees are detected but ignored
- **Road-heavy, open scenes** (rural, motorways): little distinctive structure

Try correlating per-class coverage with positioning accuracy in `results/coverage.csv`.

### 5. CLIP environment classification (from `classify_environment.py`)

The CLIP classifier gives you a coarse environment label per image. You can:
- Compare accuracy across URBAN / SUBURBAN / RURAL / INDUSTRIAL / WATERFRONT
- Vary the label set — add `PARK`, `COMMERCIAL`, `HERITAGE` etc.
- Use the continuous CLIP scores (not just the argmax) to create a softer environment embedding

### 6. Heading sensitivity

The algorithm is tested under five heading conditions (known → free). You might find that certain environments are more sensitive to heading uncertainty than others — e.g. long straight roads give strong heading cues, whereas complex junctions might confuse the sweep.

Compare `top_10_inclusion` for `known` vs `free` across different cities or environment types.

---

## Data structure

```
../data/
├── uk_dataset/
│   ├── metadata.json          ← all image metadata (GPS, heading, camera type, etc.)
│   ├── london/                ← images: {image_id}.jpg
│   ├── manchester/
│   └── ...
├── osm_tiles/
│   ├── london/                ← tiles: {image_id}_map.geojson + _preview.png
│   └── ...
├── gv_dataset/                ← alternative dataset (European + US cities, Google SV)
└── yolo_dataset_v1/           ← fine-tuning annotations (training images excluded from eval)

results/
├── positioning/
│   ├── metrics.json           ← aggregate metrics per heading regime
│   └── results_detail.json    ← per-image: image_id, bracket_percentile, error_m
├── coverage.csv               ← per-image YOLO class coverage
├── environment_labels.json    ← per-image CLIP environment label
└── environment_analysis.csv   ← joined table for custom analysis
```

---

## Suggested workflow

1. **Explore the data** — load `metadata.json`, look at the images, check what camera types and cities are represented.
2. **Run a quick baseline** — `python run_positioning.py --n-samples 50` to see that everything works.
3. **Classify environments** — `python classify_environment.py --n-samples 1000` to get environment labels.
4. **Compute coverage** — `python segment_images.py --n-samples 500` to get YOLO coverage stats.
5. **Run the full experiment** — `python run_positioning.py --n-samples 500` (takes a while).
6. **Analyse** — `python analyse_results.py` to see accuracy broken down by environment.
7. **Dig deeper** — use `results/environment_analysis.csv` in your analysis tool of choice.
