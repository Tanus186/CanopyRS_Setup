# =============================================================================
# STEPS 7.1 → 7.2 → 7.3 → 7.4 → 7.5 → 7.6 → 8  —  ALL IN ONE
# =============================================================================
# WHY THIS FILE EXISTS:
#   The individual step files (step_7_1, step_7_3, etc.) are for READING and
#   UNDERSTANDING what each Colab cell does. But you can't run them separately
#   from CMD and share variables between them (backbone, results, sample_df etc.
#   would disappear between runs).
#
#   This file runs everything in one go, in the correct order.
#   It is identical code to the individual step files — just combined.
#
# HOW TO RUN:
#   (canopyrs) D:\Tanushree> python step_7_and_8_ALL_IN_ONE.py
#
# BEFORE RUNNING:
#   Make sure D:\Tanushree\bci-crown-model exists with these 4 items inside:
#     - dinov3-14-0.000.ckpt
#     - class_names_ordered.json
#     - dinov3_vitb16_pretrain_lvd1689m
#     - dinov3_repo\   (folder)
#   Download these on your laptop and copy to HPC (see step_7_1 comments).
#
# WHAT GETS SAVED:
#   D:\Tanushree\p1\crowns_with_species.gpkg   <- final result, open in QGIS
# =============================================================================

# ── Imports ───────────────────────────────────────────────────────────────────
import glob
import json
import json as _json
import logging
import math
import os
import subprocess
import sys
import warnings
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('TkAgg')   # Windows: show plots in popup window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm.auto import tqdm

# Install extra packages if needed
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "huggingface_hub", "torchmetrics", "pycocotools", "tqdm"],
    check=True
)
from pycocotools import mask as mask_utils

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION — edit these paths if needed
# =============================================================================
OUTPUT_DIR  = r"D:\Tanushree\p1"                 # where infer.py wrote its output
MODEL_DIR   = r"D:\Tanushree\bci-crown-model"    # where you copied the HF model files

# =============================================================================
# ─────────────────────────────────────────────────────────────────────────────
# STEP 7.1 — Load the Classification Model
# Colab: "## 7.1 Download the Classification Model"
# ─────────────────────────────────────────────────────────────────────────────
# =============================================================================
print("=" * 60)
print("STEP 7.1 — Loading classification model")
print("=" * 60)

_local              = Path(MODEL_DIR)
CKPT_PATH           = _local / "dinov3-14-0.000.ckpt"
CLASS_NAMES_PATH    = _local / "class_names_ordered.json"
DINOV3_WEIGHTS_PATH = _local / "dinov3_vitb16_pretrain_lvd1689m"
DINOV3_REPO_PATH    = _local / "dinov3_repo"

# Check all files exist
print("Checking model files...")
all_found = True
for label, path in [
    ("Checkpoint",       CKPT_PATH),
    ("Class names",      CLASS_NAMES_PATH),
    ("DINOv3 weights",   DINOV3_WEIGHTS_PATH),
    ("DINOv3 repo",      DINOV3_REPO_PATH),
]:
    exists = path.exists()
    print(f"  [{'OK' if exists else 'MISSING'}]  {label}: {path}")
    if not exists:
        all_found = False

if not all_found:
    print("\nERROR: Copy bci-crown-model folder from your laptop to:", MODEL_DIR)
    sys.exit(1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU:    {torch.cuda.get_device_name(0)}")

# =============================================================================
# STEP 7.2 — Load Species Names & Build the Model
# Colab: "## 7.2 Load Species Names & Build the Model"
# =============================================================================
print("\n" + "=" * 60)
print("STEP 7.2 — Loading species names and building model")
print("=" * 60)

# Load the list of 84 tropical tree species
class_names = json.loads(CLASS_NAMES_PATH.read_text())
print(f"{len(class_names)} species loaded")
print("First 5:", class_names[:5])
print("Last 5: ", class_names[-5:])

# Load the DINOv3 backbone from local code (no internet)
print("\nLoading DINOv3 backbone...")
backbone = torch.hub.load(
    str(DINOV3_REPO_PATH),
    model="dinov3_vitb16",
    source="local",
    weights=str(DINOV3_WEIGHTS_PATH),
    check_hash=False,
)

# Attach the classifier head: Dropout(0.1) + Linear(768 -> 84 species)
classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(768, 84))

# Load fine-tuned weights from the checkpoint
print("Loading fine-tuned checkpoint weights...")
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
sd   = ckpt["state_dict"]

backbone.load_state_dict(
    {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
)
classifier.load_state_dict(
    {k[len("classifier."):]: v for k, v in sd.items() if k.startswith("classifier.")}
)

backbone   = backbone.eval().to(DEVICE)
classifier = classifier.eval().to(DEVICE)

n_params = (
    sum(p.numel() for p in backbone.parameters()) +
    sum(p.numel() for p in classifier.parameters())
) / 1e6
print(f"Model loaded — {n_params:.1f}M parameters on {DEVICE}")

# =============================================================================
# STEP 7.3 — Load Crown Tiles from the Pipeline Output
# Colab: "## 7.3 Load Crown Tiles from the Pipeline Output"
# =============================================================================
print("\n" + "=" * 60)
print("STEP 7.3 — Loading crown tiles and computing masks")
print("=" * 60)

crown_tiles_dir = os.path.join(OUTPUT_DIR, "5_tilerizer")

# Find all crown tile .tif files
tile_paths = sorted(glob.glob(os.path.join(crown_tiles_dir, "**", "*.tif"), recursive=True))
print(f"Found {len(tile_paths)} crown tiles")

if not tile_paths:
    print("ERROR: No tiles in 5_tilerizer. Did the pipeline finish with all 6 steps?")
    sys.exit(1)

# Find and load the COCO JSON (stores polygon shapes for each crown)
coco_jsons = sorted(glob.glob(os.path.join(crown_tiles_dir, "**", "*_coco_*.json"), recursive=True))
if not coco_jsons:
    coco_jsons = sorted(glob.glob(os.path.join(crown_tiles_dir, "**", "coco.json"), recursive=True))
coco_json_path = coco_jsons[0]
with open(coco_json_path) as f:
    coco = _json.load(f)
print(f"Loaded COCO annotations: {len(coco['images'])} images, {len(coco['annotations'])} annotations")

# Build lookup: filename -> list of annotation dicts
img_id_to_info = {im['id']: im for im in coco['images']}
file_to_anns   = {}
for ann in coco['annotations']:
    fname = img_id_to_info[ann['image_id']]['file_name']
    file_to_anns.setdefault(fname, []).append(ann)

sample_df = pd.DataFrame({
    'tile_path_abs': tile_paths,
    'tile_name':     [Path(p).name for p in tile_paths],
})

# Build binary masks: 1 = inside crown polygon, 0 = outside
MASK_BUFFER_PX = 0
kernel        = cv2.getStructuringElement(
    cv2.MORPH_RECT, (MASK_BUFFER_PX * 2 + 1, MASK_BUFFER_PX * 2 + 1)
)
raw_masks     = {}
dilated_masks = {}

for _, row in sample_df.iterrows():
    name = row['tile_name']
    anns = file_to_anns.get(name, [])
    if not anns:
        raw_masks[name] = dilated_masks[name] = None
        continue
    h    = img_id_to_info[anns[0]['image_id']]['height']
    w    = img_id_to_info[anns[0]['image_id']]['width']
    mask = np.zeros((h, w), dtype=np.uint8)
    for ann in anns:
        seg = ann['segmentation']
        if isinstance(seg, dict):
            mask = np.maximum(mask, mask_utils.decode(seg).astype(np.uint8))
        else:
            for poly in (seg if isinstance(seg[0], list) else [seg]):
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2).round().astype(np.int32)
                cv2.fillPoly(mask, [pts], 1)
    raw_masks[name]     = mask
    dilated_masks[name] = cv2.dilate(mask, kernel)

print(f"Pre-computed {len(raw_masks)} crown masks")

def mask_outside_polygon(img_np, tile_name):
    """Black out all pixels outside the crown polygon mask."""
    m = dilated_masks.get(tile_name)
    if m is None:
        return img_np
    return img_np * m[:, :, np.newaxis]

# Visualization: show 6 sample crown tiles (top=outline, bottom=masked)
n_preview = min(6, len(tile_paths))
fig, axes = plt.subplots(2, n_preview, figsize=(3 * n_preview, 6))
if n_preview == 1:
    axes = axes.reshape(2, 1)
for i in range(n_preview):
    img_arr   = np.array(Image.open(tile_paths[i]).convert('RGB'))
    tile_name = Path(tile_paths[i]).name
    img_outline = img_arr.copy()
    m = raw_masks.get(tile_name)
    if m is not None:
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_outline, contours, -1, (255, 255, 255), 2)
    axes[0, i].imshow(img_outline)
    axes[0, i].set_title(tile_name, fontsize=6)
    axes[0, i].axis('off')
    axes[1, i].imshow(mask_outside_polygon(img_arr, tile_name))
    axes[1, i].axis('off')
axes[0, 0].set_ylabel('Crown outline', fontsize=9)
axes[1, 0].set_ylabel(f'Masked (buf={MASK_BUFFER_PX}px)', fontsize=9)
plt.suptitle('Step 7.3 — Crown Tiles Preview', fontsize=12)
plt.tight_layout()
plt.show()

# =============================================================================
# STEP 7.4 — Run Species Inference
# Colab: "## 7.4 Run Species Inference"
# =============================================================================
print("\n" + "=" * 60)
print("STEP 7.4 — Running species inference")
print("=" * 60)

IMAGE_SIZE = 512
BATCH_SIZE = 32   # reduce to 8 if GPU runs out of memory

transform = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE * 1.15), interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Pre-process: mask + transform every crown tile
print("Pre-processing tiles (masking + resizing)...")
all_tensors = []
all_names   = []
for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc='Preparing tiles'):
    img = np.array(Image.open(row['tile_path_abs']).convert('RGB'))
    img = mask_outside_polygon(img, row['tile_name'])
    all_tensors.append(transform(Image.fromarray(img)))
    all_names.append(row['tile_name'])

# Batched inference
results   = []
n_batches = math.ceil(len(all_tensors) / BATCH_SIZE)
pbar      = tqdm(total=len(all_tensors), desc='Species inference')

with torch.inference_mode():
    for b in range(n_batches):
        batch         = torch.stack(all_tensors[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]).to(DEVICE)
        features      = backbone(batch)
        probs         = F.softmax(classifier(features), dim=1)
        top5_scores, top5_idx = probs.topk(5, dim=1)
        for j in range(batch.size(0)):
            idx      = b * BATCH_SIZE + j
            labels_j = [class_names[k] for k in top5_idx[j].tolist()]
            scores_j = [round(s * 100, 1) for s in top5_scores[j].cpu().tolist()]
            results.append({
                'tile_name':   all_names[idx],
                'top5_labels': labels_j,
                'top5_scores': scores_j,
                'best_label':  labels_j[0],
                'best_score':  scores_j[0],
            })
        pbar.update(batch.size(0))
pbar.close()

print(f"\nInference complete on {len(results)} crown tiles.")
for r in results[:10]:
    print(f"  {r['tile_name']:40s}  ->  {r['best_label']}  ({r['best_score']}%)")
if len(results) > 10:
    print(f"  ... and {len(results) - 10} more")

# =============================================================================
# STEP 7.5 — Species Distribution Chart
# Colab: "## 7.5 Species Distribution"
# =============================================================================
print("\n" + "=" * 60)
print("STEP 7.5 — Species distribution chart")
print("=" * 60)

species_counts = Counter(r['best_label'] for r in results)
top10          = species_counts.most_common(10)
other_count    = sum(species_counts.values()) - sum(c for _, c in top10)

labels = (['Other'] if other_count > 0 else []) + [s for s, _ in reversed(top10)]
counts = ([other_count] if other_count > 0 else []) + [c for _, c in reversed(top10)]
pcts   = [100 * c / len(results) for c in counts]
colors = (['#999999'] if other_count > 0 else []) + ['#4C72B0'] * len(top10)

fig, ax = plt.subplots(figsize=(10, 5))
y_pos   = range(len(labels))
bars    = ax.barh(y_pos, pcts, color=colors, edgecolor='white', height=0.7)
for bar, pct, cnt in zip(bars, pcts, counts):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f'{pct:.1f}% ({cnt})', va='center', fontsize=9)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel('Proportion of crowns (%)', fontsize=10)
ax.set_title(f'Step 7.5 — Top 10 Predicted Species  (n={len(results)} crowns)', fontsize=12)
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlim(0, max(pcts) * 1.25)
plt.tight_layout()
plt.show()

# =============================================================================
# STEP 7.6 — Per-Crown Predictions
# Colab: "## 7.6 Per-Crown Predictions"
# =============================================================================
print("\n" + "=" * 60)
print("STEP 7.6 — Per-crown prediction visualization")
print("=" * 60)

N_DISPLAY       = min(3, len(results))
display_results = results[:N_DISPLAY]
BAR_COLOR       = '#4C72B0'

fig, axes = plt.subplots(N_DISPLAY, 2, figsize=(14, N_DISPLAY * 3.5))
axes = np.array(axes).reshape(N_DISPLAY, 2)
fig.suptitle('Step 7.6 — CrownView: Top-5 Species Predictions', fontsize=14, fontweight='bold', y=1.01)

for i, r in enumerate(display_results):
    ax_img, ax_bar = axes[i, 0], axes[i, 1]
    img_arr = np.array(Image.open(sample_df.iloc[i]['tile_path_abs']).convert('RGB'))
    img_arr = mask_outside_polygon(img_arr, r['tile_name'])
    ax_img.imshow(img_arr)
    ax_img.axis('off')
    ax_img.set_title(r['tile_name'], fontsize=7, color='gray')
    labels = r['top5_labels']
    scores = r['top5_scores']
    y_pos  = range(len(labels) - 1, -1, -1)
    bars   = ax_bar.barh(list(y_pos), scores[::-1], color=BAR_COLOR, edgecolor='white', height=0.6)
    for bar, score in zip(bars, scores[::-1]):
        ax_bar.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f'{score}%', va='center', ha='left', fontsize=9)
    ax_bar.set_yticks(list(y_pos))
    ax_bar.set_yticklabels(labels[::-1], fontsize=9)
    ax_bar.set_xlabel('Confidence (%)', fontsize=9)
    ax_bar.set_xlim(0, max(scores) * 1.2)
    ax_bar.spines[['top', 'right']].set_visible(False)
    ax_bar.set_title(f"Best: {r['best_label']} ({r['best_score']}%)", fontsize=9, pad=6)

plt.tight_layout()
plt.show()

# =============================================================================
# STEP 8 — Save Results to GeoPackage
# Colab: "# Step 8: Download the Results"
# (In Colab this also triggered a browser download; here we just save to disk)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 8 — Saving enriched polygon file")
print("=" * 60)

import geopandas as gpd

final_gpkg_candidates = [
    f for f in sorted(glob.glob(os.path.join(OUTPUT_DIR, "4_aggregator", "*.gpkg")))
    if 'notaggregated' not in f
]
gdf = gpd.read_file(final_gpkg_candidates[0])
print(f"Loaded {len(gdf)} tree crown polygons from 4_aggregator")

# Add 4 new species prediction columns to the polygon file
gdf["predicted_species"]  = None
gdf["species_confidence"] = None
gdf["top5_species"]       = None
gdf["top5_confidence"]    = None

for idx, row in gdf.iterrows():
    if idx < len(results):
        r = results[idx]
        gdf.at[idx, "predicted_species"]  = r["best_label"]
        gdf.at[idx, "species_confidence"] = r["best_score"]
        gdf.at[idx, "top5_species"]       = " | ".join(r["top5_labels"])
        gdf.at[idx, "top5_confidence"]    = " | ".join(str(s) for s in r["top5_scores"])

output_gpkg = os.path.join(OUTPUT_DIR, "crowns_with_species.gpkg")
gdf.to_file(output_gpkg, driver="GPKG")

print(f"\nSaved to: {output_gpkg}")
print(f"\nSample predictions:")
print(gdf[["predicted_species", "species_confidence"]].head(10).to_string())
print("""
=============================================================
ALL DONE!

Copy these two files to your laptop:
  1. D:\\Tanushree\\p1\\crowns_with_species.gpkg   <- open in QGIS
  2. D:\\Tanushree\\CanopyRS\\assets\\20240130_zf2tower_m3m_rgb_test_crop.tif

In QGIS: drag both files in, then style the .gpkg layer
by "predicted_species" to colour-code each tree crown.
=============================================================
""")
