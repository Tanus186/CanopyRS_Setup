"""
Tree Crown Detection Accuracy Calculator
=========================================
Compares CanopyRS output polygons against manual/ground truth polygons.

Metrics produced:
- Precision: Of all predicted crowns, how many match a real crown?
- Recall: Of all real crowns, how many were detected?
- F1 Score: Balance between precision and recall (main accuracy number).
- Mean IoU: Average overlap quality of matched crowns.
- Count comparison: predicted vs ground truth totals.

Usage:
    python accuracy_score.py --predicted path/to/canopyrs_output.gpkg --ground_truth path/to/manual.gpkg

Optional:
    --iou_threshold 0.5    (default: 0.5 — minimum overlap to count as a match)
    --output path/to/results.csv  (save results to CSV)
"""

import argparse
import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore")

try:
    import geopandas as gpd
except ImportError:
    print("ERROR: geopandas not installed. Run: pip install geopandas")
    sys.exit(1)

try:
    from shapely.validation import make_valid
except ImportError:
    make_valid = None


def load_and_validate(path, name):
    """Load a .gpkg file and fix invalid geometries."""
    print(f"Loading {name}: {path}")
    gdf = gpd.read_file(path)
    print(f"  Found {len(gdf)} polygons")

    # Keep only polygon geometries
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    print(f"  {len(gdf)} polygon geometries after filtering")

    # Fix invalid geometries
    if make_valid:
        gdf["geometry"] = gdf.geometry.apply(
            lambda g: make_valid(g) if not g.is_valid else g
        )

    # Remove empty geometries
    gdf = gdf[~gdf.geometry.is_empty].copy()

    return gdf


def compute_iou_matrix(pred_gdf, gt_gdf):
    """Compute IoU between all predicted and ground truth polygons using spatial index."""
    print("Computing IoU between predicted and ground truth polygons...")

    # Build spatial index on ground truth
    gt_sindex = gt_gdf.sindex

    # For each predicted polygon, find candidate ground truth overlaps
    iou_pairs = []  # (pred_idx, gt_idx, iou_value)

    for pred_idx, pred_row in pred_gdf.iterrows():
        pred_geom = pred_row.geometry

        # Find candidate GT polygons using spatial index (bounding box check)
        candidate_idxs = list(gt_sindex.intersection(pred_geom.bounds))

        for gt_pos in candidate_idxs:
            gt_geom = gt_gdf.iloc[gt_pos].geometry

            try:
                intersection = pred_geom.intersection(gt_geom).area
                if intersection > 0:
                    union = pred_geom.union(gt_geom).area
                    iou = intersection / union if union > 0 else 0
                    iou_pairs.append((pred_idx, gt_pos, iou))
            except Exception:
                continue

    return iou_pairs


def match_polygons(iou_pairs, iou_threshold):
    """Greedy matching: assign each predicted polygon to at most one GT polygon (and vice versa)."""
    # Sort by IoU descending
    iou_pairs.sort(key=lambda x: x[2], reverse=True)

    matched_pred = set()
    matched_gt = set()
    matches = []

    for pred_idx, gt_pos, iou in iou_pairs:
        if iou < iou_threshold:
            continue
        if pred_idx in matched_pred or gt_pos in matched_gt:
            continue
        matches.append((pred_idx, gt_pos, iou))
        matched_pred.add(pred_idx)
        matched_gt.add(gt_pos)

    return matches


def calculate_metrics(pred_gdf, gt_gdf, iou_threshold=0.5):
    """Calculate precision, recall, F1, and mean IoU."""

    n_predicted = len(pred_gdf)
    n_ground_truth = len(gt_gdf)

    print(f"\nPredicted polygons: {n_predicted}")
    print(f"Ground truth polygons: {n_ground_truth}")
    print(f"IoU threshold: {iou_threshold}")
    print()

    # Compute IoU pairs
    iou_pairs = compute_iou_matrix(pred_gdf, gt_gdf)

    # Match polygons
    matches = match_polygons(iou_pairs, iou_threshold)

    true_positives = len(matches)
    false_positives = n_predicted - true_positives
    false_negatives = n_ground_truth - true_positives

    # Metrics
    precision = true_positives / n_predicted if n_predicted > 0 else 0
    recall = true_positives / n_ground_truth if n_ground_truth > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    mean_iou = np.mean([m[2] for m in matches]) if matches else 0

    # Area-based metrics
    pred_total_area = pred_gdf.geometry.area.sum()
    gt_total_area = gt_gdf.geometry.area.sum()

    results = {
        "iou_threshold": iou_threshold,
        "n_predicted": n_predicted,
        "n_ground_truth": n_ground_truth,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mean_iou": mean_iou,
        "predicted_total_area": pred_total_area,
        "ground_truth_total_area": gt_total_area,
    }

    return results


def print_results(results):
    """Print results in a clear format."""
    print("=" * 60)
    print("       TREE CROWN DETECTION ACCURACY REPORT")
    print("=" * 60)
    print()
    print(f"  IoU Threshold:          {results['iou_threshold']}")
    print()
    print("  COUNTS")
    print(f"    Predicted crowns:     {results['n_predicted']}")
    print(f"    Ground truth crowns:  {results['n_ground_truth']}")
    print(f"    True positives:       {results['true_positives']}")
    print(f"    False positives:      {results['false_positives']}  (predicted but no match)")
    print(f"    False negatives:      {results['false_negatives']}  (missed detections)")
    print()
    print("  ACCURACY METRICS")
    print(f"    Precision:            {results['precision']:.4f}  ({results['precision']*100:.1f}%)")
    print(f"    Recall:               {results['recall']:.4f}  ({results['recall']*100:.1f}%)")
    print(f"    F1 Score:             {results['f1_score']:.4f}  ({results['f1_score']*100:.1f}%)")
    print(f"    Mean IoU (matched):   {results['mean_iou']:.4f}  ({results['mean_iou']*100:.1f}%)")
    print()
    print("  AREA")
    print(f"    Predicted total:      {results['predicted_total_area']:.2f} sq units")
    print(f"    Ground truth total:   {results['ground_truth_total_area']:.2f} sq units")
    print(f"    Area ratio:           {results['predicted_total_area']/results['ground_truth_total_area']:.2f}" if results['ground_truth_total_area'] > 0 else "")
    print()
    print("=" * 60)
    print()
    print("  WHAT THESE NUMBERS MEAN:")
    print(f"    - Precision {results['precision']*100:.0f}%: Of all crowns CanopyRS found,")
    print(f"      {results['precision']*100:.0f}% actually match a real crown.")
    print(f"    - Recall {results['recall']*100:.0f}%: Of all real crowns in your ground truth,")
    print(f"      {results['recall']*100:.0f}% were detected by CanopyRS.")
    print(f"    - F1 {results['f1_score']*100:.0f}%: Overall accuracy balancing both.")
    print(f"    - Mean IoU {results['mean_iou']*100:.0f}%: How well the matched crown shapes overlap.")
    print()
    print("  QUICK INTERPRETATION:")
    if results['f1_score'] >= 0.8:
        print("    F1 >= 80%: Excellent detection quality.")
    elif results['f1_score'] >= 0.6:
        print("    F1 60-80%: Good but room for improvement.")
    elif results['f1_score'] >= 0.4:
        print("    F1 40-60%: Moderate — refinement recommended.")
    else:
        print("    F1 < 40%: Low — significant refinement needed.")

    if results['precision'] > results['recall'] + 0.15:
        print("    Precision >> Recall: CanopyRS is missing many trees (under-detection).")
        print("    -> Try lowering score_threshold in the aggregator config.")
    elif results['recall'] > results['precision'] + 0.15:
        print("    Recall >> Precision: CanopyRS is finding too many false crowns (over-detection).")
        print("    -> Try raising score_threshold in the aggregator config.")

    print()


def save_results(results, output_path):
    """Save results to CSV."""
    import csv
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for key, value in results.items():
            writer.writerow([key, value])
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare CanopyRS output against ground truth tree crown polygons."
    )
    parser.add_argument(
        "--predicted", "-p", required=True,
        help="Path to CanopyRS output .gpkg file"
    )
    parser.add_argument(
        "--ground_truth", "-g", required=True,
        help="Path to manual/ground truth .gpkg file"
    )
    parser.add_argument(
        "--iou_threshold", "-t", type=float, default=0.5,
        help="Minimum IoU to count as a match (default: 0.5)"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Save results to CSV file (optional)"
    )

    args = parser.parse_args()

    # Load data
    pred_gdf = load_and_validate(args.predicted, "CanopyRS output")
    gt_gdf = load_and_validate(args.ground_truth, "Ground truth")

    # Check CRS match
    if pred_gdf.crs != gt_gdf.crs:
        print(f"\nWARNING: CRS mismatch!")
        print(f"  Predicted: {pred_gdf.crs}")
        print(f"  Ground truth: {gt_gdf.crs}")
        print(f"  Reprojecting ground truth to match predicted...")
        gt_gdf = gt_gdf.to_crs(pred_gdf.crs)

    # Calculate metrics
    results = calculate_metrics(pred_gdf, gt_gdf, args.iou_threshold)

    # Print results
    print_results(results)

    # Optionally save
    if args.output:
        save_results(results, args.output)

    # Also run at multiple IoU thresholds for a fuller picture
    print("\n  ACCURACY AT MULTIPLE IoU THRESHOLDS:")
    print("  " + "-" * 50)
    print(f"  {'IoU Threshold':<15} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("  " + "-" * 50)
    # Reuse the IoU pairs already computed (expensive step)
    iou_pairs = compute_iou_matrix(pred_gdf, gt_gdf)
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        matches = match_polygons(iou_pairs, t)
        tp = len(matches)
        n_p, n_g = len(pred_gdf), len(gt_gdf)
        prec = tp / n_p if n_p > 0 else 0
        rec = tp / n_g if n_g > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        r = {"precision": prec, "recall": rec, "f1_score": f1}
        print(f"  {t:<15.1f} {r['precision']:<12.4f} {r['recall']:<12.4f} {r['f1_score']:<12.4f}")
    print("  " + "-" * 50)
    print()



if __name__ == "__main__":
    main()
