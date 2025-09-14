#!/usr/bin/env python3
"""
Evaluate monocular depth against Hamlyn ground-truth depth maps.

Hamlyn Rectified Dataset specifics:
- GT depth maps are uint16 .png in *millimeters*.
- Valid evaluation range: [1, 300] mm (others ignored).
- Typical structure: dataset_folder/rectifiedXX/{color,depth,intrinsics.txt}

Usage (single pair):
  python eval_hamlyn_depth.py --gt path/to/rectified01/depth/000123.png \
                              --pred path/to/your/000123.npy

Usage (folders with matching basenames):
  python eval_hamlyn_depth.py --gt_dir path/to/rectified01/depth \
                              --pred_dir path/to/your_predictions \
                              --ext_pred .npy

Options for alignment (monocular is scale ambiguous):
  --align none|scale|scale_shift|median   (default: scale)

Author: you + ChatGPT
"""
import argparse
import math
import os
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import cv2

def align_depth(
    pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, mode: str = "scale"
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Align 'pred' to 'gt' on valid mask.

    mode:
      - "none": no change
      - "scale": pred' = s * pred
      - "scale_shift": pred' = s * pred + t  (least squares)
      - "median": scale so medians match (robust scale only)

    Returns aligned prediction and dict with parameters.
    """
    p = pred[mask].astype(np.float64)
    g = gt[mask].astype(np.float64)
    if p.size == 0:
        raise ValueError("Empty mask after filtering. Check inputs.")

    if mode == "none":
        return pred, {"s": 1.0, "t": 0.0}

    if mode == "median":
        s = np.median(g) / max(np.median(p), 1e-8)
        return pred * s, {"s": float(s), "t": 0.0}

    if mode == "scale":
        # Closed-form least-squares scale (no shift): s = (p·g)/(p·p)
        denom = np.dot(p, p) + 1e-12
        s = float(np.dot(p, g) / denom)
        return pred * s, {"s": s, "t": 0.0}

    if mode == "scale_shift":
        # Solve [p 1] [s; t] = g  in least squares
        A = np.vstack([p, np.ones_like(p)]).T
        sol, _, _, _ = np.linalg.lstsq(A, g, rcond=None)
        s, t = sol.astype(np.float64)
        return pred * s + t, {"s": float(s), "t": float(t)}

    raise ValueError(f"Unknown alignment mode: {mode}")

def robust_mask_hamlyn(gt_mm: np.ndarray) -> np.ndarray:
    # keep only valid ground-truth pixels
    return (gt_mm >= 1.0) & (gt_mm <= 300.0) & np.isfinite(gt_mm)
# ------------------------------ metrics ---------------------------------- #

def depth_metrics(gt: np.ndarray, pr: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Compute standard depth metrics on masked pixels (expects metric mm)."""
    g = gt[mask].astype(np.float64)
    p = pr[mask].astype(np.float64)

    # avoid zeros/negatives for log-based metrics
    eps = 1e-6
    g_pos = np.clip(g, eps, None)
    p_pos = np.clip(p, eps, None)

    diff = p - g
    abs_diff = np.abs(diff)
    sq_diff = diff ** 2

    mae = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(sq_diff)))
    abs_rel = float(np.mean(abs_diff / g_pos))
    sq_rel = float(np.mean(sq_diff / g_pos))
    log10 = float(np.mean(np.abs(np.log10(p_pos) - np.log10(g_pos))))
    dlog = np.log(p_pos) - np.log(g_pos)
    silog = float(np.sqrt(np.mean(dlog ** 2) - (np.mean(dlog) ** 2)))

    ratio = np.maximum(p_pos / g_pos, g_pos / p_pos)
    d1 = float(np.mean(ratio < 1.25))
    d2 = float(np.mean(ratio < 1.25 ** 2))
    d3 = float(np.mean(ratio < 1.25 ** 3))

    return {
        # "MAE_mm": mae,
        "AbsRel": abs_rel,
        "SqRel": sq_rel,
        "RMSE_mm": rmse,
        # "Log10": log10,
        # "SIlog": silog,
        "δ<1.25": d1,
        # "δ<1.25²": d2,
        # "δ<1.25³": d3,
        # "N": int(g.size),
    }

def inverse(depth):
    depth = cv2.cvtColor(depth, cv2.COLOR_RGB2GRAY)
    print(depth)
    depth = depth/255.0
    print(depth)
    depth = 1-depth
    depth = depth* 255
    depth = cv2.cvtColor(depth.astype(np.uint16), cv2.COLOR_GRAY2RGB)
    return depth.astype(np.uint16)

def resize(img):
    target_size = (640, 480)
    return cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

def main():
    path_gt = "/mnt/d/projectsD/datasets/depth_val_test/colored_depth.png"
    path_DA = "/mnt/d/projectsD/datasets/depth_val_test/DepthAnything/0000001236_DA_depth.png"
    path_DA_metric = "/mnt/d/projectsD/datasets/depth_val_test/DepthAnything/0000001236_DAmetric_depth.png"
    path_DepthPro = "/mnt/d/projectsD/datasets/depth_val_test/DepthPro/0000001236.jpg"
    path_DVS = "/mnt/d/projectsD/datasets/depth_val_test/DVSMono/0000001236.jpeg"
    gt = cv2.imread(path_gt, cv2.IMREAD_UNCHANGED)
    pred_DA = cv2.imread(path_DA, cv2.IMREAD_UNCHANGED)
    pred_DA_metric = cv2.imread(path_DA_metric, cv2.IMREAD_UNCHANGED)
    pred_DepthPro = cv2.imread(path_DepthPro, cv2.IMREAD_UNCHANGED)
    # pred_DepthPro = np.load(path_DepthPro)["depth"].astype(np.float32)
    # print(pred_DepthPro)
    pred_DVS = cv2.imread(path_DVS, cv2.IMREAD_UNCHANGED)
    pred_DVS = resize(pred_DVS)
    # print(pred)
    mask = robust_mask_hamlyn(gt)
    aligned_pred_DA, _ = align_depth(pred_DA, gt, mask, mode="scale")
    aligned_pred_DAmetric, _ = align_depth(pred_DA_metric, gt, mask, mode="scale")
    aligned_pred_DepthPro, _ = align_depth(pred_DepthPro, gt, mask, mode="scale")
    aligned_pred_DVS, _ = align_depth(pred_DVS, gt, mask, mode="scale")
    metrics_DA = depth_metrics(gt, aligned_pred_DA, mask)
    metrics_DA_metric = depth_metrics(gt, aligned_pred_DAmetric, mask)
    metrics_DepthPro = depth_metrics(gt, aligned_pred_DepthPro, mask)
    metrics_DVS = depth_metrics(gt, aligned_pred_DVS, mask)
    print("DepthAnything (rel): ",metrics_DA)
    print("DepthAnything (metric): ",metrics_DA_metric)
    print("DepthPro (metric): ",metrics_DepthPro)
    print("DVS (metric): ",metrics_DVS)

if __name__ == "__main__":
    main()
