#!/usr/bin/env python3
"""
Robust ArUco Tray Cropper with Angle-Based Sorting and Adjustable Padding.
"""

import argparse
import os
from pathlib import Path
import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png"}

def iter_images(src_dir: Path):
    for root, _, files in os.walk(src_dir):
        for f in files:
            if Path(f).suffix.lower() in IMG_EXTS:
                yield Path(root) / f

def sort_points_clockwise(pts: np.ndarray) -> np.ndarray:
    """
    Sorts 4 points in a highly robust clockwise order starting from top-left:
    [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
    Uses the polar angle relative to the centroid.
    """
    # Calculate centroid
    cx, cy = pts.mean(axis=0)
    
    # Calculate angles of points relative to centroid
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    
    # Sort points by angle clockwise
    # np.arctan2 returns values from -pi to pi. Sorting them gives a clean clockwise order.
    # To ensure we start at Top-Left (which is roughly at an angle of -3pi/4 or 135 deg):
    sorted_pts = pts[np.argsort(angles)]
    
    # Shift array so that the top-left point (smallest x + smallest y) comes first
    sums = sorted_pts.sum(axis=1)
    tl_idx = np.argmin(sums)
    sorted_pts = np.roll(sorted_pts, -tl_idx, axis=0)
    
    return sorted_pts

def process_aruco_crop(img_path: Path, out_path: Path, target_w: int, target_h: int, padding_pct: float):
    img = cv2.imread(str(img_path))
    if img is None:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ArUco Detection
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    corners, ids, _ = detector.detectMarkers(gray)
    
    if ids is None or len(ids) < 3:
        print(f"Error: Too few markers found ({0 if ids is None else len(ids)}) in {img_path.name}. Skipping.")
        return False

    # Map centers
    marker_centers = [c[0].mean(axis=0) for c in corners]
    detected_pts = np.array(marker_centers, dtype="float32")
    
    if len(ids) >= 4:
        # If we have 4 or more, take the 4 most prominent ones and sort them robustly
        src_pts = sort_points_clockwise(detected_pts[:4])
    else:
        print(f"Warning: Fallback active. Only 3 markers found in {img_path.name}. Interpolating...")
        # Fallback vector math
        # Simple geometric inference based on bounding box centers
        cx, cy = detected_pts.mean(axis=0)
        # Find which corner is missing by checking quadrants relative to center
        quadrants = []
        for pt in detected_pts:
            quadrants.append((pt[0] > cx, pt[1] > cy))
            
        # Possible quadrants: (False, False)=TL, (True, False)=TR, (True, True)=BR, (False, True)=BL
        all_quads = {(False, False), (True, False), (True, True), (False, True)}
        missing_quad = all_quads - set(quadrants)
        
        # Approximate the missing point using standard parallelogram vector addition
        # TL + BR = TR + BL
        # We find the indices of the present points
        pts_dict = {q: p for q, p in zip(quadrants, detected_pts)}
        
        tl = pts_dict.get((False, False))
        tr = pts_dict.get((True, False))
        br = pts_dict.get((True, True))
        bl = pts_dict.get((False, True))
        
        if (False, False) in missing_quad: # TL missing
            tl = tr + bl - br
        elif (True, False) in missing_quad: # TR missing
            tr = tl + br - bl
        elif (True, True) in missing_quad: # BR missing
            br = tr + bl - tl
        elif (False, True) in missing_quad: # BL missing
            bl = tl + br - tr
            
        src_pts = np.array([tl, tr, br, bl], dtype="float32")
        src_pts = sort_points_clockwise(src_pts)

    # TRUE PADDING CALCULATION
    pad_w = int(target_w * padding_pct)
    pad_h = int(target_h * padding_pct)
    
    full_w = target_w + (2 * pad_w)
    full_h = target_h + (2 * pad_h)
    
    dst_pts = np.array([
        [pad_w, pad_h],                        # TL
        [target_w + pad_w, pad_h],             # TR
        [target_w + pad_w, target_h + pad_h],  # BR
        [pad_w, target_h + pad_h]              # BL
    ], dtype="float32")
    
    # Execute Perspective Warp safely
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, matrix, (full_w, full_h))
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), warped)
    return True

def main():
    p = argparse.ArgumentParser(description="Robust ArUco Tray Cropper")
    p.add_argument("src_dir")
    p.add_argument("out_dir")
    p.add_argument("--width", type=int, default=600)
    p.add_argument("--height", type=int, default=800)
    p.add_argument("--padding", type=float, default=0.25, help="Padding ratio (e.g., 0.25 = 25% extra space on all sides)")
    
    args = p.parse_args()
    src = Path(args.src_dir)
    out = Path(args.out_dir)
    
    if not src.exists():
        raise SystemExit(f"Source directory does not exist: {src}")
        
    success_count = 0
    total_count = 0
    
    for p_img in iter_images(src):
        total_count += 1
        rel = p_img.relative_to(src)
        out_p = out / rel
        
        if process_aruco_crop(p_img, out_p, args.width, args.height, args.padding):
            success_count += 1
            
    print(f"Done! Successfully processed {success_count}/{total_count} images.")

if __name__ == "__main__":
    main()