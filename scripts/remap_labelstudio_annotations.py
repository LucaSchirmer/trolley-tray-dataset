#!/usr/bin/env python3
"""Remap polygon points in a Label Studio export JSON after a rectangular crop.

This script updates polygon `points` found in the export. It supports both
absolute pixel coordinates and normalized coordinates in [0,1] (detected
automatically). For normalized coordinates the script attempts to read the
original image size from the export or from a provided `--image-root`.

Usage:
  python scripts/remap_labelstudio_annotations.py --in export.json --out remapped.json \
    --left 100 --top 50 --width 1200 --height 800 --image-root images/

Note: This script currently transforms polygon points. If your export contains
bitmap masks (encoded PNG/RLE), handle those separately (the script will warn).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from PIL import Image
except Exception:
    Image = None


def load_json(p: Path) -> Any:
    with p.open("r", encoding="utf8") as f:
        return json.load(f)


def save_json(obj: Any, p: Path) -> None:
    with p.open("w", encoding="utf8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def is_normalized(points: List[List[float]]) -> bool:
    # true if all coordinates are between 0 and 1 (inclusive)
    if not points:
        return False
    for x, y in points:
        if x < 0 or y < 0:
            return False
        if x > 1.0 or y > 1.0:
            return False
    return True


def remap_point(x: float, y: float, left: float, top: float, crop_w: float, crop_h: float, scale_x: float = 1.0, scale_y: float = 1.0, normalized: bool = False, orig_w: int | None = None, orig_h: int | None = None) -> Tuple[float, float]:
    # if normalized True, x,y are in [0,1] relative to orig_w/orig_h
    if normalized:
        if orig_w is None or orig_h is None:
            raise ValueError("orig_w and orig_h required for normalized coords")
        ax = x * orig_w
        ay = y * orig_h
    else:
        ax = x
        ay = y

    nx = (ax - left) * scale_x
    ny = (ay - top) * scale_y

    if normalized:
        # return normalized coordinates relative to new cropped size
        return nx / (crop_w * scale_x), ny / (crop_h * scale_y)
    else:
        return nx, ny


def remap_points_list(points: List[List[float]], **kwargs) -> List[List[float]]:
    norm = is_normalized(points)
    out = []
    for x, y in points:
        out.append(list(remap_point(x, y, normalized=norm, **kwargs)))
    return out


def find_image_size(task: Dict[str, Any], image_root: Path | None) -> Tuple[int, int] | None:
    # Try common Label Studio fields
    if isinstance(task, dict):
        data = task.get("data") or task.get("task") or task
        if isinstance(data, dict):
            for key in ("image", "img", "image_url", "unconsumed", "consumed"):
                val = data.get(key)
                if isinstance(val, str):
                    # try to find local image by basename if image_root provided
                    if image_root:
                        candidate = image_root / Path(val).name
                        if candidate.exists() and Image:
                            with Image.open(candidate) as im:
                                return im.size
    # fallback: try original_width/height in export result
    if isinstance(task, dict):
        ow = task.get("original_width")
        oh = task.get("original_height")
        if ow and oh:
            return int(ow), int(oh)
    return None


def process_export(obj: Any, left: int, top: int, crop_w: int, crop_h: int, image_root: Path | None = None, scale_x: float = 1.0, scale_y: float = 1.0) -> Any:
    # obj is typically a list of task dicts (Label Studio export)
    if not isinstance(obj, list):
        raise ValueError("expected export JSON to be a list of task dicts")

    for task in obj:
        orig_size = find_image_size(task, image_root)
        # results may be under task['annotations'][0]['result'] or task['result']
        results = None
        if isinstance(task, dict):
            if "annotations" in task and task["annotations"]:
                ann = task["annotations"][0]
                results = ann.get("result")
            results = results or task.get("result") or task.get("results")

        if not results:
            continue

        for r in results:
            # polygon case: value.points
            val = r.get("value") if isinstance(r, dict) else None
            if not isinstance(val, dict):
                continue

            if "points" in val and isinstance(val["points"], list):
                try:
                    kwargs = {"left": left, "top": top, "crop_w": crop_w, "crop_h": crop_h, "scale_x": scale_x, "scale_y": scale_y}
                    if is_normalized(val["points"]):
                        if orig_size:
                            kwargs["orig_w"], kwargs["orig_h"] = orig_size
                        else:
                            raise RuntimeError("Normalized coords detected but original image size unknown; provide --image-root or include original size in export.")
                    val["points"] = remap_points_list(val["points"], **kwargs)
                except Exception as e:
                    print(f"Warning: couldn't remap polygon in task {task.get('id')}: {e}")

            # mask handling not implemented fully
            if "mask" in val or ("rle" in val) or ("bitmap" in val):
                print("Warning: export contains bitmap masks; this script does not auto-crop masks. Handle mask files separately.")

    return obj


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Remap polygon coords in Label Studio export after crop")
    p.add_argument("--in", dest="in_json", required=True, help="Label Studio export JSON")
    p.add_argument("--out", dest="out_json", required=True, help="Output JSON path")
    p.add_argument("--left", type=int, required=True)
    p.add_argument("--top", type=int, required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--image-root", help="Local folder where source images live (optional)")
    p.add_argument("--scale-x", type=float, default=1.0, help="Optional scale factor applied after crop on X")
    p.add_argument("--scale-y", type=float, default=1.0, help="Optional scale factor applied after crop on Y")
    return p


def main():
    p = build_parser()
    args = p.parse_args()
    inp = Path(args.in_json)
    out = Path(args.out_json)
    image_root = Path(args.image_root) if args.image_root else None
    obj = load_json(inp)
    new = process_export(obj, args.left, args.top, args.width, args.height, image_root=image_root, scale_x=args.scale_x, scale_y=args.scale_y)
    save_json(new, out)
    print(f"Saved remapped JSON to {out}")


if __name__ == "__main__":
    main()
