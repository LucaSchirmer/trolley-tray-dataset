#!/usr/bin/env python3
"""Batch crop images in a directory.

Usage examples:
  # fixed crop for all images
  python scripts/batch_crop.py fixed src_dir out_dir --left 100 --top 50 --width 1200 --height 800

  # center crop to 1024x768
  python scripts/batch_crop.py center src_dir out_dir --crop-width 1024 --crop-height 768 --resize 1024

Supported formats: .jpg .jpeg .png
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

try:
    from PIL import Image
except Exception as e:  # pragma: no cover - helpful error message
    raise ImportError("Pillow is required. Install with: pip install pillow") from e


IMG_EXTS = {".jpg", ".jpeg", ".png"}


def iter_images(src_dir: Path):
    for root, _, files in os.walk(src_dir):
        for f in files:
            if Path(f).suffix.lower() in IMG_EXTS:
                yield Path(root) / f


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def crop_fixed(img: Image.Image, left: int, top: int, w: int, h: int) -> Image.Image:
    return img.crop((left, top, left + w, top + h))


def crop_center(img: Image.Image, crop_w: int, crop_h: int) -> Image.Image:
    iw, ih = img.size
    left = max(0, (iw - crop_w) // 2)
    top = max(0, (ih - crop_h) // 2)
    return img.crop((left, top, left + crop_w, top + crop_h))


def process(src: Path, out: Path, mode: str, args) -> int:
    count = 0
    for p in iter_images(src):
        rel = p.relative_to(src)
        outp = out / rel
        ensure_dir(outp)
        with Image.open(p) as im:
            if mode == "fixed":
                cropped = crop_fixed(im, args.left, args.top, args.width, args.height)
            else:
                cropped = crop_center(im, args.crop_width, args.crop_height)

            if args.resize:
                # resize by max side while keeping aspect ratio
                max_side = int(args.resize)
                iw, ih = cropped.size
                if max(iw, ih) > max_side:
                    if iw >= ih:
                        new_w = max_side
                        new_h = int(round(max_side * ih / iw))
                    else:
                        new_h = max_side
                        new_w = int(round(max_side * iw / ih))
                    cropped = cropped.resize((new_w, new_h), Image.LANCZOS)

            cropped.save(outp)
            count += 1
    return count


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch crop images")
    sub = p.add_subparsers(dest="mode", required=True)

    p_fixed = sub.add_parser("fixed", help="Fixed rectangle crop for all images")
    p_fixed.add_argument("src_dir")
    p_fixed.add_argument("out_dir")
    p_fixed.add_argument("--left", type=int, required=True)
    p_fixed.add_argument("--top", type=int, required=True)
    p_fixed.add_argument("--width", type=int, required=True)
    p_fixed.add_argument("--height", type=int, required=True)
    p_fixed.add_argument("--resize", type=int, help="Resize long side to this value (keeps aspect)")

    p_center = sub.add_parser("center", help="Center crop to width/height")
    p_center.add_argument("src_dir")
    p_center.add_argument("out_dir")
    p_center.add_argument("--crop-width", type=int, required=True)
    p_center.add_argument("--crop-height", type=int, required=True)
    p_center.add_argument("--resize", type=int, help="Resize long side to this value (keeps aspect)")

    return p


def main():
    p = build_parser()
    args = p.parse_args()
    src = Path(args.src_dir)
    out = Path(args.out_dir)
    if not src.exists():
        raise SystemExit(f"src_dir does not exist: {src}")
    out.mkdir(parents=True, exist_ok=True)

    n = process(src, out, args.mode, args)
    print(f"Processed {n} images")


if __name__ == "__main__":
    main()
