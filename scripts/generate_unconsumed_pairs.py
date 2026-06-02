#!/usr/bin/env python3
"""Generate pairs-style JSON entries for unconsumed images.

This scans a directory of unconsumed images and writes one JSON object per
image with the following shape:

    {
    "source_dir": "images/unconsumed/salad",
      "unconsumed": "images/unconsumed/.../image.jpg",
      "consumed": [],
      "possibleElements": []
    }

The output is useful as a starting point before the consumed-side pairs are
filled in.

If you point the script at the parent directory (for example
`images/unconsumed`), it can also create one JSON file per populated
subdirectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def build_entries(
    input_dir: Path, root_dir: Path, source_dir: str
) -> list[dict[str, object]]:
    """Create one empty pair entry per image found under input_dir."""
    entries: list[dict[str, object]] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        relative_path = path.relative_to(root_dir).as_posix()
        entries.append(
            {
                "source_dir": source_dir,
                "unconsumed": relative_path,
                "consumed": [],
                "possibleElements": [],
            }
        )

    return entries


def resolve_source_dir(input_dir: Path, root_dir: Path) -> str:
    """Return the directory path that should be stored in the JSON."""
    try:
        return input_dir.relative_to(root_dir).as_posix()
    except ValueError:
        return input_dir.as_posix()


def iter_image_dirs(input_dir: Path) -> list[Path]:
    """Return directories under input_dir that contain image files.

    The directory itself is included if it contains images. Otherwise, every
    descendant directory that contains at least one image anywhere below it is
    returned once.
    """
    targets: list[Path] = []
    for directory in sorted(
        (path for path in input_dir.rglob("*") if path.is_dir()),
        key=lambda path: path.as_posix(),
    ):
        if any(
            path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            for path in directory.rglob("*")
        ):
            targets.append(directory)

    if any(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        for path in input_dir.rglob("*")
    ):
        targets.insert(0, input_dir)

    unique_targets: list[Path] = []
    seen: set[Path] = set()
    for directory in targets:
        if directory in seen:
            continue
        seen.add(directory)
        unique_targets.append(directory)

    return unique_targets


def output_suffix_for_target(input_dir: Path, target_dir: Path) -> str:
    """Build a safe filename suffix for a batch output target."""
    try:
        relative_name = target_dir.relative_to(input_dir).as_posix()
    except ValueError:
        relative_name = target_dir.name

    relative_name = relative_name.strip().replace("\\", "/")
    if relative_name in {"", "."}:
        relative_name = target_dir.name

    relative_name = relative_name.replace("/", "_")
    if not relative_name or relative_name == "_":
        relative_name = target_dir.name

    return relative_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="images/unconsumed",
        help="Directory to scan for unconsumed images.",
    )
    parser.add_argument(
        "--root-dir",
        default=".",
        help="Root directory used to store paths in the output JSON.",
    )
    parser.add_argument(
        "--out",
        default="annotations/pairs_unconsumed.json",
        help="Output JSON file, or a prefix when --batch is used.",
    )
    parser.add_argument(
        "--dir-name",
        default=None,
        help="Optional name to store in the JSON for the scanned directory.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Write one JSON file per populated subdirectory under --input-dir.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    root_dir = Path(args.root_dir)
    output_file = Path(args.out)

    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")
    if not root_dir.exists():
        raise SystemExit(f"Root directory not found: {root_dir}")

    if args.batch:
        targets = iter_image_dirs(input_dir)
        if not targets:
            print(f"No image files found under {input_dir}")
            return

        for target_dir in targets:
            source_dir = args.dir_name or resolve_source_dir(target_dir, root_dir)
            entries = build_entries(target_dir, root_dir, source_dir)
            if not entries:
                continue

            relative_name = output_suffix_for_target(input_dir, target_dir)
            if output_file.suffix.lower() == ".json":
                output_path = output_file.with_name(
                    f"{output_file.stem}_{relative_name}{output_file.suffix}"
                )
            else:
                output_path = Path(f"{output_file}_{relative_name}.json")

            output_path.write_text(
                json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"Wrote {len(entries)} entries to {output_path}")
        return

    source_dir = args.dir_name or resolve_source_dir(input_dir, root_dir)
    entries = build_entries(input_dir, root_dir, source_dir)
    output_file.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(entries)} entries to {output_file}")


if __name__ == "__main__":
    main()