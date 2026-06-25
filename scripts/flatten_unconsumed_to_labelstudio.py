#!/usr/bin/env python3
"""Flatten an unconsumed pairs JSON into a Label Studio import JSON.

This script reads files like annotations/pairs_unconsumed_wrap.json where each
entry contains an unconsumed image and metadata, then converts the image path
into an absolute URL for Label Studio.

Output tasks contain:
- unconsumed (absolute URL)
- possibleElements (array)
- source_dir (if present)
- pair_id / id (if present)

Usage:
  python scripts/flatten_unconsumed_to_labelstudio.py \
      --in annotations/pairs_unconsumed_wrap.json \
      --out annotations/pairs_unconsumed_wrap_labelstudio.json \
      --base-url http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def flatten(infile: Path, outfile: Path, base_url: str) -> int:
    """Convert unconsumed-only entries to Label Studio tasks."""
    data = json.loads(infile.read_text(encoding="utf-8"))
    tasks: list[dict[str, object]] = []

    for entry in data:
        unconsumed = entry.get("unconsumed", "")
        if not unconsumed:
            continue

        if isinstance(unconsumed, str) and unconsumed.startswith("http"):
            unconsumed_field = unconsumed
        else:
            unconsumed_field = f"{base_url}/{unconsumed}"

        task: dict[str, object] = {
            "unconsumed": unconsumed_field,
            "possibleElements": entry.get("possibleElements", []),
        }
        if "source_dir" in entry:
            task["source_dir"] = entry["source_dir"]
        if "pair_id" in entry:
            task["pair_id"] = entry["pair_id"]
        elif "id" in entry:
            task["pair_id"] = entry["id"]

        tasks.append(task)

    outfile.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(tasks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", default="annotations/pairs_unconsumed_wrap.json")
    parser.add_argument(
        "--out",
        dest="outfile",
        default="annotations/pairs_unconsumed_wrap_labelstudio.json",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL for image paths (e.g. http://localhost:8000)",
    )
    args = parser.parse_args()

    infile = Path(args.infile)
    outfile = Path(args.outfile)
    if not infile.exists():
        raise SystemExit(f"Input file not found: {infile}")

    n = flatten(infile, outfile, args.base_url)
    print(f"Wrote {n} tasks to {outfile}")


if __name__ == "__main__":
    main()
