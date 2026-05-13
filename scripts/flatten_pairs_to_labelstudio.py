#!/usr/bin/env python3
"""Flatten annotations/pairs.json into a Label Studio import JSON.

Each output task corresponds to one consumed image (so Label Studio's
`Image value="$consumed"` gets a single string). Fields produced:
- `unconsumed` (absolute URL or "")
- `consumed` (absolute URL)
- `possibleElements` (array)
- `pair_id` (if present)

Usage:
  python scripts/flatten_pairs_to_labelstudio.py \
      --in annotations/pairs.json \
      --out annotations/pairs_labelstudio.json \
      --base-url http://localhost:8000
"""
import json
import argparse
from pathlib import Path


def flatten(infile: Path, outfile: Path, base_url: str) -> int:
    """Flatten pairs.json to Label Studio tasks with absolute URLs."""
    data = json.loads(infile.read_text(encoding="utf-8"))
    tasks = []
    for entry in data:
        unconsumed = entry.get("unconsumed", "")
        # Convert to absolute URL or empty string
        if unconsumed == "invalid" or not unconsumed:
            unconsumed_field = ""
        else:
            unconsumed_field = f"{base_url}/{unconsumed}" if not unconsumed.startswith("http") else unconsumed
        
        possible = entry.get("possibleElements", [])
        consumed_list = entry.get("consumed", []) or []
        for c in consumed_list:
            if not c or (isinstance(c, str) and c.strip() == ""):
                continue
            # Convert to absolute URL
            consumed_field = f"{base_url}/{c}" if not c.startswith("http") else c
            task = {
                "unconsumed": unconsumed_field,
                "consumed": consumed_field,
                "possibleElements": possible,
            }
            if "id" in entry:
                task["pair_id"] = entry["id"]
            tasks.append(task)
    outfile.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(tasks)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", default="annotations/pairs.json")
    p.add_argument("--out", dest="outfile", default="annotations/pairs_labelstudio.json")
    p.add_argument("--base-url", default="http://localhost:8000",
                   help="Base URL for image paths (e.g., http://localhost:8000)")
    args = p.parse_args()
    infile = Path(args.infile)
    outfile = Path(args.outfile)
    if not infile.exists():
        raise SystemExit(f"Input file not found: {infile}")
    n = flatten(infile, outfile, args.base_url)
    print(f"Wrote {n} tasks to {outfile}")


if __name__ == "__main__":
    main()
