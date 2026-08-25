#!/usr/bin/env python3
"""Compare generated YOLO segmentation masks with hand-annotated reference masks."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


CORE_CLASSES = {"fish_salmon", "rice", "broccoli", "carrots"}


def read_names(path: Path) -> dict[int, str]:
    return {index: name.strip() for index, name in enumerate(path.read_text(encoding="utf-8").splitlines()) if name.strip()}


def label_masks(path: Path, names: dict[int, str], size: int) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    if not path.is_file():
        return masks
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 7 or len(fields) % 2 == 0:
            raise ValueError(f"{path}:{line_number}: invalid YOLO polygon")
        class_id = int(fields[0])
        if class_id not in names:
            raise ValueError(f"{path}:{line_number}: unknown class id {class_id}")
        values = np.asarray([float(value) for value in fields[1:]], dtype=np.float64).reshape(-1, 2)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError(f"{path}:{line_number}: coordinates must be finite and normalized")
        points = np.rint(values * (size - 1)).astype(np.int32)
        mask = masks.setdefault(names[class_id], np.zeros((size, size), dtype=np.uint8))
        cv2.fillPoly(mask, [points], 1)
    return masks


def generated_name(reference_name: str) -> str:
    parts = reference_name.split("_", 2)
    return parts[2] if len(parts) == 3 and parts[0] == "task" and parts[1].isdigit() else reference_name


def evaluate(reference_dir: Path, generated_dir: Path, reference_names: dict[int, str],
             generated_names: dict[int, str], size: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    references = sorted(reference_dir.glob("*.txt"))
    if not references:
        raise ValueError(f"No reference labels found in {reference_dir}")
    for reference_path in references:
        generated_path = generated_dir / generated_name(reference_path.name)
        reference = label_masks(reference_path, reference_names, size)
        generated = label_masks(generated_path, generated_names, size)
        for label in sorted(CORE_CLASSES):
            truth = reference.get(label, np.zeros((size, size), dtype=np.uint8)).astype(bool)
            prediction = generated.get(label, np.zeros((size, size), dtype=np.uint8)).astype(bool)
            intersection = int(np.logical_and(truth, prediction).sum())
            union = int(np.logical_or(truth, prediction).sum())
            predicted = int(prediction.sum())
            actual = int(truth.sum())
            totals[label][0] += intersection
            totals[label][1] += union
            totals[label][2] += actual
            rows.append({"image": generated_path.stem, "class": label,
                         "iou": 1.0 if union == 0 else intersection / union,
                         "precision": 1.0 if predicted == 0 and actual == 0 else intersection / max(predicted, 1),
                         "recall": 1.0 if actual == 0 and predicted == 0 else intersection / max(actual, 1),
                         "reference_pixels": actual, "generated_pixels": predicted})
    per_class = {label: {"intersection": values[0], "union": values[1],
                         "iou": 1.0 if values[1] == 0 else values[0] / values[1]}
                 for label, values in sorted(totals.items())}
    present_rows = [row for row in rows if row["reference_pixels"]]
    missing_rows = [row for row in present_rows if not row["generated_pixels"]]
    summary = {"reference_images": len(references), "raster_size": size,
               "mean_iou_present_classes": sum(float(row["iou"]) for row in present_rows) / max(len(present_rows), 1),
               "present_image_classes": len(present_rows),
               "completely_missing_image_classes": len(missing_rows),
               "missing": [{"image": row["image"], "class": row["class"]} for row in missing_rows],
               "per_class": per_class}
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-labels", type=Path, default=Path("yolo_dataset_salmon_veg_rice_15/labels"))
    parser.add_argument("--reference-classes", type=Path, default=Path("yolo_dataset_salmon_veg_rice_15/classes.txt"))
    parser.add_argument("--generated-labels", type=Path, default=Path("automatic_masks/outputs/labels"))
    parser.add_argument("--generated-classes", type=Path, default=Path("yolo_dataset_salmon_veg_rice_15/classes.txt"))
    parser.add_argument("--output", type=Path, default=Path("automatic_masks/outputs/reports/evaluation.json"))
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    rows, summary = evaluate(args.reference_labels, args.generated_labels, read_names(args.reference_classes),
                             read_names(args.generated_classes), args.size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
