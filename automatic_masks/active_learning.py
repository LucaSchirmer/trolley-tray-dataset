#!/usr/bin/env python3
"""Prepare corrected consumed-tray annotations and fine-tune the segmentation model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

if __package__:
    from automatic_masks.mask_pipeline import REQUIRED_FOOD_CLASSES, ROOT, canonical_label, url_to_repo_path
else:
    from mask_pipeline import REQUIRED_FOOD_CLASSES, ROOT, canonical_label, url_to_repo_path

CLASS_NAMES = ["fish_salmon", "rice", "broccoli", "carrots"]
CLASS_IDS = {name: index for index, name in enumerate(CLASS_NAMES)}


def load_export(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Label Studio export must contain a top-level list")
    return rows


def latest_annotation(task: dict[str, Any]) -> dict[str, Any]:
    annotations = [row for row in task.get("annotations", []) if not row.get("was_cancelled", False)]
    if not annotations:
        raise ValueError(f"Task {task.get('id', '<unknown>')} has no completed human annotation")
    return max(annotations, key=lambda row: (row.get("updated_at", ""), int(row.get("id", 0))))


def yolo_rows(annotation: dict[str, Any]) -> list[str]:
    rows: list[str] = []
    for result in annotation.get("result", []):
        if result.get("type") != "polygonlabels":
            continue
        value = result.get("value", {})
        labels = value.get("polygonlabels", [])
        if len(labels) != 1:
            raise ValueError("Every polygon must contain exactly one polygon label")
        label = canonical_label(str(labels[0]))
        if label not in REQUIRED_FOOD_CLASSES:
            continue
        points = value.get("points", [])
        if len(points) < 3:
            raise ValueError(f"{label!r} polygon has fewer than three points")
        coordinates: list[str] = []
        for point in points:
            if not isinstance(point, Sequence) or len(point) < 2:
                raise ValueError(f"{label!r} polygon contains an invalid point")
            x, y = float(point[0]) / 100, float(point[1]) / 100
            if not 0 <= x <= 1 or not 0 <= y <= 1:
                raise ValueError(f"{label!r} polygon point is outside the image: {point}")
            coordinates.extend((f"{x:.6f}", f"{y:.6f}"))
        rows.append(" ".join((str(CLASS_IDS[label]), *coordinates)))
    return rows


def core_yolo_rows(path: Path, source_names: Sequence[str]) -> list[str]:
    """Remap an existing YOLO label file to the four core classes."""
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 7 or len(fields) % 2 == 0:
            raise ValueError(f"{path}:{line_number}: invalid YOLO polygon")
        class_id = int(fields[0])
        if class_id < 0 or class_id >= len(source_names):
            raise ValueError(f"{path}:{line_number}: unknown class id {class_id}")
        label = source_names[class_id]
        if label in CLASS_IDS:
            rows.append(" ".join((str(CLASS_IDS[label]), *fields[1:])))
    return rows


def prepare_sample_dataset(sample_dir: Path, output_dir: Path, val_fraction: float = .20,
                           seed: int = 0) -> dict[str, Any]:
    """Create a four-class train/validation dataset from a YOLO sample directory."""
    images_dir, labels_dir = sample_dir / "images", sample_dir / "labels"
    source_names = [line.strip() for line in (sample_dir / "classes.txt").read_text(encoding="utf-8").splitlines()
                    if line.strip()]
    images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(images) < 5:
        raise ValueError("At least five hand-labeled sample images are required")
    missing = [path.name for path in images if not (labels_dir / f"{path.stem}.txt").is_file()]
    if missing:
        raise FileNotFoundError(f"Sample images missing labels: {missing}")
    indices = list(range(len(images)))
    random.Random(seed).shuffle(indices)
    val_count = max(1, round(len(images) * val_fraction))
    val_indices = set(indices[:val_count])
    manifest, counts = [], {"train": 0, "val": 0, "polygons": 0}
    for index, source_image in enumerate(images):
        split = "val" if index in val_indices else "train"
        source_label = labels_dir / f"{source_image.stem}.txt"
        image_target = output_dir / "images" / split / source_image.name
        label_target = output_dir / "labels" / split / source_label.name
        image_target.parent.mkdir(parents=True, exist_ok=True); label_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, image_target)
        rows = core_yolo_rows(source_label, source_names)
        label_target.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        counts[split] += 1; counts["polygons"] += len(rows)
        manifest.append({"source": str(source_image.resolve()), "split": split,
                         "image": str(image_target.resolve()), "core_polygons": len(rows)})
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0])); writer.writeheader(); writer.writerows(manifest)
    yaml = [f"path: {output_dir.resolve()}", "train: images/train", "val: images/val", "names:"]
    yaml.extend(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    (output_dir / "dataset.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    summary = {"source_sample": str(sample_dir.resolve()), "output": str(output_dir.resolve()),
               "val_fraction": val_fraction, "seed": seed, **counts, "classes": CLASS_NAMES}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def prepare_checkpoint_sample_dataset(sample_dir: Path, model_report: Path, output_dir: Path,
                                      val_fraction: float = .20, seed: int = 0) -> dict[str, Any]:
    """Remap a YOLO sample to the checkpoint taxonomy without replacing its trained head."""
    report = json.loads(model_report.read_text(encoding="utf-8"))
    embedded = {int(key): str(value) for key, value in report["embedded_names"].items()}
    target_by_canonical = {canonical_label(raw): class_id for class_id, raw in embedded.items()
                           if canonical_label(raw) is not None}
    source_names = [line.strip() for line in (sample_dir / "classes.txt").read_text(encoding="utf-8").splitlines()
                    if line.strip()]
    images = sorted(path for path in (sample_dir / "images").iterdir()
                    if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    indices = list(range(len(images))); random.Random(seed).shuffle(indices)
    val_indices = set(indices[:max(1, round(len(images) * val_fraction))])
    manifest, counts = [], {"train": 0, "val": 0, "polygons": 0}
    for index, source_image in enumerate(images):
        split = "val" if index in val_indices else "train"
        source_label = sample_dir / "labels" / f"{source_image.stem}.txt"
        rows = []
        for line_number, line in enumerate(source_label.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split()
            if len(fields) < 7 or len(fields) % 2 == 0:
                raise ValueError(f"{source_label}:{line_number}: invalid YOLO polygon")
            source_id = int(fields[0]); canonical = canonical_label(source_names[source_id])
            if canonical in target_by_canonical:
                rows.append(" ".join((str(target_by_canonical[canonical]), *fields[1:])))
        image_target = output_dir / "images" / split / source_image.name
        label_target = output_dir / "labels" / split / source_label.name
        image_target.parent.mkdir(parents=True, exist_ok=True); label_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, image_target)
        label_target.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        counts[split] += 1; counts["polygons"] += len(rows)
        manifest.append({"source": str(source_image.resolve()), "split": split,
                         "image": str(image_target.resolve()), "polygons": len(rows)})
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0])); writer.writeheader(); writer.writerows(manifest)
    yaml = [f"path: {output_dir.resolve()}", "train: images/train", "val: images/val", "names:"]
    yaml.extend(f"  {index}: {embedded[index]}" for index in sorted(embedded))
    (output_dir / "dataset.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    summary = {"source_sample": str(sample_dir.resolve()), "model_report": str(model_report.resolve()),
               "output": str(output_dir.resolve()), "val_fraction": val_fraction, "seed": seed,
               **counts, "classes": [embedded[index] for index in sorted(embedded)]}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def grouped_split(tasks: Sequence[dict[str, Any]], val_fraction: float, seed: int) -> dict[int, str]:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, task in enumerate(tasks):
        data = task.get("data", task)
        group = str(data.get("unconsumed") or data.get("pair_id") or data.get("consumed") or index)
        groups[group].append(index)
    if len(groups) < 2:
        raise ValueError("At least two unconsumed-reference groups are required for a leakage-safe split")
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    target = max(1, round(len(tasks) * val_fraction))
    val_keys: set[str] = set()
    val_size = 0
    for key in keys:
        if len(val_keys) >= len(keys) - 1 or val_size >= target:
            break
        val_keys.add(key); val_size += len(groups[key])
    return {index: ("val" if key in val_keys else "train") for key, indices in groups.items() for index in indices}


def prepare_dataset(export_path: Path, output_dir: Path, val_fraction: float = .20, seed: int = 0) -> dict[str, Any]:
    tasks = load_export(export_path)
    if not tasks:
        raise ValueError("Label Studio export is empty")
    splits = grouped_split(tasks, val_fraction, seed)
    manifest: list[dict[str, Any]] = []
    counts = {"train": 0, "val": 0, "polygons": 0, "empty_core_labels": 0}
    for index, task in enumerate(tasks):
        data = task.get("data", task)
        relative = url_to_repo_path(str(data.get("consumed", "")))
        source = ROOT / relative
        if not relative or not source.is_file():
            raise FileNotFoundError(f"Consumed image for task {task.get('id', index)} not found: {source}")
        split = splits[index]
        token = hashlib.sha1(relative.encode()).hexdigest()[:10]
        filename = f"{token}_{source.name}"
        image_target = output_dir / "images" / split / filename
        label_target = output_dir / "labels" / split / f"{Path(filename).stem}.txt"
        image_target.parent.mkdir(parents=True, exist_ok=True); label_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, image_target)
        labels = yolo_rows(latest_annotation(task))
        label_target.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
        counts[split] += 1; counts["polygons"] += len(labels)
        if not labels: counts["empty_core_labels"] += 1
        manifest.append({"task_id": task.get("id", ""), "source": relative, "split": split,
                         "image": str(image_target.resolve()), "core_polygons": len(labels)})
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0])); writer.writeheader(); writer.writerows(manifest)
    yaml = [f"path: {output_dir.resolve()}", "train: images/train", "val: images/val", "names:"]
    yaml.extend(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    (output_dir / "dataset.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    summary = {"source_export": str(export_path.resolve()), "output": str(output_dir.resolve()),
               "val_fraction": val_fraction, "seed": seed, **counts, "classes": CLASS_NAMES}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def select_batch(preannotations: Path, priorities: Path, output: Path, size: int = 40, seed: int = 0
                 ) -> dict[str, Any]:
    if size < 3:
        raise ValueError("Batch size must be at least 3")
    tasks = load_export(preannotations)
    by_path = {url_to_repo_path(str(row.get("data", row).get("consumed", ""))): row for row in tasks}
    buckets: dict[str, list[dict[str, Any]]] = {"nearly_empty": [], "low_confidence": [], "regular": []}
    with priorities.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["image"] not in by_path:
                continue
            nearly_empty = row["nearly_empty"].lower() == "true"
            low_confidence = row["low_core_confidence"].lower() == "true"
            category = "nearly_empty" if nearly_empty else "low_confidence" if low_confidence else "regular"
            buckets[category].append(by_path[row["image"]])
    rng = random.Random(seed)
    for rows in buckets.values(): rng.shuffle(rows)
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    quota = size // 3
    counts: dict[str, int] = {}
    for category in ("nearly_empty", "low_confidence", "regular"):
        chosen = buckets[category][:quota]
        selected.extend(chosen); selected_ids.update(id(row) for row in chosen); counts[category] = len(chosen)
    remaining = [row for category in buckets.values() for row in category if id(row) not in selected_ids]
    rng.shuffle(remaining); selected.extend(remaining[:max(0, size - len(selected))])
    if len(selected) < min(size, len(tasks)):
        raise ValueError("Priority report does not cover enough pre-annotated tasks")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = {"output": str(output.resolve()), "requested": size, "selected": len(selected),
              "seed": seed, "initial_category_quota": counts}
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def triage_preannotations(preannotations: Path, priorities: Path, reference_images: Path,
                          output_dir: Path) -> dict[str, Any]:
    """Split predictions into practical human-review queues."""
    tasks = load_export(preannotations)
    priority_by_path: dict[str, dict[str, str]] = {}
    with priorities.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            priority_by_path[row["image"]] = row
    reference_stems = {
        path.stem.split("_all_markers_shot_", 1)[-1]
        for path in reference_images.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    grouped: dict[str, list[dict[str, Any]]] = {
        "full_redo": [], "minor_rework": [], "no_rework": []}
    audit: list[dict[str, Any]] = []
    for task in tasks:
        data = task.get("data", task)
        image = url_to_repo_path(str(data.get("consumed", "")))
        priority = priority_by_path.get(image)
        if priority is None:
            raise ValueError(f"Priority report has no row for {image}")
        reference_key = Path(image).stem.split("all_markers_shot_", 1)[-1]
        if reference_key in reference_stems:
            category = "no_rework"
            reason = "existing human-labeled reference; use the original human mask"
        elif priority["nearly_empty"].lower() == "true":
            category = "full_redo"
            reason = "near-empty tray; start blank to avoid correcting model speckles"
        else:
            category = "minor_rework"
            reason = "retain predictions; correct boundaries, misses, and false positives"
        output_data = {**data, "triage_category": category}
        output_task = ({**task, "data": output_data} if category != "full_redo"
                       else {"data": output_data})
        grouped[category].append(output_task)
        audit.append({"image": image, "category": category, "reason": reason,
                      "core_polygon_count": priority["core_polygon_count"],
                      "core_min_confidence": priority["core_min_confidence"],
                      "core_area_fraction": priority["core_area_fraction"]})
    output_dir.mkdir(parents=True, exist_ok=True)
    for category, rows in grouped.items():
        (output_dir / f"{category}.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output_dir / "triage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
        writer.writeheader(); writer.writerows(audit)
    summary = {category: len(rows) for category, rows in grouped.items()}
    summary["total"] = len(tasks)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def train(data: Path, model: Path, epochs: int, imgsz: int, batch: int, device: str, name: str,
          lr0: float | None = None, freeze: int | None = None) -> None:
    from ultralytics import YOLO

    if not data.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {data}")
    if not model.is_file():
        raise FileNotFoundError(f"Starting checkpoint not found: {model}")
    options = {"data": str(data), "epochs": epochs, "imgsz": imgsz, "batch": batch, "device": device,
               "project": str(ROOT / "automatic_masks/training_runs"), "name": name, "seed": 0,
               "deterministic": True, "patience": 20, "close_mosaic": 10}
    if lr0 is not None: options.update({"optimizer": "AdamW", "lr0": lr0, "lrf": .1})
    if freeze is not None: options["freeze"] = freeze
    YOLO(str(model)).train(**options)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Convert corrected Label Studio annotations to YOLO segmentation")
    prepare.add_argument("--export", type=Path, required=True)
    prepare.add_argument("--output", type=Path, default=ROOT / "automatic_masks/training_data/consumed_core_v1")
    prepare.add_argument("--val-fraction", type=float, default=.20)
    prepare.add_argument("--seed", type=int, default=0)
    sample = commands.add_parser("prepare-sample", help="Convert an existing YOLO sample to four core classes")
    sample.add_argument("--sample", type=Path, default=ROOT / "yolo_dataset_salmon_veg_rice_15")
    sample.add_argument("--output", type=Path, default=ROOT / "automatic_masks/training_data/sample_core_v1")
    sample.add_argument("--val-fraction", type=float, default=.20)
    sample.add_argument("--seed", type=int, default=0)
    preserve = commands.add_parser("prepare-checkpoint-sample",
                                   help="Remap a YOLO sample while preserving the checkpoint's class head")
    preserve.add_argument("--sample", type=Path, default=ROOT / "yolo_dataset_salmon_veg_rice_15")
    preserve.add_argument("--model-report", type=Path,
                          default=ROOT / "automatic_masks/outputs/reports/model_validation.json")
    preserve.add_argument("--output", type=Path, default=ROOT / "automatic_masks/training_data/sample_checkpoint_v1")
    preserve.add_argument("--val-fraction", type=float, default=.20)
    preserve.add_argument("--seed", type=int, default=0)
    select = commands.add_parser("select", help="Select a balanced Label Studio correction batch")
    select.add_argument("--preannotations", type=Path,
                        default=ROOT / "automatic_masks/outputs/label_studio/preannotated_tasks.json")
    select.add_argument("--priorities", type=Path,
                        default=ROOT / "automatic_masks/outputs/reports/review_priorities.csv")
    select.add_argument("--output", type=Path,
                        default=ROOT / "automatic_masks/outputs/label_studio/active_learning_batch.json")
    select.add_argument("--size", type=int, default=40)
    select.add_argument("--seed", type=int, default=0)
    triage = commands.add_parser("triage", help="Group masks into redo, minor-edit, and finished queues")
    triage.add_argument("--preannotations", type=Path,
                        default=ROOT / "automatic_masks/outputs/label_studio/preannotated_tasks.json")
    triage.add_argument("--priorities", type=Path,
                        default=ROOT / "automatic_masks/outputs/reports/review_priorities.csv")
    triage.add_argument("--references", type=Path,
                        default=ROOT / "yolo_dataset_salmon_veg_rice_15/images")
    triage.add_argument("--output", type=Path,
                        default=ROOT / "automatic_masks/outputs/label_studio/triage")
    fit = commands.add_parser("train", help="Fine-tune the model on a prepared consumed-tray dataset")
    fit.add_argument("--data", type=Path, default=ROOT / "automatic_masks/training_data/consumed_core_v1/dataset.yaml")
    fit.add_argument("--model", type=Path, default=ROOT / "segmentation_model/best_salmon_finetuned.pt")
    fit.add_argument("--epochs", type=int, default=50)
    fit.add_argument("--imgsz", type=int, default=1024)
    fit.add_argument("--batch", type=int, default=4)
    fit.add_argument("--device", default="0")
    fit.add_argument("--name", default="consumed-core-v1")
    fit.add_argument("--lr0", type=float)
    fit.add_argument("--freeze", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "select":
        print(json.dumps(select_batch(args.preannotations, args.priorities, args.output, args.size, args.seed), indent=2))
    elif args.command == "triage":
        print(json.dumps(triage_preannotations(args.preannotations, args.priorities,
                                               args.references, args.output), indent=2))
    elif args.command == "prepare":
        print(json.dumps(prepare_dataset(args.export, args.output, args.val_fraction, args.seed), indent=2))
    elif args.command == "prepare-sample":
        print(json.dumps(prepare_sample_dataset(args.sample, args.output, args.val_fraction, args.seed), indent=2))
    elif args.command == "prepare-checkpoint-sample":
        print(json.dumps(prepare_checkpoint_sample_dataset(args.sample, args.model_report, args.output,
                                                            args.val_fraction, args.seed), indent=2))
    else:
        train(args.data, args.model, args.epochs, args.imgsz, args.batch, args.device, args.name, args.lr0, args.freeze)


if __name__ == "__main__":
    main()
