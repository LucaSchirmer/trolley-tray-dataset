#!/usr/bin/env python3
"""Generate filtered YOLO segmentation labels and Label Studio predictions."""
from __future__ import annotations

import argparse
import colorsys
import csv
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = ROOT / "annotations/pairs_fish_rice_labelstudio.json"
DEFAULT_IMAGES = ROOT / "images_cropped/consumed/fish_rice"
DEFAULT_MODEL = ROOT / "segmentation_model/best_salmon_finetuned.pt"
DEFAULT_CLASSES = ROOT / "yolo_dataset_salmon_veg_rice_15/classes.txt"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs"
DEFAULT_FISH_CONFIDENCE = .10
EXPECTED_UNIQUE = 131
REQUIRED_FOOD_CLASSES = {"fish_salmon", "rice", "broccoli", "carrots"}
# The source checkpoint was trained jointly on visually similar chicken and salmon trays.
# A fish-only task cannot contain chicken, so use that known task taxonomy to resolve the
# systematic late-meal confusion. The override is applied only when chicken is forbidden.
CONDITIONAL_CLASS_OVERRIDES = {"chicken": "fish_salmon"}

LABEL_ALIASES = {
    "salmon": "fish_salmon", "fish salmon": "fish_salmon", "fish_salmon": "fish_salmon",
    "rice": "rice", "broccoli": "broccoli", "carrot": "carrots", "carrots": "carrots",
    "bread roll": "bread_roll", "bread_roll": "bread_roll", "side salad": "side_salad",
    "side_salad": "side_salad", "chocolate cake": "brownie", "brownie": "brownie",
    "vanilla pudding with fruits": "vanilla_pudding_with_fruits",
    "vanilla_pudding_with_fruits": "vanilla_pudding_with_fruits", "fruit salad": "fruit_salad",
    "fruit_salad": "fruit_salad", "water": "water_bottle", "water bottle": "water_bottle",
    "water_bottle": "water_bottle", "coffee": "coffee_cup", "coffee cup": "coffee_cup",
    "coffee_cup": "coffee_cup", "tea": "tea_cup", "tea cup": "tea_cup", "tea_cup": "tea_cup",
    "orange juice": "orange_juice_bottle", "orange juice bottle": "orange_juice_bottle",
    "orange_juice_bottle": "orange_juice_bottle", "cola": "cola_can", "cola can": "cola_can",
    "cola_can": "cola_can", "honey": "honey", "plum jam": "plum_jam", "plum_jam": "plum_jam",
    "cherry jam": "cherry_jam", "cherry_jam": "cherry_jam", "butter": "butter", "cookie": "cookie",
    "chicken": "chicken", "main salad": "salad_main", "main_salad": "salad_main",
    "salad main": "salad_main", "salad_main": "salad_main",
    "wrap half 1": "wrap_half_1", "wrap_half_1": "wrap_half_1", "wrap half 2": "wrap_half_2",
    "wrap_half_2": "wrap_half_2", "pasta pesto": "pasta_pesto", "pasta_pesto": "pasta_pesto",
}

COLORS = {"fish_salmon": (96, 160, 244), "rice": (217, 144, 74), "broccoli": (92, 184, 92),
          "carrots": (0, 127, 255)}
CANONICAL_LABELS = sorted(set(LABEL_ALIASES.values()))


class CheckpointTaxonomyError(ValueError):
    """The checkpoint cannot produce the labels required by this workflow."""


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[-_]+", " ", name.strip().lower())).strip()


def canonical_label(name: str) -> str | None:
    raw = name.strip().lower()
    return LABEL_ALIASES.get(raw) or LABEL_ALIASES.get(normalize_name(name))


def url_to_repo_path(value: str) -> str:
    if not value:
        return ""
    parsed = urlparse(value)
    return unquote(parsed.path.lstrip("/")) if parsed.scheme or parsed.netloc else unquote(value.lstrip("/"))


@dataclass
class Task:
    source_index: int
    consumed_url: str
    consumed_path: str
    unconsumed_url: str
    possible_raw: list[str]
    allowed: set[str]
    original: dict[str, Any]
    duplicate_indices: list[int] = field(default_factory=list)


@dataclass
class Polygon:
    class_id: int
    label: str
    confidence: float
    points: list[tuple[float, float]]  # pixel coordinates
    width: int
    height: int


def load_tasks(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Task JSON must contain a top-level list")
    return data


def deduplicate_tasks(rows: Sequence[dict[str, Any]]) -> tuple[list[Task], dict[str, Any]]:
    unique: dict[str, Task] = {}
    duplicates, conflicts, invalid_rows = [], [], []
    for index, row in enumerate(rows):
        consumed = row.get("consumed") or row.get("data", {}).get("consumed", "")
        unconsumed = row.get("unconsumed") or row.get("data", {}).get("unconsumed", "")
        possible = row.get("possibleElements") or row.get("data", {}).get("possibleElements", [])
        path = url_to_repo_path(consumed)
        if not path:
            raise ValueError(f"Task {index} has no consumed image")
        unknown = [x for x in possible if canonical_label(str(x)) is None]
        if unknown:
            raise ValueError(f"Task {index} has unmapped possibleElements: {unknown}")
        task = Task(index, consumed, path, unconsumed, list(possible),
                    {canonical_label(str(x)) for x in possible if canonical_label(str(x))}, dict(row))
        if not unconsumed or unconsumed == "invalid":
            invalid_rows.append(index)
        if path in unique:
            kept = unique[path]
            kept.duplicate_indices.append(index)
            duplicate = {"consumed": path, "kept_row": kept.source_index, "duplicate_row": index}
            duplicates.append(duplicate)
            if kept.unconsumed_url != unconsumed or kept.allowed != task.allowed:
                conflicts.append({**duplicate, "kept_unconsumed": kept.unconsumed_url,
                                  "duplicate_unconsumed": unconsumed,
                                  "kept_possibleElements": kept.possible_raw,
                                  "duplicate_possibleElements": task.possible_raw})
            continue
        unique[path] = task
    invalid_unique = [t.consumed_path for t in unique.values() if not t.unconsumed_url or t.unconsumed_url == "invalid"]
    report = {"input_rows": len(rows), "unique_consumed_images": len(unique),
              "duplicate_occurrences": len(duplicates), "duplicates": duplicates,
              "conflicting_reference_assignments": conflicts,
              "invalid_reference_rows": invalid_rows, "invalid_reference_unique_images": invalid_unique}
    return list(unique.values()), report


def audit_physical_images(tasks: Sequence[Task], image_dir: Path) -> list[str]:
    selected = {Path(t.consumed_path).name for t in tasks}
    return sorted(p.name for p in image_dir.iterdir() if p.is_file() and p.name not in selected)


def include_unlisted_images(tasks: Sequence[Task], image_dir: Path) -> list[Task]:
    """Add physical images omitted from the task export so every image is processed."""
    expanded = list(tasks)
    selected = {Path(task.consumed_path).name for task in tasks}
    for image_path in sorted(path for path in image_dir.iterdir() if path.is_file() and path.name not in selected):
        try:
            relative = image_path.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError:  # Keeps the helper usable with isolated test fixtures.
            relative = image_path.as_posix().lstrip("/")
        possible = ["Salmon", "Rice", "Broccoli", "Carrots"]
        expanded.append(Task(-1, f"http://localhost:8000/{relative}", relative, "invalid", possible,
                             set(REQUIRED_FOOD_CLASSES), {"auto_added": True}))
    return expanded


def load_export_class_ids(path: Path) -> dict[str, int]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate export class names in {path}: {duplicates}")
    missing = sorted(REQUIRED_FOOD_CLASSES - set(names))
    if missing:
        raise ValueError(f"Export taxonomy {path} is missing required classes: {missing}")
    return {name: index for index, name in enumerate(names)}


def validate_model_names(names: dict[int, str] | Sequence[str]) -> dict[int, str]:
    items = names.items() if isinstance(names, dict) else enumerate(names)
    mapped, unknown = {}, []
    for class_id, name in items:
        canonical = canonical_label(str(name))
        if canonical is None:
            unknown.append(str(name))
        else:
            mapped[int(class_id)] = canonical
    missing = sorted(REQUIRED_FOOD_CLASSES - set(mapped.values()))
    if unknown or missing:
        raw_names = [str(name) for name in (names.values() if isinstance(names, dict) else names)]
        stock_coco = len(raw_names) == 80 and {"person", "bicycle", "car"}.issubset(raw_names)
        kind = "stock COCO checkpoint" if stock_coco else "incompatible checkpoint"
        unknown_preview = unknown[:12]
        unknown_suffix = f" (and {len(unknown) - len(unknown_preview)} more)" if len(unknown) > len(unknown_preview) else ""
        raise CheckpointTaxonomyError(
            f"{kind}: missing required classes {missing}. "
            f"Expected a custom fish-and-rice model containing {sorted(REQUIRED_FOOD_CLASSES)}. "
            f"Found {len(raw_names)} embedded classes; unmapped examples: {unknown_preview}{unknown_suffix}. "
            "See automatic_masks/outputs/reports/model_validation.json for the complete taxonomy."
        )
    return mapped


def load_and_validate_model(model_path: Path, output_dir: Path):
    """Load a checkpoint, validate its embedded taxonomy, and always write a report."""
    from ultralytics import YOLO

    resolved = model_path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    model = YOLO(str(resolved))
    model_report: dict[str, Any] = {"checkpoint": str(resolved), "embedded_names": model.names}
    try:
        class_map = validate_model_names(model.names)
    except CheckpointTaxonomyError as exc:
        message = f"Checkpoint taxonomy validation failed for {resolved}. {exc}"
        model_report.update({"valid": False, "error": message})
        write_json(output_dir / "reports/model_validation.json", model_report)
        raise CheckpointTaxonomyError(message) from None
    model_report.update({"valid": True, "canonical_mapping": class_map})
    write_json(output_dir / "reports/model_validation.json", model_report)
    return model, class_map


def clean_polygon(points: Iterable[Sequence[float]], width: int, height: int) -> list[tuple[float, float]] | None:
    try:
        clean = [(float(p[0]), float(p[1])) for p in points]
    except (TypeError, ValueError, IndexError):
        return None
    if len(clean) < 3 or width <= 0 or height <= 0 or not all(math.isfinite(v) for p in clean for v in p):
        return None
    clean = [(min(max(x, 0.0), float(width)), min(max(y, 0.0), float(height))) for x, y in clean]
    area = abs(sum(clean[i][0] * clean[(i + 1) % len(clean)][1] - clean[(i + 1) % len(clean)][0] * clean[i][1]
                   for i in range(len(clean))) / 2)
    return clean if area >= 1.0 else None


def filter_predictions(predictions: Sequence[Polygon], allowed: set[str]) -> tuple[list[Polygon], list[dict[str, Any]]]:
    kept: list[Polygon] = []
    rejected: list[dict[str, Any]] = []
    for pred in predictions:
        reason = None
        if pred.label not in allowed:
            reason = "not_in_possibleElements"
        elif clean_polygon(pred.points, pred.width, pred.height) is None:
            reason = "malformed_polygon"
        if reason:
            rejected.append({"class": pred.label, "confidence": pred.confidence, "reason": reason})
        else:
            # These are instance-segmentation outputs. Repeated food classes are usually
            # separate rice clusters, carrot pieces, or broccoli florets, not duplicates.
            # Ultralytics has already applied mask/box NMS using the requested IoU threshold.
            kept.append(pred)
    return sorted(kept, key=lambda p: (p.class_id, -p.confidence)), rejected


def apply_task_class_overrides(predictions: Sequence[Polygon], allowed: set[str]) -> tuple[list[Polygon], list[dict[str, Any]]]:
    corrected, changes = [], []
    for pred in predictions:
        target = CONDITIONAL_CLASS_OVERRIDES.get(pred.label)
        if target in allowed and pred.label not in allowed:
            changes.append({"from": pred.label, "to": target, "confidence": pred.confidence,
                            "reason": "target_allowed_source_forbidden"})
            corrected.append(Polygon(pred.class_id, target, pred.confidence, pred.points, pred.width, pred.height))
        else:
            corrected.append(pred)
    return corrected, changes


def apply_class_confidence_thresholds(predictions: Sequence[Polygon], allowed: set[str], default: float,
                                      override: float, class_thresholds: dict[str, float] | None = None
                                      ) -> tuple[list[Polygon], list[dict[str, Any]]]:
    kept, rejected = [], []
    class_thresholds = class_thresholds or {}
    for pred in predictions:
        target = CONDITIONAL_CLASS_OVERRIDES.get(pred.label)
        if target in allowed and pred.label not in allowed:
            threshold = override
        else:
            threshold = class_thresholds.get(pred.label, default)
        if pred.confidence < threshold:
            rejected.append({"class": pred.label, "confidence": pred.confidence,
                             "threshold": threshold, "reason": "below_class_confidence"})
        else:
            kept.append(pred)
    return kept, rejected


def yolo_line(pred: Polygon, export_class_id: int | None = None) -> str:
    pts = clean_polygon(pred.points, pred.width, pred.height)
    if pts is None:
        raise ValueError("Malformed polygon")
    coords = " ".join(f"{x / pred.width:.6f} {y / pred.height:.6f}" for x, y in pts)
    return f"{pred.class_id if export_class_id is None else export_class_id} {coords}"


def label_studio_result(pred: Polygon, result_id: str) -> dict[str, Any]:
    pts = clean_polygon(pred.points, pred.width, pred.height)
    if pts is None:
        raise ValueError("Malformed polygon")
    return {"id": result_id, "from_name": "masks_consumed", "to_name": "img_consumed", "type": "polygonlabels",
            "original_width": pred.width, "original_height": pred.height, "image_rotation": 0,
            "value": {"points": [[100 * x / pred.width, 100 * y / pred.height] for x, y in pts],
                      "polygonlabels": [pred.label]}, "score": pred.confidence}


def stable_result_id(image_path: str, label: str, points: Sequence[tuple[float, float]]) -> str:
    geometry = ";".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return hashlib.sha1(f"{image_path}:{label}:{geometry}".encode()).hexdigest()[:10]


def color_for_label(label: str) -> tuple[int, int, int]:
    if label in COLORS:
        return COLORS[label]
    index = CANONICAL_LABELS.index(label) if label in CANONICAL_LABELS else 0
    red, green, blue = colorsys.hsv_to_rgb(index / max(len(CANONICAL_LABELS), 1), .78, .95)
    return round(255 * blue), round(255 * green), round(255 * red)


def polygon_area_fraction(predictions: Sequence[Polygon], width: int, height: int) -> float:
    area = 0.0
    for pred in predictions:
        area += abs(sum(pred.points[i][0] * pred.points[(i + 1) % len(pred.points)][1] -
                        pred.points[(i + 1) % len(pred.points)][0] * pred.points[i][1]
                        for i in range(len(pred.points))) / 2)
    return area / (width * height) if width > 0 and height > 0 else 0.0


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_inputs(tasks_path: Path, images_dir: Path, output_dir: Path, require_count: bool = True) -> tuple[list[Task], dict[str, Any]]:
    tasks, report = deduplicate_tasks(load_tasks(tasks_path))
    missing = [t.consumed_path for t in tasks if not (ROOT / t.consumed_path).is_file()]
    outside = audit_physical_images(tasks, images_dir)
    report.update({"missing_selected_images": missing, "physical_images_outside_selection": outside,
                   "physical_images_outside_selection_count": len(outside), "expected_unique": EXPECTED_UNIQUE})
    write_json(output_dir / "reports/input_validation.json", report)
    if missing:
        raise ValueError(f"{len(missing)} selected images are missing; see input_validation.json")
    tasks = include_unlisted_images(tasks, images_dir)
    report["processed_after_auto_add"] = len(tasks)
    if require_count and len(tasks) != EXPECTED_UNIQUE:
        raise ValueError(f"Expected {EXPECTED_UNIQUE} physical images after auto-add, found {len(tasks)}")
    return tasks, report


def load_native_export(path: Path) -> dict[str, int]:
    mapping = {}
    for row in load_tasks(path):
        if "id" not in row:
            raise ValueError("Every native export task must have an id")
        data = row.get("data", row)
        consumed = url_to_repo_path(data.get("consumed", ""))
        if not consumed:
            raise ValueError(f"Native export task {row['id']} has no consumed URL")
        if consumed in mapping:
            raise ValueError(f"Native export contains duplicate consumed image: {consumed}")
        mapping[consumed] = row["id"]
    return mapping


def render_preview(image: Any, predictions: Sequence[Polygon]) -> Any:
    import cv2
    import numpy as np

    canvas = image.copy()
    overlay = image.copy()
    for pred in predictions:
        points = np.array(pred.points, dtype=np.int32)
        color = color_for_label(pred.label)
        cv2.fillPoly(overlay, [points], color)
        cv2.polylines(canvas, [points], True, color, 3)
    canvas = cv2.addWeighted(overlay, .28, canvas, .72, 0)
    for pred in predictions:
        points = np.array(pred.points, dtype=np.int32)
        color = color_for_label(pred.label)
        text = f"{pred.label} {pred.confidence:.2f}"
        x, y = tuple(points[points[:, 1].argmin()])
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .55, 2)
        x = min(max(int(x), 0), max(canvas.shape[1] - text_width - 4, 0))
        y = min(max(int(y), text_height + 4), canvas.shape[0] - baseline - 2)
        cv2.rectangle(canvas, (x, y - text_height - 4), (x + text_width + 4, y + baseline + 2), (20, 20, 20), -1)
        cv2.putText(canvas, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2, cv2.LINE_AA)
    return canvas


def make_contact_sheets(preview_paths: Sequence[Path], output_dir: Path, group: str,
                        columns: int = 4, thumb_width: int = 480) -> None:
    group_dir = output_dir / "contact_sheets" / group
    group_dir.mkdir(parents=True, exist_ok=True)
    for stale in group_dir.glob("sheet_*.jpg"):
        stale.unlink()
    if not preview_paths:
        return
    import cv2
    import numpy as np
    page_size = columns * 5
    for page, start in enumerate(range(0, len(preview_paths), page_size), 1):
        images = []
        for path in preview_paths[start:start + page_size]:
            image = cv2.imread(str(path))
            if image is None:
                continue
            scale = thumb_width / image.shape[1]
            images.append(cv2.resize(image, (thumb_width, int(image.shape[0] * scale))))
        if not images:
            continue
        h = max(x.shape[0] for x in images)
        blank = np.full((h, thumb_width, 3), 255, dtype=np.uint8)
        rows = []
        for i in range(0, len(images), columns):
            row = images[i:i + columns] + [blank] * (columns - len(images[i:i + columns]))
            row = [cv2.copyMakeBorder(x, 0, h - x.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255)) for x in row]
            rows.append(cv2.hconcat(row))
        out = group_dir / f"sheet_{page:02d}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), cv2.vconcat(rows))


def run_inference(tasks: Sequence[Task], model_path: Path, output_dir: Path, native_export: Path | None = None,
                  device: str = "0", imgsz: int = 1024, conf: float = .20, iou: float = .70,
                  classes_path: Path = DEFAULT_CLASSES, override_conf: float = .08,
                  fish_conf: float = DEFAULT_FISH_CONFIDENCE) -> None:
    import cv2
    import numpy as np
    import torch

    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    model, class_map = load_and_validate_model(model_path, output_dir)
    export_class_ids = load_export_class_ids(classes_path)
    labels_dir = output_dir / "labels"
    all_preview_dir, core_preview_dir = output_dir / "previews/all_classes", output_dir / "previews/core_food"
    labels_dir.mkdir(parents=True, exist_ok=True)
    all_preview_dir.mkdir(parents=True, exist_ok=True); core_preview_dir.mkdir(parents=True, exist_ok=True)
    for stale in labels_dir.glob("*.txt"):
        stale.unlink()
    preannotated, predictions_by_path, confidence_rows, rejected_rows, priority_rows = [], {}, [], [], []
    all_previews, core_previews, low_confidence_previews, nearly_empty_previews = [], [], [], []
    for task in tasks:
        image_path = ROOT / task.consumed_path
        result = model.predict(source=str(image_path), imgsz=imgsz, conf=min(conf, override_conf, fish_conf), iou=iou, device=device,
                               deterministic=True, verbose=False)[0]
        image = cv2.imread(str(image_path))
        if image is None: raise ValueError(f"Could not read image: {image_path}")
        height, width = image.shape[:2]
        raw = []
        if result.boxes is not None and result.masks is not None:
            classes = result.boxes.cls.detach().cpu().tolist()
            scores = result.boxes.conf.detach().cpu().tolist()
            polygons = result.masks.xy
            for class_id_f, score, points in zip(classes, scores, polygons):
                class_id = int(class_id_f)
                label = class_map.get(class_id)
                if label is None:
                    rejected_rows.append({"image": task.consumed_path, "class_id": class_id, "reason": "unmapped_model_class"})
                    continue
                raw.append(Polygon(class_id, label, float(score), [tuple(x) for x in points.tolist()], width, height))
        raw, threshold_rejections = apply_class_confidence_thresholds(
            raw, task.allowed, conf, override_conf, {"fish_salmon": fish_conf})
        for row in threshold_rejections: rejected_rows.append({"image": task.consumed_path, **row})
        raw, overrides = apply_task_class_overrides(raw, task.allowed)
        for row in overrides: rejected_rows.append({"image": task.consumed_path, "reason": "class_override", **row})
        kept, rejected = filter_predictions(raw, task.allowed)
        for row in rejected: rejected_rows.append({"image": task.consumed_path, **row})
        (labels_dir / f"{image_path.stem}.txt").write_text(
            "\n".join(yolo_line(x, export_class_ids[x.label]) for x in kept) + ("\n" if kept else ""), encoding="utf-8")
        ls_results = []
        for pred in kept:
            rid = stable_result_id(task.consumed_path, pred.label, pred.points)
            ls_results.append(label_studio_result(pred, rid))
            confidence_rows.append({"image": task.consumed_path, "class": pred.label, "class_id": pred.class_id,
                                    "confidence": f"{pred.confidence:.6f}"})
        core_kept = [pred for pred in kept if pred.label in REQUIRED_FOOD_CLASSES]
        all_preview_path = all_preview_dir / image_path.name
        core_preview_path = core_preview_dir / image_path.name
        cv2.imwrite(str(all_preview_path), render_preview(image, kept))
        cv2.imwrite(str(core_preview_path), render_preview(image, core_kept))
        all_previews.append(all_preview_path); core_previews.append(core_preview_path)
        core_area = polygon_area_fraction(core_kept, width, height)
        core_min_confidence = min((pred.confidence for pred in core_kept), default=None)
        low_core_confidence = core_min_confidence is not None and core_min_confidence < .40
        nearly_empty = core_area < .01
        if low_core_confidence: low_confidence_previews.append(core_preview_path)
        if nearly_empty: nearly_empty_previews.append(core_preview_path)
        priority_rows.append({"image": task.consumed_path, "core_polygon_count": len(core_kept),
                              "core_min_confidence": "" if core_min_confidence is None else f"{core_min_confidence:.6f}",
                              "core_area_fraction": f"{core_area:.6f}",
                              "low_core_confidence": low_core_confidence, "nearly_empty": nearly_empty})
        prediction = {"model_version": model_path.name, "score": max((x.confidence for x in kept), default=0.0), "result": ls_results}
        data = {"unconsumed": task.unconsumed_url, "consumed": task.consumed_url,
                "possibleElements": task.possible_raw}
        if "pair_id" in task.original: data["pair_id"] = task.original["pair_id"]
        preannotated.append({"data": data, "predictions": [prediction]})
        predictions_by_path[task.consumed_path] = prediction
    write_json(output_dir / "label_studio/preannotated_tasks.json", preannotated)
    if native_export:
        ids = load_native_export(native_export)
        missing = sorted(set(predictions_by_path) - set(ids))
        if missing: raise ValueError(f"Native export is missing {len(missing)} selected consumed images")
        payload = [{"task": ids[path], **prediction} for path, prediction in predictions_by_path.items()]
        write_json(output_dir / "label_studio/api_predictions.json", payload)
    with (output_dir / "reports/confidences.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "class", "class_id", "confidence"]); writer.writeheader(); writer.writerows(confidence_rows)
    with (output_dir / "reports/review_priorities.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["image", "core_polygon_count", "core_min_confidence", "core_area_fraction",
                  "low_core_confidence", "nearly_empty"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(priority_rows)
    write_json(output_dir / "reports/rejected_polygons.json", rejected_rows)
    make_contact_sheets(all_previews, output_dir, "all_classes")
    make_contact_sheets(core_previews, output_dir, "core_food")
    make_contact_sheets(low_confidence_previews, output_dir, "low_confidence")
    make_contact_sheets(nearly_empty_previews, output_dir, "nearly_empty")
    summary = {"processed_unique_images": len(tasks), "yolo_label_files": len(list(labels_dir.glob("*.txt"))),
               "preannotated_tasks": len(preannotated), "retained_polygons": len(confidence_rows),
               "rejected_polygons": len(rejected_rows),
               "retained_core_food_polygons": sum(int(row["core_polygon_count"]) for row in priority_rows),
               "review_sets": {"all_classes": len(all_previews), "core_food": len(core_previews),
                               "low_confidence": len(low_confidence_previews),
                               "nearly_empty": len(nearly_empty_previews)},
               "settings": {"imgsz": imgsz, "conf": conf, "iou": iou,
               "fish_conf": fish_conf, "override_conf": override_conf,
               "device": device, "deterministic": True}}
    write_json(output_dir / "reports/run_summary.json", summary)
    if summary["yolo_label_files"] != len(tasks): raise RuntimeError("Not every selected image received a YOLO label file")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "infer", "run"):
        p = sub.add_parser(name)
        p.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
        p.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
        p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
        p.add_argument("--allow-nonstandard-count", action="store_true")
        if name != "validate":
            p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
            p.add_argument("--native-export", type=Path)
            p.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
            p.add_argument("--device", default="0")
            p.add_argument("--imgsz", type=int, default=1024)
            p.add_argument("--conf", type=float, default=.20)
            p.add_argument("--fish-conf", type=float, default=DEFAULT_FISH_CONFIDENCE,
                           help="Confidence for direct fish_salmon predictions (calibrated on hand labels)")
            p.add_argument("--override-conf", type=float, default=.08,
                           help="Confidence for a source class corrected by a task-taxonomy override")
            p.add_argument("--iou", type=float, default=.70)
    p = sub.add_parser("convert", help="Build an API payload from existing pre-annotations and a native export")
    p.add_argument("--preannotations", type=Path, default=DEFAULT_OUTPUT / "label_studio/preannotated_tasks.json")
    p.add_argument("--native-export", type=Path, required=True)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "label_studio/api_predictions.json")
    p = sub.add_parser("validate-model", help="Validate checkpoint taxonomy without running inference")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def convert_payload(preannotations: Path, native_export: Path, output: Path) -> None:
    ids = load_native_export(native_export); payload = []
    for row in load_tasks(preannotations):
        data = row.get("data", row); path = url_to_repo_path(data.get("consumed", ""))
        if path not in ids: raise ValueError(f"No native task id for {path}")
        for prediction in row.get("predictions", []): payload.append({"task": ids[path], **prediction})
    write_json(output, payload)


def main() -> None:
    args = parse_args()
    if args.command == "validate-model":
        _, class_map = load_and_validate_model(args.model, args.output)
        print(f"Validated checkpoint: {args.model.expanduser().resolve()}")
        print(f"Canonical classes: {sorted(set(class_map.values()))}")
        return
    if args.command == "convert":
        convert_payload(args.preannotations, args.native_export, args.output); print(f"Wrote {args.output}"); return
    tasks, report = validate_inputs(args.tasks, args.images, args.output, not args.allow_nonstandard_count)
    print(f"Validated {len(tasks)} unique images; {report['duplicate_occurrences']} duplicate rows; "
          f"{len(report['invalid_reference_unique_images'])} invalid-reference tasks; "
          f"{report['physical_images_outside_selection_count']} physical images outside selection")
    if args.command != "validate":
        run_inference(tasks, args.model, args.output, args.native_export, args.device, args.imgsz, args.conf, args.iou,
                      args.classes, args.override_conf, args.fish_conf)
        print(f"Generated masks in {args.output}")


if __name__ == "__main__":
    try:
        main()
    except (CheckpointTaxonomyError, FileNotFoundError) as exc:
        raise SystemExit(f"Error: {exc}") from None
