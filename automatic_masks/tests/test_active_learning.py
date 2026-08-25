import json
import csv

import pytest

from automatic_masks import active_learning


def polygon(label="rice"):
    return {"type": "polygonlabels", "value": {"polygonlabels": [label],
            "points": [[10, 20], [30, 20], [20, 40]]}}


def task(task_id, consumed, unconsumed, results):
    return {"id": task_id, "data": {"consumed": consumed, "unconsumed": unconsumed},
            "annotations": [{"id": task_id, "result": results}]}


def test_yolo_rows_keeps_only_core_food_and_normalizes_points():
    rows = active_learning.yolo_rows({"result": [polygon("Rice"), polygon("Cookie")]})
    assert rows == ["1 0.100000 0.200000 0.300000 0.200000 0.200000 0.400000"]


def test_yolo_rows_keeps_disconnected_polygons_of_same_class():
    assert len(active_learning.yolo_rows({"result": [polygon(), polygon()]})) == 2


def test_core_yolo_rows_remaps_and_filters_classes(tmp_path):
    labels = tmp_path / "labels.txt"
    labels.write_text("2 0 0 1 0 1 1\n1 0 0 1 0 1 1\n", encoding="utf-8")
    assert active_learning.core_yolo_rows(labels, ["cookie", "rice", "fish_salmon"]) == [
        "0 0 0 1 0 1 1", "1 0 0 1 0 1 1"]


def test_prepare_dataset_groups_related_images_and_writes_empty_negatives(tmp_path, monkeypatch):
    monkeypatch.setattr(active_learning, "ROOT", tmp_path)
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (tmp_path / name).write_bytes(b"image fixture")
    rows = [task(1, "a.jpg", "reference-1.jpg", [polygon()]),
            task(2, "b.jpg", "reference-1.jpg", []),
            task(3, "c.jpg", "reference-2.jpg", [polygon("Broccoli")])]
    export = tmp_path / "export.json"; export.write_text(json.dumps(rows), encoding="utf-8")
    output = tmp_path / "dataset"
    summary = active_learning.prepare_dataset(export, output, val_fraction=.34, seed=0)
    manifest = (output / "manifest.csv").read_text(encoding="utf-8").splitlines()
    assert summary["train"] and summary["val"]
    assert summary["empty_core_labels"] == 1
    assert len(manifest) == 4
    assert (output / "dataset.yaml").is_file()
    label_files = list((output / "labels").rglob("*.txt"))
    assert len(label_files) == 3 and any(path.read_text() == "" for path in label_files)


def test_select_batch_balances_review_categories(tmp_path):
    tasks = [{"data": {"consumed": f"images/{index}.jpg"}, "predictions": []} for index in range(9)]
    preannotations = tmp_path / "preannotations.json"
    preannotations.write_text(json.dumps(tasks), encoding="utf-8")
    priorities = tmp_path / "priorities.csv"
    with priorities.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "low_core_confidence", "nearly_empty"])
        writer.writeheader()
        for index in range(9):
            writer.writerow({"image": f"images/{index}.jpg", "nearly_empty": index < 3,
                             "low_core_confidence": 3 <= index < 6})
    output = tmp_path / "batch.json"
    report = active_learning.select_batch(preannotations, priorities, output, size=9)
    assert report["selected"] == 9
    assert report["initial_category_quota"] == {"nearly_empty": 3, "low_confidence": 3, "regular": 3}
    assert len(json.loads(output.read_text())) == 9


def test_triage_uses_references_and_clears_full_redo_predictions(tmp_path):
    tasks = [{"data": {"consumed": f"images/all_markers_shot_{name}.jpg"},
              "predictions": [{"result": [polygon()]}]} for name in ("reference", "empty", "edit")]
    preannotations = tmp_path / "preannotations.json"
    preannotations.write_text(json.dumps(tasks), encoding="utf-8")
    priorities = tmp_path / "priorities.csv"
    fields = ["image", "core_polygon_count", "core_min_confidence", "core_area_fraction", "nearly_empty"]
    with priorities.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for name, empty in (("reference", True), ("empty", True), ("edit", False)):
            writer.writerow({"image": f"images/all_markers_shot_{name}.jpg", "core_polygon_count": 1,
                             "core_min_confidence": .5, "core_area_fraction": .005 if empty else .1,
                             "nearly_empty": empty})
    references = tmp_path / "references"; references.mkdir()
    (references / "task_1_all_markers_shot_reference.jpg").write_bytes(b"fixture")
    output = tmp_path / "triage"
    summary = active_learning.triage_preannotations(preannotations, priorities, references, output)
    assert summary == {"full_redo": 1, "minor_rework": 1, "no_rework": 1, "total": 3}
    assert "predictions" not in json.loads((output / "full_redo.json").read_text())[0]
    assert "predictions" in json.loads((output / "minor_rework.json").read_text())[0]
