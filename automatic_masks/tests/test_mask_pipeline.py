import json
from pathlib import Path

import pytest

from automatic_masks.mask_pipeline import (Polygon, apply_class_confidence_thresholds, apply_task_class_overrides,
                                           canonical_label, clean_polygon,
                                           color_for_label, convert_payload,
                                           CheckpointTaxonomyError, deduplicate_tasks, filter_predictions,
                                           include_unlisted_images, label_studio_result, load_export_class_ids,
                                           stable_result_id, validate_model_names,
                                           yolo_line)


def task(consumed="http://localhost:8000/images/a.jpg", unconsumed="images/u.jpg", possible=None):
    return {"consumed": consumed, "unconsumed": unconsumed,
            "possibleElements": possible or ["Salmon", "Rice", "Broccoli", "Carrots"]}


def test_class_mapping_and_checkpoint_validation():
    assert canonical_label("Salmon") == "fish_salmon"
    assert canonical_label("main_salad") == "salad_main"
    assert canonical_label("orange_juice_bottle") == "orange_juice_bottle"
    assert validate_model_names({0: "Salmon", 1: "rice", 2: "broccoli", 3: "carrots"})[0] == "fish_salmon"
    with pytest.raises(CheckpointTaxonomyError, match="missing required classes"):
        validate_model_names({0: "rice"})


def test_stock_coco_checkpoint_has_actionable_error():
    names = {index: name for index, name in enumerate(["person", "bicycle", "car"] + [f"c{n}" for n in range(77)])}
    with pytest.raises(CheckpointTaxonomyError, match="stock COCO checkpoint"):
        validate_model_names(names)


def test_deduplication_reports_reference_conflict_and_keeps_first():
    rows = [task(), task(unconsumed="images/other.jpg", possible=["Salmon", "Rice", "Broccoli", "Carrots", "Cookie"])]
    tasks, report = deduplicate_tasks(rows)
    assert len(tasks) == 1 and tasks[0].unconsumed_url == "images/u.jpg"
    assert tasks[0].duplicate_indices == [1]
    assert report["duplicate_occurrences"] == 1
    assert len(report["conflicting_reference_assignments"]) == 1


def test_filtering_retains_all_valid_instances_per_allowed_class():
    low = Polygon(0, "rice", .3, [(0, 0), (10, 0), (5, 10)], 20, 20)
    high = Polygon(0, "rice", .9, [(1, 1), (12, 1), (6, 12)], 20, 20)
    forbidden = Polygon(1, "cookie", .99, [(0, 0), (10, 0), (5, 10)], 20, 20)
    kept, rejected = filter_predictions([low, high, forbidden], {"rice"})
    assert kept == [high, low]
    assert {x["reason"] for x in rejected} == {"not_in_possibleElements"}


def test_filtering_keeps_disconnected_regions_of_one_class():
    left = Polygon(0, "rice", .9, [(0, 0), (10, 0), (10, 10), (0, 10)], 100, 100)
    duplicate = Polygon(0, "rice", .8, [(1, 1), (11, 1), (11, 11), (1, 11)], 100, 100)
    right = Polygon(0, "rice", .7, [(50, 50), (60, 50), (60, 60), (50, 60)], 100, 100)
    kept, rejected = filter_predictions([left, duplicate, right], {"rice"})
    assert kept == [left, duplicate, right]
    assert rejected == []


def test_fish_task_relabels_forbidden_chicken_as_salmon():
    chicken = Polygon(5, "chicken", .8, [(0, 0), (10, 0), (5, 10)], 20, 20)
    corrected, changes = apply_task_class_overrides([chicken], {"fish_salmon", "rice"})
    assert corrected[0].label == "fish_salmon"
    assert changes[0]["reason"] == "target_allowed_source_forbidden"
    unchanged, changes = apply_task_class_overrides([chicken], {"chicken", "rice"})
    assert unchanged[0].label == "chicken" and changes == []


def test_task_override_has_lower_class_specific_confidence():
    chicken = Polygon(5, "chicken", .10, [(0, 0), (10, 0), (5, 10)], 20, 20)
    rice = Polygon(16, "rice", .10, [(0, 0), (10, 0), (5, 10)], 20, 20)
    kept, rejected = apply_class_confidence_thresholds([chicken, rice], {"fish_salmon", "rice"}, .20, .05)
    assert kept == [chicken]
    assert rejected[0]["class"] == "rice"


def test_direct_salmon_uses_calibrated_class_threshold():
    salmon = Polygon(10, "fish_salmon", .11, [(0, 0), (10, 0), (5, 10)], 20, 20)
    rice = Polygon(16, "rice", .11, [(0, 0), (10, 0), (5, 10)], 20, 20)
    kept, rejected = apply_class_confidence_thresholds(
        [salmon, rice], {"fish_salmon", "rice"}, .20, .08, {"fish_salmon": .10})
    assert kept == [salmon]
    assert rejected[0]["class"] == "rice"


def test_result_ids_are_unique_for_multiple_polygons_of_one_class():
    first = stable_result_id("image.jpg", "rice", [(0, 0), (1, 0), (1, 1)])
    second = stable_result_id("image.jpg", "rice", [(2, 2), (3, 2), (3, 3)])
    assert first != second


def test_review_colors_distinguish_non_core_classes():
    labels = ["bread_roll", "butter", "cookie", "coffee_cup", "cola_can", "fruit_salad"]
    assert len({color_for_label(label) for label in labels}) == len(labels)


def test_polygon_conversions_match_yolo_and_label_studio_contract():
    pred = Polygon(2, "broccoli", .8, [(10, 20), (30, 20), (20, 40)], 100, 200)
    assert yolo_line(pred).startswith("2 0.100000 0.100000")
    assert yolo_line(pred, 17).startswith("17 0.100000 0.100000")
    result = label_studio_result(pred, "abc")
    assert result["from_name"] == "masks_consumed" and result["to_name"] == "img_consumed"
    assert result["value"]["points"][0] == [10.0, 10.0]
    assert set(result) <= {"id", "from_name", "to_name", "type", "original_width", "original_height", "image_rotation", "value", "score"}


@pytest.mark.parametrize("points", [[], [(0, 0), (1, 1)], [(0, 0), (1, 1), (2, 2)], [(0, 0), (1, float("nan")), (2, 0)], None])
def test_malformed_masks_are_rejected(points):
    assert clean_polygon(points or [], 100, 100) is None


def test_invalid_reference_task_is_kept():
    tasks, report = deduplicate_tasks([task(unconsumed="")])
    assert len(tasks) == 1
    assert report["invalid_reference_unique_images"] == ["images/a.jpg"]


def test_unlisted_physical_images_are_added_with_core_taxonomy(tmp_path: Path):
    (tmp_path / "listed.jpg").touch(); (tmp_path / "extra.jpg").touch()
    listed = Task = deduplicate_tasks([task(consumed=f"{tmp_path.name}/listed.jpg")])[0][0]
    # Use a repo-relative-looking consumed path while the physical audit uses file names.
    listed.consumed_path = "images/listed.jpg"
    expanded = include_unlisted_images([listed], tmp_path)
    assert len(expanded) == 2
    assert expanded[1].allowed == {"fish_salmon", "rice", "broccoli", "carrots"}


def test_export_taxonomy_controls_yolo_ids(tmp_path: Path):
    classes = tmp_path / "classes.txt"
    classes.write_text("rice\nfish_salmon\nbroccoli\ncarrots\n", encoding="utf-8")
    assert load_export_class_ids(classes)["fish_salmon"] == 1


def test_native_export_conversion(tmp_path: Path):
    pre = [{"data": task(), "predictions": [{"model_version": "m", "result": []}]}]
    native = [{"id": 42, "data": task()}]
    p1, p2, out = tmp_path / "pre.json", tmp_path / "native.json", tmp_path / "out.json"
    p1.write_text(json.dumps(pre)); p2.write_text(json.dumps(native))
    convert_payload(p1, p2, out)
    assert json.loads(out.read_text()) == [{"task": 42, "model_version": "m", "result": []}]
