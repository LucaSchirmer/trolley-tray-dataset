#!/usr/bin/env python3
"""Download selected Label Studio tasks as a YOLO detection/segmentation dataset."""

import argparse
import json
import os
import sys
import urllib.parse
from pathlib import Path

import requests


DEFAULT_TASK_IDS = [221, 222, 223, 224, 261, 264, 269, 277, 281, 283, 293, 303, 310, 316, 337]
IMAGE_LABEL_KEYS = ("polygonlabels", "rectanglelabels")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.getenv("LABEL_STUDIO_URL", "http://localhost:8080"))
    parser.add_argument("--project-id", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("yolo_dataset_salmon_veg_rice_15"))
    parser.add_argument("--task-ids", type=int, nargs="+", default=DEFAULT_TASK_IDS)
    return parser.parse_args()


def create_session(base_url, api_key):
    """Authenticate with either a Label Studio PAT (JWT refresh token) or legacy key."""
    session = requests.Session()
    api_key = api_key.strip()

    if api_key.count(".") == 2:
        response = session.post(
            f"{base_url}/api/token/refresh",
            json={"refresh": api_key},
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(
                "Personal access token refresh failed "
                f"(HTTP {response.status_code}): {response.text[:300]}"
            )
        session.headers["Authorization"] = f"Bearer {response.json()['access']}"
    else:
        session.headers["Authorization"] = f"Token {api_key}"

    return session


def api_get(session, url):
    response = session.get(url, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"GET {url} failed (HTTP {response.status_code}): {response.text[:300]}")
    return response


def label_for_result(result):
    value = result.get("value", {})
    for key in IMAGE_LABEL_KEYS:
        labels = value.get(key)
        if labels:
            return labels[0], key
    return None, None


def selected_results(task):
    """Use the most recent non-cancelled annotation for a task."""
    annotations = [ann for ann in task.get("annotations", []) if not ann.get("was_cancelled")]
    if not annotations:
        return []
    annotation = max(annotations, key=lambda ann: ann.get("updated_at") or ann.get("created_at") or "")
    return [result for result in annotation.get("result", []) if label_for_result(result)[0]]


def image_field_for_results(task, results):
    """Map a result's to_name (for example img_consumed) back to task data."""
    data = task.get("data", {})
    for result in results:
        to_name = result.get("to_name", "")
        candidates = [to_name]
        if to_name.startswith("img_"):
            candidates.append(to_name[4:])
        for key in candidates:
            if data.get(key):
                return key, data[key]

    for key in ("image", "consumed", "unconsumed"):
        if data.get(key):
            return key, data[key]
    raise RuntimeError(f"task {task.get('id')} has no recognizable image field")


def yolo_line(result, class_id):
    value = result["value"]
    _, kind = label_for_result(result)
    if kind == "polygonlabels":
        points = value.get("points", [])
        if len(points) < 3:
            return None
        coords = [coordinate / 100.0 for point in points for coordinate in point]
        return f"{class_id} " + " ".join(f"{coordinate:.6f}" for coordinate in coords)

    width = value.get("width", 0) / 100.0
    height = value.get("height", 0) / 100.0
    x_center = value.get("x", 0) / 100.0 + width / 2.0
    y_center = value.get("y", 0) / 100.0 + height / 2.0
    if width <= 0 or height <= 0:
        return None
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def download_image(session, base_url, source, destination):
    url = urllib.parse.urljoin(f"{base_url}/", source)
    base_origin = urllib.parse.urlsplit(base_url)[:2]
    image_origin = urllib.parse.urlsplit(url)[:2]
    # Do not leak the Label Studio credential to S3 or another external host.
    requester = session if image_origin == base_origin else requests
    response = requester.get(url, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"GET {url} failed (HTTP {response.status_code}): {response.text[:300]}")
    content_type = response.headers.get("Content-Type", "")
    if content_type and not content_type.startswith("image/"):
        raise RuntimeError(f"{url} returned {content_type}, not an image")
    destination.write_bytes(response.content)


def main():
    args = parse_args()
    base_url = args.url.rstrip("/")
    api_key = os.getenv("LABEL_STUDIO_API_KEY")
    if not api_key:
        sys.exit("Set LABEL_STUDIO_API_KEY to a Personal Access Token or legacy token first.")

    session = create_session(base_url, api_key)
    tasks = []
    failures = []
    print(f"Fetching exactly {len(args.task_ids)} selected tasks...")
    for task_id in args.task_ids:
        try:
            task = api_get(session, f"{base_url}/api/tasks/{task_id}").json()
            if task.get("project") != args.project_id:
                raise RuntimeError(f"belongs to project {task.get('project')}, not {args.project_id}")
            tasks.append((task, selected_results(task)))
        except (requests.RequestException, RuntimeError, ValueError) as error:
            failures.append(f"task {task_id}: {error}")

    if failures:
        sys.exit("Nothing was written because some tasks failed:\n  " + "\n  ".join(failures))

    class_names = sorted(
        {label_for_result(result)[0] for _, results in tasks for result in results}
    )
    class_map = {name: class_id for class_id, name in enumerate(class_names)}
    images_dir = args.output / "images"
    labels_dir = args.output / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for task, results in tasks:
        task_id = task["id"]
        field, image_source = image_field_for_results(task, results)
        source_path = urllib.parse.urlparse(image_source).path
        suffix = Path(urllib.parse.unquote(source_path)).suffix or ".jpg"
        stem = Path(urllib.parse.unquote(source_path)).stem or "image"
        filename = f"task_{task_id}_{stem}{suffix}"

        try:
            download_image(session, base_url, image_source, images_dir / filename)
        except (requests.RequestException, RuntimeError) as error:
            sys.exit(f"Image download failed for task {task_id}: {error}")

        lines = []
        for result in results:
            label, _ = label_for_result(result)
            line = yolo_line(result, class_map[label])
            if line:
                lines.append(line)
        (labels_dir / f"{Path(filename).stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        manifest.append({"task_id": task_id, "image_field": field, "file": filename})
        print(f"  task {task_id}: {filename} ({len(lines)} objects)")

    (args.output / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")
    yaml_names = "\n".join(f"  {class_id}: {json.dumps(name)}" for name, class_id in class_map.items())
    (args.output / "dataset.yaml").write_text(
        f"path: {args.output.resolve()}\ntrain: images\nval: images\nnames:\n{yaml_names}\n",
        encoding="utf-8",
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Done: exported {len(tasks)} images to {args.output.resolve()}")


if __name__ == "__main__":
    main()
