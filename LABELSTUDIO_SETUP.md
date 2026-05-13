# Label Studio Setup Guide

This guide explains how to set up and use your dataset with Label Studio for food consumption annotation.

## Problem: Why You Need a Local Server

Label Studio runs in your browser and needs to load images over HTTP. Simply using relative file paths like `images/consumed/...` won't work because:

1. **Browser security** — browsers can't directly access local file paths from a web page
2. **CORS headers** — Label Studio needs special `Access-Control-Allow-Origin` headers to load images from a server

## Solution: CORS-Enabled HTTP Server

The `serve_with_cors.py` script starts a local HTTP server that:
- Serves your entire dataset folder on `http://localhost:8000`
- Adds CORS headers so Label Studio can access the images
- Allows you to annotate without uploading files to an external server

## Workflow Overview

```
pairs.json (relative paths)
    ↓
flatten_pairs_to_labelstudio.py (adds absolute URLs)
    ↓
pairs_labelstudio.json (http://localhost:8000/images/...)
    ↓
serve_with_cors.py (serves with CORS headers)
    ↓
Label Studio (imports JSON, loads images from server)
```

## Step-by-Step Setup

### 1. Generate Import JSON

Convert your `pairs.json` to Label Studio format with absolute URLs:

```bash
cd c:\Users\LucaS\OneDrive\Desktop\programmieren_code\ba_thesis\trolley-tray-dataset
python scripts/flatten_pairs_to_labelstudio.py \
    --in annotations/pairs.json \
    --out annotations/pairs_labelstudio.json \
    --base-url http://localhost:8000
```

This creates `annotations/pairs_labelstudio.json` with 111 tasks (one per consumed image).

**What this does:**
- Reads each entry in `pairs.json`
- Flattens the `consumed[]` array (if one unconsumed image has 5 consumed images, creates 5 tasks)
- Converts relative paths to absolute URLs: `images/consumed/...` → `http://localhost:8000/images/consumed/...`
- Skips empty consumed entries

### 2. Start the CORS-Enabled Server

Open a terminal and run:

```bash
cd c:\Users\LucaS\OneDrive\Desktop\programmieren_code\ba_thesis\trolley-tray-dataset
python scripts/serve_with_cors.py --port 8000
```

You should see:
```
Serving CORS-enabled HTTP on port 8000
  → http://localhost:8000
Press Ctrl+C to stop
```

**Keep this terminal open** while you're working with Label Studio.

### 3. Clear Old Tasks in Label Studio

If you've already imported data with broken relative paths:

1. Open Label Studio in your browser: `http://127.0.0.1:8080`
2. Go to your project: **Projects** → **GalleryEye-Dataset** → **Labeling**
3. Click **Settings** (top right)
4. Scroll to bottom → **Dangerous zone** → **Delete all tasks**
5. Confirm deletion
6. Go back to the **Labeling** tab

### 4. Import the New JSON

1. Click **Import** (or the + icon in the task list)
2. Choose **JSON** or upload the file
3. Select `annotations/pairs_labelstudio.json`
4. Click **Import**

Label Studio will now create 111 tasks with absolute URLs pointing to your local server.

### 5. Verify Images Load

On the first task:
- You should see **two images** side-by-side
- Left: unconsumed reference image
- Right: consumed image (to annotate)
- **No red error boxes** = success ✓

If you still see errors:
- Check the terminal running `serve_with_cors.py` — it should log requests
- Make sure port 8000 is not used by another app
- Try accessing `http://localhost:8000/images/` directly in your browser

## Understanding the Data Structure

### pairs.json Format
```json
{
  "possibleElements": ["Cola", "Salad", "Bread roll", "Fruit Salad"],
  "unconsumed": "images/unconsumed/salad/all_markers_shot_20260508_175238.jpg",
  "consumed": [
    "images/consumed/salad/all_markers_shot_20260508_180041.jpg",
    "images/consumed/salad/all_markers_shot_20260508_180239.jpg",
    ...
  ]
}
```

### pairs_labelstudio.json Format (Flattened)
```json
{
  "unconsumed": "http://localhost:8000/images/unconsumed/salad/all_markers_shot_20260508_175238.jpg",
  "consumed": "http://localhost:8000/images/consumed/salad/all_markers_shot_20260508_180041.jpg",
  "possibleElements": ["Cola", "Salad", "Bread roll", "Fruit Salad"]
}
```

**Why flattening?**
- Label Studio's `<Image name="img_consumed" value="$consumed"/>` expects a single URL string
- If `consumed` is an array, Label Studio can't use it
- Flattening creates one task per consumed image, so each task has a single image to annotate

## Annotation Workflow

For each task, you will:

1. **Draw polygons** on the consumed image (right side) for each food component
   - Use the **Polygon** tool from the left panel
   - Select the component label (Cola, Salad, etc.) from the dropdown
   - Trace the region on the image

2. **Enter consumption percentages** (Step 3 section)
   - For each component, enter 0-100 for how much was eaten
   - 0 = untouched, 100 = fully consumed
   - Leave blank if component not present

3. **Mark drink consumption** (binary choices)
   - Select whether water, coffee, tea, orange juice, cola were consumed
   - "Consumed" = bottle/can visibly empty or removed

4. **Click Submit** when done

## Troubleshooting

### Images still show error boxes with relative paths

**Problem:** Old tasks are still loaded  
**Solution:** Delete all tasks (Settings → Dangerous zone → Delete all tasks), then re-import

### Port 8000 already in use

**Problem:** Another app is using port 8000  
**Solution:** Use a different port:
```bash
python scripts/serve_with_cors.py --port 9000
```
Then re-generate the import JSON with `--base-url http://localhost:9000`

### "Cannot reach http://localhost:8000"

**Problem:** Server crashed or wasn't started  
**Solution:**
- Check that the terminal running `serve_with_cors.py` is still open
- Restart it with: `python scripts/serve_with_cors.py --port 8000`
- Test by visiting `http://localhost:8000` in your browser — you should see a file listing

### Slow image loading

**Problem:** Network is slow  
**Solution:**
- The server is on your local machine, so it should be instant
- Check CPU/memory usage on your computer
- If many tasks are loaded, Label Studio may be slow — try scrolling slowly

## Advanced: Uploading to Remote Server

If you want to annotate from another machine:

1. Host your images on a web server (e.g., AWS S3, nginx, Apache)
2. Update the base URL when flattening:
   ```bash
   python scripts/flatten_pairs_to_labelstudio.py \
       --in annotations/pairs.json \
       --out annotations/pairs_labelstudio.json \
       --base-url https://your-domain.com/datasets/trolley-tray
   ```
3. Make sure the server sends CORS headers (or ask your DevOps team)

## Summary

| Step | Command | Purpose |
|------|---------|---------|
| 1 | `python scripts/flatten_pairs_to_labelstudio.py ...` | Convert pairs.json to Label Studio format with absolute URLs |
| 2 | `python scripts/serve_with_cors.py --port 8000` | Start local HTTP server with CORS headers |
| 3 | Label Studio UI | Delete old broken tasks |
| 4 | Label Studio UI | Import `pairs_labelstudio.json` |
| 5 | Label Studio UI | Annotate tasks |

Keep the server running in step 2 while you work. Happy annotating! 🎯
