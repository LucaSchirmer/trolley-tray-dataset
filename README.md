# trolley-tray-dataset

This repository contains the trolley tray image dataset organized by consumption state and meal type.

## Dataset Structure

- `images/consumed/`
- `images/unconsumed/`

Each folder is further split into the meal categories `chicken_rice`, `fish_rice`, `pasta_pesto`, `salad`, `wrap`, and `others`.

## Image Context Spreadsheet

The spreadsheet used to track the images and add context is here:

[Image tracking spreadsheet](https://docs.google.com/spreadsheets/d/1laz2lDZd-KpwE_SEZZdjidFYOjZks5Md3kDekuKRR9s/edit?usp=sharing)

## Annotation with Label Studio

### Setup

1. **Install Label Studio:**
   ```bash
   pip install label-studio
   ```

2. **Start Label Studio:**
   ```bash
   label-studio
   ```
   This opens at `http://localhost:8080`

3. **Create a new project** and import this dataset.

### Configuration

The annotation interface is defined in:
- [annotations/galleyeye_label_studio_config_v2.xml](annotations/galleyeye_label_studio_config_v2.xml)

When creating your Label Studio project:
1. Go to **Settings → Labeling Interface**
2. Paste the XML config from the file above
3. Save

### Annotation Guidelines

Detailed guidelines for annotators:
- [annotations/galleyeye_annotation_guidelines_v2.md](annotations/galleyeye_annotation_guidelines_v2.md)

Each image pair requires:
- **Polygon masks** on both unconsumed and consumed images for YOLO training
- **Consumption percentages** (0–100) per food component for Siamese NN ground truth
- **Drink binary values** (consumed / not consumed / not present)
- **Quality flags** if applicable
- **Optional notes** for edge cases

### Task Data Format

Tasks must have this JSON structure:

```json
{
  "id": 1,
  "unconsumed_url": "path/to/unconsumed/image.jpg",
  "consumed_url": "path/to/consumed/image.jpg"
}
```

The field names `unconsumed_url` and `consumed_url` must match exactly for images to load in the interface.
