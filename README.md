# trolley-tray-dataset

This repository contains the trolley tray image dataset organized by consumption state and meal type.

## Dataset Structure

- `images_cropped/consumed/`
- `images_cropped/unconsumed/`

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

Use two Label Studio projects:
- [annotations/galleyeye_label_studio_config_unconsumed_v2.xml](annotations/galleyeye_label_studio_config_unconsumed_v2.xml) for unconsumed reference masks
- [annotations/galleyeye_label_studio_config_v2.xml](annotations/galleyeye_label_studio_config_v2.xml) for consumed comparison masks and consumption estimates

When creating your Label Studio project:
1. Go to **Settings → Labeling Interface**
2. Paste the XML config for the project type you want
3. Save

### Annotation Guidelines

Detailed guidelines for annotators:
- [annotations/galleyeye_annotation_guidelines_v2.md](annotations/galleyeye_annotation_guidelines_v2.md)

Each image pair requires:
- **Unconsumed reference masks** once per image for YOLO training
- **Consumed comparison masks** once per consumed image for YOLO training
- **Consumption percentages** (0–100) per food component for Siamese NN ground truth
- **Drink binary values** (consumed / not consumed / not present)
- **Quality flags** if applicable
- **Optional notes** for edge cases

### Task Data Format

Unconsumed-reference tasks use this JSON structure:

```json
{
  "id": 1,
   "unconsumed": "path/to/unconsumed/image.jpg",
   "possibleElements": ["Rice", "Chicken"]
}
```

Consumed-comparison tasks use `annotations/pairs_labelstudio.json` with the `unconsumed` and `consumed` fields.
