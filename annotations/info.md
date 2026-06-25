# JSON File Descriptions

### 1. `pairs_unconsumed_(meal).json`
* **Format:** 1-to-N mapping
* **Description:** Pairs a single unconsumed image with multiple potential consumed images.

### 2. `pairs_(meal)_labelstudio.json`
* **Format:** 1-to-1 mapping
* **Description:** Flattens the 1-to-N relationships into strict 1-to-1 pairs, ready for Label Studio.

### 3. `pairs_unconsumed_(meal)_labelstudio.json`
* **Format:** Label Studio import schema
* **Description:** A flat list of unconsumed files formatted for direct import into Label Studio.

### 4. Deprecated:
- `paris_salad_old.json`
