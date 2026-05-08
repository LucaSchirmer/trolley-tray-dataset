# GalleyEye — Annotation Guidelines v2

---

## 0. What You Are Annotating and Why

Each image pair produces two types of annotation:

| Annotation | Used for | How |
|---|---|---|
| Polygon masks | YOLO segmentation training | Draw on both images |
| Consumption % per component | Siamese NN ground truth | Number 0–100 per component |

These serve different models. Be precise about both. A sloppy mask hurts YOLO. A wrong percentage hurts the Siamese NN.

---

## 1. General Principles

- **Always study the left (unconsumed) image first** before drawing anything or entering any percentage.
- **One polygon per food category** — not per individual item. All rice on the tray = one polygon. Each broccoli floret is NOT its own polygon — broccoli region = one polygon.
- **Percentages come from weight data + visual correction.** Open your Excel tracker for the image pair you are annotating, check the weight-derived %, then apply visual correction where needed (see Section 6).
- **When genuinely unsure about a %, round to nearest 25.** 0, 25, 50, 75, 100 are your anchor points.
- **Consistency over precision.** The same situation annotated the same way every time is more valuable than one perfectly precise annotation surrounded by inconsistent ones.

---

## 2. Workflow — Step by Step

Follow this order for every image pair:

```
1. Open image pair in Label Studio
2. Study unconsumed image (left) — identify all food components present
3. Draw masks on UNCONSUMED image (Step 1 in interface)
4. Draw masks on CONSUMED image (Step 2 in interface)
5. Enter % for each component (Step 3 in interface)
   → Check Excel tracker weight % first
   → Apply visual correction if needed
6. Set drink binary values
7. Flag any quality issues (Step 4)
8. Add notes for anything ambiguous (Step 5)
```

Never skip steps or do them out of order. In particular: always draw masks before entering percentages — seeing the masks helps you judge the % more accurately.

---

## 3. How to Draw Polygon Masks

### 3.1 General mask rules

- Draw **tightly** around the food region — do not include empty tray space inside the polygon
- On the **unconsumed image**: draw around the full starting portion
- On the **consumed image**: draw around what is **actually there** — if food is gone, do not draw a mask for it. If food moved, draw where it is now
- If a food component is completely consumed (nothing remaining), **skip it** on the consumed image — no mask needed
- Overlap between masks is acceptable where food components touch — draw the natural boundary

### 3.2 Class-specific mask drawing rules

| Class | How to draw the mask |
|---|---|
| `rice` | One polygon covering the entire rice portion as a blob. Include the full mound. |
| `chicken` | One polygon covering all chicken pieces together as a group region. Do not outline each piece individually — draw one shape that encompasses all pieces. |
| `fish_salmon` | One polygon tightly around the fillet. |
| `broccoli` | One polygon covering the broccoli region as a group. Not each floret. |
| `carrots` | One polygon covering the carrot region as a group. |
| `salad_main` | One polygon covering the entire salad bowl/region. Includes all ingredients — do not separate cucumber, tomato, feta. |
| `wrap_half_1` | One polygon around the first half (left half when looking at the tray). |
| `wrap_half_2` | One polygon around the second half (right half). If only one half is present, only draw that one. |
| `pasta_pesto` | One polygon covering the entire pasta portion. Pesto mixed in is included. |
| `bread roll` | One polygon tightly around the bread roll. |
| `side_salad` | One polygon covering the entire side salad portion including croutons and cheese. |
| `brownie` | One polygon tightly around the brownie piece. |
| `vanilla_pudding_with_fruits` | One polygon covering the entire cup/bowl including fruits on top. |
| `fruit_salad` | One polygon covering the entire cup/bowl. |
| `water_bottle` | One polygon around the entire bottle. |
| `coffee_cup` | One polygon around the cup including handle. |
| `tea_cup` | One polygon around the cup including teabag if visible. |
| `orange_juice_bottle` | One polygon around the entire bottle. |
| `cola_can` | One polygon around the entire can. |

---

## 4. Consumption % — Per Component Rules

### 4.1 Anchor points

Always think in these anchor points first, then adjust:

| % | What it means |
|---|---|
| 0% | Completely untouched — identical to reference |
| 25% | About a quarter consumed |
| 50% | About half consumed |
| 75% | About three quarters consumed |
| 100% | Fully consumed — see per-item rules below for what counts as 100% |

### 4.2 What counts as 100% consumed — per item

This is where weight data fails. Use these visual rules to override weight-derived % when applicable:

| Component | 100% consumed when... |
|---|---|
| Rice | Less than ~5–10 scattered grains remaining — no meaningful portion left |
| Chicken | Only sauce traces — no meat pieces visible |
| Fish / Salmon | Only skin or flakes too small to eat remaining |
| Broccoli | Only stems or isolated tiny florets remaining |
| Carrots | Only small scraps or sauce-covered traces remaining |
| Salad (main) | Only dressing and isolated leaf traces at bottom of bowl |
| Wrap half 1 | The entire half is gone from the tray |
| Wrap half 2 | The entire half is gone from the tray |
| Pasta with Pesto | Less than a few strands remaining — no meaningful portion |
| Bread roll | Only crumbs remaining — no recognizable piece of roll |
| Side Salad | Only dressing and isolated leaf traces remaining |
| Brownie | Only crumbs and powder remaining — no piece with substance |
| Vanilla Pudding | Cup visibly empty or only smear on sides remaining |
| Fruit Salad | Cup visibly empty or only juice at bottom remaining |

### 4.3 Main course component rules

**Rice**
Compare visible surface area and mound height vs unconsumed reference.
- Full mound visible = 0%
- Half the area/height = 50%
- Scattered grains only = 100%

**Chicken**
Count approximate piece proportion remaining vs reference.
- Example: 4 pieces in reference, 1 remaining = 75% consumed
- If pieces are broken up — estimate total mass proportion, not piece count

**Fish / Salmon**
Compare fillet surface area remaining vs reference.
- Full fillet = 0%
- Half fillet = 50%
- Skin/flakes only = 100%

**Broccoli / Carrots**
Compare overall vegetable region coverage vs reference.
- Full portion = 0%
- Half the pieces/region remaining = 50%
- Isolated small scraps = 100%
- Note: judge broccoli and carrots independently even though they sit near each other

**Salad (main dish)**
Judge by overall bowl/plate fill level — do not count individual pieces.
- Full bowl = 0%
- Half fill level = 50%
- Only dressing and traces = 100%

**Wrap Half 1 and Half 2**
Each half is binary-like — either there or not:
- Half present and untouched = 0%
- Half partially bitten = 25–75% (judge by how much of the half is missing)
- Half completely gone = 100%

**Pasta with Pesto**
Compare volume/pile size vs reference.
- Full portion = 0%
- Half remaining = 50%
- Few strands only = 100%

---

## 5. Side Dish % Rules

**Bread roll**
Judge as a single whole unit:

| State | % |
|---|---|
| Whole roll, untouched | 0% |
| Slightly bitten | 10–20% |
| Half eaten | 50% |
| Small piece remaining | 80–90% |
| Only crumbs | 100% |

Rule: ignore crumbs — only count recognizable roll portions.

**Side Salad (Blattsalat als Beilage)**
Judge by overall volume remaining including cheese cubes and croutons:
- Do not count individual croutons or cheese cubes separately
- Judge the salad mass as a whole fill level
- Full bowl = 0%
- Half fill level = 50%
- Dressing traces only = 100%
- 

---

## 6. Dessert % Rules

**Brownie**
Judge by physical size of piece remaining:
- Whole piece = 0%, half = 50%, crumbs only = 100%
- Ignore sugar powder, sauce traces, and crumbs on tray
- Only the brownie piece itself counts

**Vanilla Pudding with Fruits**
Judge by cup fill level. Fruits and pudding together as one:
- Full cup = 0%
- Fruits picked out but pudding remains = ~20–30%
- Half consumed = 50%
- Cup empty or smear on sides only = 100%

**Fruit Salad**
Judge by cup fill level:
- Full cup = 0%
- Half remaining = 50%
- Juice only at bottom = 100%
- Do not count individual fruit pieces

---

## 7. Drink Binary Rules

A drink counts as **Consumed** when:
- It was opened

A drink counts as **Not consumed** when:
- when it is still closed and could be reused

Use **Not present** when that drink type was never on this tray.

---

## 8. Using Weight Data From Excel

Before entering any % in Label Studio:
1. Find the image filename in your Excel tracker
2. Check the auto-calculated **Actual Consumption %** column
3. Use this as your starting point
4. Apply visual correction using the rules in Section 4.2 if:
   - The weight % is >85% but non-edible residuals are clearly present → round up to 100%
   - The weight % seems inconsistent with what you see → trust your eyes and note the discrepancy in the notes field

---

## 9. Edge Cases

| Situation | What to do |
|---|---|
| Food completely gone, no mask to draw on consumed image | Skip mask for that class on consumed image. Enter 100% in slider. |
| Food moved to different part of tray | Draw mask where food actually is on consumed image. Note "food rearranged" in quality flags. |
| Wrap — only one half was ever served | Draw mask for present half only. Leave other half blank. |
| Weight % and visual judgment strongly disagree | Trust visual judgment. Note the discrepancy in the notes field with the weight value. |
| Completely uneaten tray — all 0% | Still draw all masks on both images. Enter 0% for all components. This is valid training data. |

---

## 10. Quick Reference — Class Colors

| Class | Color | Group |
|---|---|---|
| rice | Blue | Main |
| chicken | Amber | Main |
| fish_salmon | Sandy brown | Main |
| broccoli | Green | Main |
| carrots | Orange | Main |
| salad_main | Light green | Main |
| wrap_half_1 | Purple | Main |
| wrap_half_2 | Light purple | Main |
| pasta_pesto | Teal | Main |
| bread_roll | Brown | Side |
| side_salad | Yellow-green | Side |
| brownie | Dark brown | Dessert |
| vanilla_pudding_with_fruits | Yellow | Dessert |
| fruit_salad | Pink-red | Dessert |
| water_bottle | Light blue | Drink |
| coffee_cup | Dark brown | Drink |
| tea_cup | Light green | Drink |
| orange_juice_bottle | Amber | Drink |
| cola_can | Dark red | Drink |

---

## 11. Consistency Checklist

Run through before submitting every annotation:

- [ ] Did I study the unconsumed image before annotating?
- [ ] Did I draw masks on BOTH images?
- [ ] Did I draw one polygon per food category (not per individual piece)?
- [ ] Did I check the Excel weight % before entering Siamese % values?
- [ ] Did I apply visual correction for non-edible residuals (cores, bones, crumbs)?
- [ ] Are my percentages anchored to 0 / 25 / 50 / 75 / 100 where appropriate?
- [ ] Did I set all drink binary values including "Not present" for absent drinks?
- [ ] Did I flag quality issues?
- [ ] Did I note any ambiguities or visual overrides?
