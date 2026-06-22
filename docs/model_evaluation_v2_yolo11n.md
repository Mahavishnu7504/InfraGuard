## Validation Metrics

| Metric | Value |
|--------|-------|
| Precision | 0.789 |
| Recall | 0.533 |
| mAP50 | 0.605 |
| mAP50-95 | 0.319 |

---

## Observations
- Strong helmet and person detection.
- Moderate recall indicates missed detections in some PPE categories.
- Boots and gloves performance requires dataset balancing or longer training.
- Model stable on CPU with 416 image size.

---

## Production Decision

Status: Approved for Production (Baseline v2)