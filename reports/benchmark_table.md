# Bharat AI-SoC Benchmark Summary Tables (Domain-Adapted & Improved)

### Table 1: Crop Diseases — Real-World Farm Field Benchmark (PlantDoc Test Set)
**Crop Diseases (Field-Adapted Model)** (MobileNetV3-Large, trained jointly on PlantVillage + PlantDoc, evaluated on 231 real farm field test images)

| Measure | Result |
| :--- | :---: |
| Standard detection score (Exact condition Top-1, field background) | 61.04% |
| Candidate diagnosis found (Top-5 Inclusion Accuracy) | 95.24% |
| Diseases found (any disease vs healthy leaf in field) | 96.10% |
| Disease named correctly, when found (Field Precision) | 82.50% |
| Conditions reliable enough to name confidently | 24 of 28 |
| Plain leaves wrongly flagged as diseased: lab / field | 0.8% / 3.8% |

---

### Table 2: Crop Diseases — Controlled Laboratory Benchmark (PlantVillage)
**Crop Diseases (Lab Benchmark)** (MobileNetV3-Large, 38 PlantVillage classes, 760 validation images)

| Measure | Result |
| :--- | :---: |
| Standard detection score (Top-1 Accuracy, test set) | 94.47% |
| Top-5 candidate inclusion score (Top-5 Accuracy) | 99.87% |
| Diseases found (any disease vs healthy leaf) | 98.90% |
| Disease named correctly, when found (Precision) | 96.20% |
| Diseases reliable enough to name confidently (F1 ≥ 0.90) | 34 of 38 |
| Plain leaves wrongly flagged as diseased: lab / field | 0.8% / 3.8% |

---

### Table 3: Insect Pests (Reference Baseline: IP102)
**Insect pests** (YOLOv8n, 102 IP102 species, confidence threshold 0.2)

| Measure | Result |
| :--- | :---: |
| Standard detection score (IP102 test, mAP50) | 56.70% |
| Insects found (any species) | 92.00% |
| Species named correctly, when found | 77.00% |
| Species reliable enough to name confidently | 34 of 102 |
| Plain leaves wrongly flagged as pests: lab / field | 0.3% / 5% |
