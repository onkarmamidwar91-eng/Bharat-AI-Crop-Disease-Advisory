# Bharat AI-SoC Benchmark Summary Tables

### Table 1: Crop Diseases — Controlled Laboratory Benchmark (PlantVillage)
**Crop Diseases (Lab Benchmark)** (MobileNetV3-Small, 38 PlantVillage classes, confidence threshold 0.20)

| Measure | Result |
| :--- | :---: |
| Standard detection score (Top-1 Accuracy, test set) | 95.53% |
| Top-5 candidate inclusion score (Top-5 Accuracy) | 99.87% |
| Diseases found (any disease vs healthy leaf) | 98.70% |
| Disease named correctly, when found (Precision) | 95.81% |
| Diseases reliable enough to name confidently (F1 ≥ 0.90) | 32 of 38 |
| Plain leaves wrongly flagged as diseased: lab / field | 1.3% / 4.5% |

---

### Table 2: Crop Diseases — Real-World Farm Field Benchmark (PlantDoc Dataset)
**Crop Diseases (Real Field Stress Test)** (Evaluated across all 2,485 images from https://github.com/pratikkayal/PlantDoc-Dataset)

| Measure | Result |
| :--- | :---: |
| Standard detection score (Exact condition Top-1, field) | 17.02% |
| Candidate diagnosis found (Top-5 Inclusion Accuracy) | 47.81% |
| Healthy vs. Infected leaves correctly distinguished | 76.10% |
| Crop species correctly identified in wild background | 37.30% |
| Real field images evaluated (test + train splits) | 2,485 |
| On-device inference latency per field image (Arm CPU) | 46.5 ms |

---

### Table 3: Insect Pests (IP102 Benchmark)
**Insect pests** (YOLOv8n, 102 IP102 species, confidence threshold 0.2)

| Measure | Result |
| :--- | :---: |
| Standard detection score (IP102 test, mAP50) | 56.70% |
| Insects found (any species) | 92.00% |
| Species named correctly, when found | 77.00% |
| Species reliable enough to name confidently | 34 of 102 |
| Plain leaves wrongly flagged as pests: lab / field | 0.3% / 5% |
