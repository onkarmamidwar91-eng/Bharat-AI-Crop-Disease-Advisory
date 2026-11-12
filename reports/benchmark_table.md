# Bharat AI-SoC Benchmark Summary Tables

### Table 1: Crop Diseases (PlantVillage Dataset)
**Crop Diseases** (MobileNetV3-Small, 38 PlantVillage classes, confidence threshold 0.20)

| Measure | Result |
| :--- | :---: |
| Standard classification score (Top-1 Accuracy) | 95.53% |
| Top-5 classification score (Top-5 Accuracy) | 99.87% |
| Diseases found (any disease vs healthy) | 98.70% |
| Disease named correctly, when found (Precision) | 95.81% |
| Diseases reliable enough to name confidently (F1 ≥ 0.90) | 32 of 38 |
| Healthy leaves wrongly flagged as diseased: lab / field | 1.3% / 4.5% |

---

### Table 2: Insect Pests (IP102 Dataset Benchmark)
**Insect pests** (YOLOv8n, 102 IP102 species, confidence threshold 0.2)

| Measure | Result |
| :--- | :---: |
| Standard detection score (IP102 test, mAP50) | 56.70% |
| Insects found (any species) | 92.00% |
| Species named correctly, when found | 77.00% |
| Species reliable enough to name confidently | 34 of 102 |
| Plain leaves wrongly flagged as pests: lab / field | 0.3% / 5% |
