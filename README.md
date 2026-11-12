# 🌿 Edge-AgriAI: Offline Crop Disease & Pest Detection with On-Device Advisory

[![Bharat AI-SoC Challenge](https://img.shields.io/badge/Competition-Bharat%20AI--SoC%202026--27-brightgreen)](https://arm-education.github.io/Arm-Developer-Labs/Bharat_AI_SoC_2026_27.html)
[![Target Platform](https://img.shields.io/badge/Hardware-Arm%20SoC%20%7C%20Raspberry%20Pi%205%20%7C%20Android-blue)](#hardware-software-co-design)
[![Runtime](https://img.shields.io/badge/Runtime-ExecuTorch%20%7C%20HailoRT%20%7C%20KleidiAI-orange)](#system-architecture)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, privacy-preserving, edge-native AI system developed for **Bharat AI-SoC Challenge (MeitY C2S, Arm, and IIT Delhi)** under **PS1: Agriculture & Food Security**.

The system operates **100% offline** without internet connectivity, running on Arm-based edge platforms (Raspberry Pi 5, Hailo AI HAT+, and Android smartphones). It diagnoses crop diseases across 38 classes, calculates severity ratings, and synthesizes localized, ICAR-aligned recovery advisories via **Llama 3.2** deployed using **ExecuTorch**.

---

## 📌 Problem Statement & Objective

Rural farming communities in India frequently face crop losses exceeding 30–40% due to delayed diagnosis of foliar diseases and pest infestations. Existing digital farming applications rely on cloud APIs, rendering them ineffective in rural areas with poor connectivity.

**Core Objective:**
Deliver a sub-2-second, completely offline diagnostic and multilingual agronomy advisory kiosk/handheld tool running on Arm SoCs.

---

## 🏗️ Hardware-Software Co-Design

To maximize silicon efficiency and avoid thermal throttling on edge SoCs, we decouple the vision perception pipeline from the language reasoning pipeline:

```
[ Camera / Farmer Photo ]
          │
          ▼
[ Vision Perception Pipeline ]
MobileNetV3-Small (INT8 Quantized) / YOLOv8
Runs on: Hailo NPU via HailoRT / Arm Cortex-A76 CPU
Latency: ~15ms (NPU) / ~65ms (CPU)
          │
          ▼  Outputs: { Crop, Condition, Confidence, Severity % }
[ Structured Context Engine ]
Injects: Verified ICAR / KVK Agronomic Packages of Practices (Zero Hallucination)
          │
          ▼
[ Reasoning & Advisory Pipeline ]
Llama 3.2 1B-Instruct (4-bit Quantized via SpinQuant / Q4_0)
Runs on: Arm Cortex-A76 CPU via ExecuTorch + Arm KleidiAI Micro-kernels
          │
          ▼
[ Offline Audio & Visual Interface ]
Sherpa-ONNX Speech-to-Text & Piper TTS for Vernacular Voice Output (Hindi, Tamil, etc.)
```

---

## 🔬 Dataset & Data Augmentation Pipeline

Trained and evaluated on the **PlantVillage** dataset spanning **38 classes** (including Tomato, Potato, Corn, Apple, Grape, Peach, Bell Pepper, etc.).

### Robust Augmentation Strategy
Field photographs taken by farmers have varying angles, harsh midday glare, motion blur, and uneven framing. To maximize field generalization, we implemented:
* **RandomResizedCrop (224x224, scale 0.8–1.0)**: Invariant to leaf distance and framing.
* **RandomHorizontalFlip & RandomRotation (25°)**: Invariant to camera and leaf tilt.
* **ColorJitter (Brightness, Contrast, Saturation, Hue)**: Invariant to direct sunlight vs. cloudy conditions.
* **GaussianBlur (3x3 kernel)**: Invariant to hand tremor and lens defocus.
* **Standard Normalization**: Zero-mean unit-variance matching ImageNet distributions.

---

## 📊 Accuracy & Performance Benchmarks

* **Model Architecture**: MobileNetV3-Small (Edge-Optimized for Arm Cortex-A76)
* **Model Size**: ~9.5 MB (.pth weights) / ~2.6 MB (INT8 Quantized)
* **Dataset**: PlantVillage (Color Dataset, 38 Classes)
* **Validation Top-1 Accuracy**: **95.53%**
* **Validation Top-5 Accuracy**: **99.87%**
* **Macro Precision**: **95.81%**
* **Macro Recall**: **95.53%**
* **Macro F1-Score**: **95.49%**
* **Weighted F1-Score**: **95.49%**
* **Training Time**: 9.48 minutes on 16 CPU threads
* **Edge Inference Latency**: ~65ms on Arm Cortex-A76 / ~15ms on Hailo-8 NPU

*(Detailed confusion matrix and class-by-class F1-scores available in `reports/accuracy_report.json` and `reports/classification_report.txt`)*

---

## 🚀 Quickstart & Usage

### 1. Installation
```bash
git clone https://github.com/onkarmamidwar/Bharat-AI-Crop-Disease-Advisory.git
cd Bharat-AI-Crop-Disease-Advisory
pip install -r requirements.txt
```

### 2. Run Inference on a Leaf Image
```bash
python src/inference.py --image path/to/leaf.jpg
```

### 3. Generate Offline Llama 3.2 Advisory via ExecuTorch
```bash
python src/llama_executorch_advisory.py
```

### 4. Retrain / Fine-tune with Custom Crops
```bash
python src/train.py
```

---

## 🏆 Bharat AI-SoC Hackathon Compliance Checklist

- [x] **PS1 Agriculture Theme**: Offline crop disease and pest identification with advisory.
- [x] **ExecuTorch Runtime**: Pipeline structured for reference ExecuTorch deployment with Arm KleidiAI.
- [x] **Llama Family Model**: Reasoning engine prompts Llama 3.2 1B-Instruct offline.
- [x] **Data Augmentation**: Heavy field-invariant transformations to increase real-world accuracy.
- [x] **Reproducible Codebase & Reports**: Comprehensive classification metrics, models, and docs.
