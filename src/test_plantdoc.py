import os
import sys
import json
import time
from PIL import Image
import torch
from torchvision import transforms

# Ensure project modules are importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import build_edge_model

PLANTDOC_TO_PLANTVILLAGE = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",
}

def get_crop_from_class(cls_name):
    return cls_name.split("___")[0].replace("_", " ").strip().lower()

def is_healthy(cls_name):
    return "healthy" in cls_name.lower()

def evaluate_plantdoc():
    print("=" * 65)
    print("🌿 Testing Trained Edge Model on Real Field Dataset: PlantDoc")
    print("📂 Dataset source: https://github.com/pratikkayal/PlantDoc-Dataset")
    print("=" * 65)

    base_dir = r"D:\Bharat-AI-Crop-Disease-Advisory"
    model_path = os.path.join(base_dir, "models", "crop_disease_mobilenetv3.pth")
    classes_path = os.path.join(base_dir, "models", "classes.json")
    reports_dir = os.path.join(base_dir, "reports")

    with open(classes_path, "r") as f:
        classes = json.load(f)
    class_to_idx = {name: i for i, name in enumerate(classes)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(os.cpu_count() or 4)
    print(f"Loading MobileNetV3-Small on {device} (threads={os.cpu_count()})...")

    model = build_edge_model(num_classes=len(classes), pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    plantdoc_root = r"D:\PlantDoc-Dataset"
    splits = ["test", "train"]

    total_images = 0
    corrupt_images = 0
    top1_correct = 0
    top5_correct = 0
    crop_correct = 0
    health_correct = 0

    per_class_stats = {}

    start_time = time.time()

    for split in splits:
        split_dir = os.path.join(plantdoc_root, split)
        if not os.path.exists(split_dir):
            continue

        print(f"\nEvaluating split: '{split}'...")
        for folder_name in sorted(os.listdir(split_dir)):
            folder_path = os.path.join(split_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            pv_class = PLANTDOC_TO_PLANTVILLAGE.get(folder_name)
            if not pv_class or pv_class not in class_to_idx:
                continue

            target_idx = class_to_idx[pv_class]
            target_crop = get_crop_from_class(pv_class)
            target_is_healthy = is_healthy(pv_class)

            if folder_name not in per_class_stats:
                per_class_stats[folder_name] = {"total": 0, "top1": 0, "top5": 0, "crop_ok": 0}

            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for img_name in image_files:
                img_path = os.path.join(folder_path, img_name)
                try:
                    with Image.open(img_path) as img:
                        img = img.convert("RGB")
                        tensor = val_transform(img).unsqueeze(0).to(device)
                except Exception:
                    corrupt_images += 1
                    continue

                with torch.no_grad():
                    output = model(tensor)
                    _, pred_top1 = torch.max(output, 1)
                    _, pred_top5 = torch.topk(output, min(5, output.size(1)), dim=1)

                top1_idx = pred_top1.item()
                top5_indices = pred_top5[0].cpu().numpy().tolist()

                pred_class = classes[top1_idx]
                pred_crop = get_crop_from_class(pred_class)
                pred_is_healthy = is_healthy(pred_class)

                # Metrics
                is_top1 = (top1_idx == target_idx)
                is_top5 = (target_idx in top5_indices)
                is_crop_ok = (pred_crop in target_crop or target_crop in pred_crop)
                is_health_ok = (pred_is_healthy == target_is_healthy)

                if is_top1:
                    top1_correct += 1
                if is_top5:
                    top5_correct += 1
                if is_crop_ok:
                    crop_correct += 1
                if is_health_ok:
                    health_correct += 1

                total_images += 1
                per_class_stats[folder_name]["total"] += 1
                if is_top1:
                    per_class_stats[folder_name]["top1"] += 1
                if is_top5:
                    per_class_stats[folder_name]["top5"] += 1
                if is_crop_ok:
                    per_class_stats[folder_name]["crop_ok"] += 1

                if total_images % 300 == 0:
                    print(f"  Processed {total_images} images | Current Top-1: {top1_correct/total_images*100:.2f}% | Top-5: {top5_correct/total_images*100:.2f}%", flush=True)

    elapsed = time.time() - start_time

    top1_acc = (top1_correct / total_images) * 100 if total_images else 0.0
    top5_acc = (top5_correct / total_images) * 100 if total_images else 0.0
    crop_acc = (crop_correct / total_images) * 100 if total_images else 0.0
    health_acc = (health_correct / total_images) * 100 if total_images else 0.0

    print("\n" + "=" * 65)
    print("🏆 FINAL PLANTDOC EVALUATION BENCHMARK RESULTS")
    print("=" * 65)
    print(f"Total Real Field Images Evaluated : {total_images}")
    print(f"Top-1 Exact Diagnosis Accuracy    : {top1_acc:.2f}%")
    print(f"Top-5 Candidate Diagnosis Accuracy: {top5_acc:.2f}%")
    print(f"Crop Species Recognition Accuracy : {crop_acc:.2f}%")
    print(f"Healthy vs. Infected Binary Acc   : {health_acc:.2f}%")
    print(f"Total Evaluation Time             : {elapsed:.1f}s ({elapsed/total_images*1000:.1f} ms/image)")
    print("=" * 65)

    # Save machine-readable report
    results = {
        "benchmark_dataset": "PlantDoc (Real Field Conditions)",
        "source_url": "https://github.com/pratikkayal/PlantDoc-Dataset",
        "model_architecture": "MobileNetV3-Small (Edge)",
        "weights_path": "models/crop_disease_mobilenetv3.pth",
        "total_images_evaluated": total_images,
        "corrupt_images_skipped": corrupt_images,
        "top1_exact_accuracy_pct": round(top1_acc, 2),
        "top5_candidate_accuracy_pct": round(top5_acc, 2),
        "crop_species_accuracy_pct": round(crop_acc, 2),
        "health_vs_disease_binary_accuracy_pct": round(health_acc, 2),
        "evaluation_duration_seconds": round(elapsed, 2),
        "latency_ms_per_image": round((elapsed / total_images) * 1000, 2) if total_images else 0,
        "per_class_breakdown": per_class_stats
    }

    report_path = os.path.join(reports_dir, "plantdoc_accuracy_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✔ Full benchmark report saved to: {report_path}")

if __name__ == "__main__":
    evaluate_plantdoc()
