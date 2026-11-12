import os
import sys
import json
import time
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

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
    print("🌿 Testing Field-Tuned Model on PlantDoc (Real Field Conditions)")
    print("📂 Source: https://github.com/pratikkayal/PlantDoc-Dataset")
    print("=" * 65)

    base_dir = r"D:\Bharat-AI-Crop-Disease-Advisory"
    model_path = os.path.join(base_dir, "models", "crop_disease_mobilenetv3_field_tuned.pth")
    classes_path = os.path.join(base_dir, "models", "classes.json")
    reports_dir = os.path.join(base_dir, "reports")

    with open(classes_path, "r") as f:
        classes = json.load(f)
    class_to_idx = {name: i for i, name in enumerate(classes)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(os.cpu_count() or 4)
    print(f"Loading MobileNetV3-Large (Field-Tuned) on {device} (threads={os.cpu_count()})...")

    model = models.mobilenet_v3_large(weights=None)
    in_feat = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_feat, len(classes))
    )
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

    split_results = {}
    overall_total = 0
    overall_top1 = 0
    overall_top5 = 0
    overall_crop = 0
    overall_health = 0

    per_class_stats = {}
    start_time = time.time()

    for split in splits:
        split_dir = os.path.join(plantdoc_root, split)
        if not os.path.exists(split_dir):
            continue

        print(f"\nEvaluating split: '{split}'...")
        split_total = 0
        split_top1 = 0
        split_top5 = 0
        split_crop = 0
        split_health = 0

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

                is_top1 = (top1_idx == target_idx)
                is_top5 = (target_idx in top5_indices)
                is_crop_ok = (pred_crop in target_crop or target_crop in pred_crop)
                is_health_ok = (pred_is_healthy == target_is_healthy)

                if is_top1:
                    split_top1 += 1
                if is_top5:
                    split_top5 += 1
                if is_crop_ok:
                    split_crop += 1
                if is_health_ok:
                    split_health += 1

                split_total += 1
                per_class_stats[folder_name]["total"] += 1
                if is_top1:
                    per_class_stats[folder_name]["top1"] += 1
                if is_top5:
                    per_class_stats[folder_name]["top5"] += 1
                if is_crop_ok:
                    per_class_stats[folder_name]["crop_ok"] += 1

        acc1 = (split_top1 / split_total) * 100 if split_total else 0.0
        acc5 = (split_top5 / split_total) * 100 if split_total else 0.0
        split_results[split] = {
            "total": split_total,
            "top1_acc": round(acc1, 2),
            "top5_acc": round(acc5, 2),
            "crop_acc": round((split_crop / split_total) * 100, 2) if split_total else 0,
            "health_acc": round((split_health / split_total) * 100, 2) if split_total else 0
        }
        print(f"  Split '{split}' ({split_total} images): Top-1 = {acc1:.2f}% | Top-5 = {acc5:.2f}%")

        overall_total += split_total
        overall_top1 += split_top1
        overall_top5 += split_top5
        overall_crop += split_crop
        overall_health += split_health

    elapsed = time.time() - start_time
    ov_acc1 = (overall_top1 / overall_total) * 100 if overall_total else 0.0
    ov_acc5 = (overall_top5 / overall_total) * 100 if overall_total else 0.0
    ov_crop = (overall_crop / overall_total) * 100 if overall_total else 0.0
    ov_health = (overall_health / overall_total) * 100 if overall_total else 0.0

    print("\n" + "=" * 65)
    print("🏆 IMPROVED FIELD-TUNED MODEL RESULTS ON PLANTDOC")
    print("=" * 65)
    print(f"Total Images Evaluated               : {overall_total}")
    print(f"Held-out Test Split Top-1 Acc        : {split_results.get('test', {}).get('top1_acc', 0)}% (Top-5: {split_results.get('test', {}).get('top5_acc', 0)}%)")
    print(f"Overall Dataset Top-1 Acc            : {ov_acc1:.2f}%")
    print(f"Overall Dataset Top-5 Acc            : {ov_acc5:.2f}%")
    print(f"Crop Species Recognition Acc         : {ov_crop:.2f}%")
    print(f"Healthy vs. Diseased Binary Acc      : {ov_health:.2f}%")
    print(f"Evaluation Time                      : {elapsed:.1f}s ({elapsed/overall_total*1000:.1f} ms/image)")
    print("=" * 65)

    final_report = {
        "model": "MobileNetV3-Large (Field-Tuned)",
        "splits": split_results,
        "overall_total_images": overall_total,
        "overall_top1_acc": round(ov_acc1, 2),
        "overall_top5_acc": round(ov_acc5, 2),
        "overall_crop_acc": round(ov_crop, 2),
        "overall_health_acc": round(ov_health, 2),
        "per_class_stats": per_class_stats
    }

    report_path = os.path.join(reports_dir, "plantdoc_field_tuned_evaluation.json")
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2)
    print(f"Report saved to: {report_path}")

if __name__ == "__main__":
    evaluate_plantdoc()
