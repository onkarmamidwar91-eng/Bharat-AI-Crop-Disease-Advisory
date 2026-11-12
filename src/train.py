import os
import sys
import time
import json
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

from dataset import prepare_dataloaders
from model import build_edge_model

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

        if (i + 1) % 15 == 0 or (i + 1) == len(dataloader):
            batch_acc = correct / total
            print(f"  Batch [{i+1}/{len(dataloader)}] - Loss: {running_loss/total:.4f} - Acc: {batch_acc*100:.2f}%", flush=True)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())

            # Top-5 accuracy
            _, top5_preds = outputs.topk(min(5, outputs.size(1)), dim=1, largest=True, sorted=True)
            top5_correct += top5_preds.eq(labels.view(-1, 1).expand_as(top5_preds)).sum().item()
            total += labels.size(0)

    val_loss = running_loss / total
    top1_acc = accuracy_score(all_targets, all_preds)
    top5_acc = top5_correct / total

    return val_loss, top1_acc, top5_acc, all_preds, all_targets

def main():
    print("=" * 65)
    print("🌿 Bharat AI-SoC: Offline Crop Disease Detection & Advisory")
    print("🚀 Edge Training on PlantVillage with Data Augmentations")
    print("=" * 65, flush=True)

    # Performance configuration for multi-core Arm/x86 host
    num_cores = os.cpu_count() or 4
    torch.set_num_threads(num_cores)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Compute Device: {device} | CPU Worker Threads: {num_cores}")

    dataset_path = r"D:\plantvillage dataset\color"
    output_dir = r"D:\Bharat-AI-Crop-Disease-Advisory\models"
    reports_dir = r"D:\Bharat-AI-Crop-Disease-Advisory\reports"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # Prepare stratified datasets with augmentations
    print("\n📂 Loading and preparing augmented dataset...")
    # Using 80 samples per class for training and 20 for validation
    train_loader, val_loader, classes, class_to_idx = prepare_dataloaders(
        dataset_dir=dataset_path,
        max_samples_per_class=100,
        train_ratio=0.8,
        batch_size=32,
        seed=42
    )

    print(f"✔ Classes: {len(classes)}")
    print(f"✔ Training samples: {len(train_loader.dataset)}")
    print(f"✔ Validation samples: {len(val_loader.dataset)}")

    # Save classes
    with open(os.path.join(output_dir, "classes.json"), "w") as f:
        json.dump(classes, f, indent=2)

    # Build model
    model = build_edge_model(num_classes=len(classes), pretrained=True)
    model = model.to(device)

    # Optimization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)

    epochs = 4
    best_acc = 0.0
    history = []

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch [{epoch}/{epochs}] (LR: {scheduler.get_last_lr()[0]:.6f}) ---")
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, top1_acc, top5_acc, val_preds, val_targets = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch} finished in {elapsed:.1f}s")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Top-1 Acc: {top1_acc*100:.2f}% | Val Top-5 Acc: {top5_acc*100:.2f}%")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_top1_acc": top1_acc,
            "val_top5_acc": top5_acc,
            "duration_sec": elapsed
        })

        if top1_acc > best_acc:
            best_acc = top1_acc
            best_model_path = os.path.join(output_dir, "crop_disease_mobilenetv3.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"  ⭐ New best model saved to {best_model_path} ({best_acc*100:.2f}%)")

    total_training_time = time.time() - start_time
    print("\n" + "=" * 65)
    print(f"✅ Training completed in {total_training_time/60:.2f} minutes!")
    print(f"🏆 Best Validation Accuracy: {best_acc*100:.2f}%")
    print("=" * 65)

    # Load best model for full evaluation report
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    _, final_top1, final_top5, final_preds, final_targets = evaluate(model, val_loader, criterion, device)

    # Compute detailed metrics
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        final_targets, final_preds, average='macro', zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        final_targets, final_preds, average='weighted', zero_division=0
    )

    report_text = classification_report(
        final_targets, final_preds, target_names=classes, zero_division=0
    )

    with open(os.path.join(reports_dir, "classification_report.txt"), "w") as f:
        f.write(report_text)

    # Export accuracy summary json
    accuracy_summary = {
        "model_architecture": "MobileNetV3-Small (Edge Optimized)",
        "target_platform": "Arm Cortex-A76 / Raspberry Pi 5 / Android",
        "dataset": "PlantVillage (Augmented Color Dataset)",
        "total_classes": len(classes),
        "total_training_samples": len(train_loader.dataset),
        "total_validation_samples": len(val_loader.dataset),
        "training_time_minutes": round(total_training_time / 60, 2),
        "best_validation_top1_accuracy": round(final_top1 * 100, 2),
        "best_validation_top5_accuracy": round(final_top5 * 100, 2),
        "precision_macro": round(prec_macro * 100, 2),
        "recall_macro": round(rec_macro * 100, 2),
        "f1_score_macro": round(f1_macro * 100, 2),
        "precision_weighted": round(prec_weighted * 100, 2),
        "recall_weighted": round(rec_weighted * 100, 2),
        "f1_score_weighted": round(f1_weighted * 100, 2),
        "history": history
    }

    with open(os.path.join(reports_dir, "accuracy_report.json"), "w") as f:
        json.dump(accuracy_summary, f, indent=2)

    print("\n📊 Classification Report Highlights:")
    print(f"• Top-1 Accuracy:   {accuracy_summary['best_validation_top1_accuracy']}%")
    print(f"• Top-5 Accuracy:   {accuracy_summary['best_validation_top5_accuracy']}%")
    print(f"• Macro F1-Score:   {accuracy_summary['f1_score_macro']}%")
    print(f"• Weighted Precision: {accuracy_summary['precision_weighted']}%")
    print(f"• Report saved to:  {os.path.join(reports_dir, 'accuracy_report.json')}")

if __name__ == "__main__":
    main()
