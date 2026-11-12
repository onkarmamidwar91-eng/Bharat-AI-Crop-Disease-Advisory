import os
import json
import torch
from PIL import Image
from torchvision import transforms
from model import build_edge_model

def load_inference_pipeline(model_path=r"D:\Bharat-AI-Crop-Disease-Advisory\models\crop_disease_mobilenetv3.pth",
                            classes_path=r"D:\Bharat-AI-Crop-Disease-Advisory\models\classes.json"):
    with open(classes_path, "r") as f:
        classes = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_edge_model(num_classes=len(classes), pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return model, classes, val_transform, device

def predict_crop_disease(image_path, model, classes, transform, device):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top_prob, top_idx = torch.topk(probs, 3)

    results = []
    for prob, idx in zip(top_prob, top_idx):
        raw_name = classes[idx.item()]
        # Parse crop and disease name
        parts = raw_name.split("___")
        crop = parts[0].replace("_", " ")
        disease = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
        results.append({
            "class_raw": raw_name,
            "crop": crop,
            "condition": disease,
            "confidence": float(round(prob.item() * 100, 2))
        })

    return results

if __name__ == "__main__":
    model, classes, transform, device = load_inference_pipeline()
    print("Edge inference pipeline successfully initialized!")
