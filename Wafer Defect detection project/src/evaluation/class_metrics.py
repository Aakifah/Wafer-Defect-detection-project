"""Computes class-wise F1-scores and Mean Average Precision (mAP) for the quantized PyTorch model."""

import json
import os
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, classification_report
from torch.utils.data import DataLoader

from src.ETL_data.loader import get_dataloaders


def evaluate_class_performance(
    model_path: str, dataloader: DataLoader[Any], class_names: list[str]  
) -> dict[str, Any]:
    """Runs data batches through a PyTorch model to evaluate detailed class metrics."""
    model = torch.jit.load(model_path)
    model.eval()

    all_targets = []
    all_probs = []
    all_preds = []

    for images, labels in dataloader:
        with torch.no_grad():
            logits = model(images)
            
        logits_np = logits.cpu().numpy()

        probs = 1 / (1 + np.exp(-logits_np))

        if len(labels.shape) > 1 and labels.shape[1] > 1:
            target_labels = labels.numpy()
        else:
            target_labels = labels.numpy()

        preds = (probs >= 0.5).astype(int)

        all_targets.append(target_labels)
        all_probs.append(probs)
        all_preds.append(preds)

    targets = np.concatenate(all_targets)
    probs   = np.concatenate(all_probs)
    preds   = np.concatenate(all_preds)

    num_classes = len(class_names)

    print(f"[DEBUG] Model output shape  : {probs.shape}  ->  {probs.shape[1]} logits")
    print(f"[DEBUG] Unique target labels: {sorted(np.unique(targets).tolist())}")
    print(f"[DEBUG] Expected classes    : {num_classes}  ({class_names})")

    if probs.shape[1] != num_classes:
        probs = probs[:, :num_classes]
        preds = (probs >= 0.5).astype(int)

    report = classification_report(
        targets,
        preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    print("\n --- PRODUCTION QUANTIZED ONNX REPORT ---")
    print(classification_report(
        targets, preds, target_names=class_names, zero_division=0
    ))

    class_ap: dict[str, float] = {}
    total_ap = 0.0

    for i, class_name in enumerate(class_names):
        ap = average_precision_score(targets[:, i], probs[:, i])
        class_ap[f"{class_name}_AP"] = float(ap)
        total_ap += ap

    mAP = total_ap / num_classes
    print(f"Mean Average Precision (mAP): {mAP:.4f}\n")

    metrics_summary = {
        "class_wise_f1": {c: report[c]["f1-score"] for c in class_names},
        "class_wise_ap": class_ap,
        "mean_average_precision_mAP": float(mAP),
        "macro_avg_f1": report["macro avg"]["f1-score"],
        "weighted_avg_f1": report["weighted avg"]["f1-score"],
    }

    return metrics_summary


if __name__ == "__main__":
    DEFECT_CLASSES = [
    "Center",    
    "Donut",     
    "Edge_Loc",  
    "Edge_Ring", 
    "Loc",       
    "Near_Full", 
    "Scratch",   
    "Random",    
]

    MODEL_PATH = "evaluation/quantized_cnn/model.pt"

    print("Connecting to S3 endpoints to stream dataset components...")

    _, val_dataloader, _ = get_dataloaders(batch_size=32)

    metrics_summary = evaluate_class_performance(
        model_path=MODEL_PATH,
        dataloader=val_dataloader,
        class_names=DEFECT_CLASSES,
    )

    output_path = "evaluation/quantized_cnn/class_metrics.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)

    print(f"Class performance metrics saved cleanly to {output_path}")