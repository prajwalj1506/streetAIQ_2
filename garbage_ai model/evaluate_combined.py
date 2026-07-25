import os
import torch
from ultralytics import YOLO

def evaluate_models(
    original_weights="garbage_ai model/weights/garbage_detector.pt",
    combined_weights="garbage_ai model/weights/combined_model.pt",
    data_yaml="garbage_ai model/combined_data.yaml"
):
    print("=" * 60)
    print("Evaluating Models on Validation/Test Dataset...")
    print("=" * 60)

    device = "0" if torch.cuda.is_available() else "cpu"
    results_summary = {}

    # Check combined model weights with fallback to runs folder
    if not os.path.exists(combined_weights):
        fallback_path = "runs/detect/outputs_new/combined_model/weights/best.pt"
        if os.path.exists(fallback_path):
            combined_weights = fallback_path

    # 1. Evaluate Fine-Tuned Combined Model
    if os.path.exists(combined_weights):
        print(f"\n[1/2] Evaluating Fine-Tuned Combined Model: {combined_weights}")
        try:
            model_comb = YOLO(combined_weights)
            metrics_comb = model_comb.val(data=data_yaml, split="val", device=device, verbose=False)
            
            results_summary["Fine-Tuned Combined Model"] = {
                "mAP50": metrics_comb.box.map50,
                "mAP50-95": metrics_comb.box.map,
                "Precision": metrics_comb.box.mp,
                "Recall": metrics_comb.box.mr,
            }
        except Exception as e:
            print(f"Error evaluating combined model: {e}")
    else:
        print(f"Combined model weights not found at {combined_weights}")

    # 2. Evaluate Original Model
    if os.path.exists(original_weights):
        print(f"\n[2/2] Evaluating Original Detector Model: {original_weights}")
        try:
            model_orig = YOLO(original_weights)
            orig_yaml = "garbage_ai model/garbage_data.yaml"
            target_yaml = orig_yaml if os.path.exists(orig_yaml) else data_yaml
            metrics_orig = model_orig.val(data=target_yaml, split="val", device=device, verbose=False)
            
            results_summary["Original Model"] = {
                "mAP50": metrics_orig.box.map50,
                "mAP50-95": metrics_orig.box.map,
                "Precision": metrics_orig.box.mp,
                "Recall": metrics_orig.box.mr,
            }
        except Exception as e:
            print(f"Error evaluating original model: {e}")
    else:
        print(f"Original model weights not found at {original_weights}")

    print("\n" + "=" * 60)
    print("EVALUATION & COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Model':<30} | {'mAP50':<10} | {'mAP50-95':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 78)

    for name, metrics in results_summary.items():
        print(f"{name:<30} | {metrics['mAP50']:<10.4f} | {metrics['mAP50-95']:<10.4f} | {metrics['Precision']:<10.4f} | {metrics['Recall']:<10.4f}")

    print("=" * 60 + "\n")
    return results_summary

if __name__ == "__main__":
    evaluate_models()

