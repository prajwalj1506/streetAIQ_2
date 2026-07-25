import os
import shutil
import torch
from ultralytics import YOLO

def train_two_class_pothole_model(
    data_yaml="Roadhealthiness_ai model/modelssync-main/pothole_two_class.yaml",
    base_weights="yolov8n.pt",
    output_dir="pothole_detector/two_class_model",
    epochs=50,
    batch_size=16,
    img_size=640
):
    print("=" * 65)
    print("Starting 4-Class Road Health & Defect YOLO Model Training...")
    print(f"Target Epochs: {epochs}")
    print("Class 0: Pothole | Class 1: Damaged road | Class 2: Debris | Class 3: Abandoned Vehicle")
    print("=" * 65)

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training using device: {device}")

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Dataset YAML config not found at: {data_yaml}")

    # Load YOLO base model
    model = YOLO(base_weights)

    # Execute YOLO training
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=os.path.dirname(output_dir) if os.path.dirname(output_dir) else "pothole_detector",
        name=os.path.basename(output_dir),
        exist_ok=True,
        workers=0,
        save=True
    )

    # Synchronize best model weights to target directories
    best_weights_src = os.path.join(output_dir, "weights", "best.pt")
    
    target_locations = [
        "garbage_ai model/weights/pothole_two_class_detector.pt",
        "Roadhealthiness_ai model/modelssync-main/pothole_detector/mobile_model5/weights/best.pt",
        "Roadhealthiness_ai model/modelssync-main/pothole_detector/mobile_model/weights/best.pt"
    ]

    if os.path.exists(best_weights_src):
        for dst_path in target_locations:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(best_weights_src, dst_path)
            print(f"Copied fine-tuned best model weights to: {dst_path}")
    else:
        print(f"Warning: best.pt was not found at {best_weights_src}")

    print("\nTwo-class pothole model training completed successfully!\n")
    return results

if __name__ == "__main__":
    train_two_class_pothole_model()
