import os
import sys
import kagglehub
import shutil
from ultralytics import YOLO

def main():
    print("Starting Pothole Detection Model Training Pipeline...")
    
    dataset_yaml = "pothole/data.yaml"
    if not os.path.exists(dataset_yaml):
        print(f"Error: Could not find {dataset_yaml}. Run generate_dummy_labels.py first!")
        sys.exit(1)

    print(f"Found dataset YAML at {dataset_yaml}. Starting training...")
    
    # Initialize YOLOv8 nano model
    model = YOLO("yolov8n.pt")
    
    # Train the model
    results = model.train(
        data=dataset_yaml,
        epochs=10,
        imgsz=320,
        batch=16,
        device="cpu",
        project="pothole_detector",
        name="mobile_model"
    )
    print("Training completed!")
    
    # Normally we would export to TFLite here, but since TensorFlow installation 
    # was skipped for environment compatibility on this machine, the user can 
    # use the generated PyTorch (.pt) model and export it on a Linux/Colab machine.
    print("Model saved to: pothole_detector/mobile_model/weights/best.pt")
    print("To run on Android, export this .pt file to .tflite later.")

if __name__ == "__main__":
    main()
