import os
import sys
import time

# Ensure garbage_ai model module import path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pseudo_label import run_pseudo_labeling
from prepare_combined_dataset import prepare_combined_dataset
from train_detector import train_garbage_detector
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Roadhealthiness_ai model", "modelssync-main"))
from prepare_two_class_pothole_dataset import prepare_two_class_dataset
from train_two_class_pothole import train_two_class_pothole_model

def main():
    print("=" * 70)
    print("      DUAL-APP MODEL ENHANCEMENT PIPELINE (50 EPOCHS)")
    print("=" * 70)
    start_time = time.time()

    # Step 1: Pseudo-labeling AIQ photos
    print("\n>>> STEP 1: Generating Pseudo-Labels for AIQ Photos...")
    run_pseudo_labeling(
        aiq_root="aiq photos",
        weights_path="garbage_ai model/weights/garbage_detector.pt",
        output_dir="aiq_pseudo_labels"
    )

    # Step 2: Prepare & Train Road Cleanliness Detector (50 Epochs)
    print("\n>>> STEP 2: Preparing Road Cleanliness Dataset (Garbage & Natural Waste)...")
    prepare_combined_dataset(
        original_dataset_path=r"E:\AImodel\Garbage.v36-garbage-11-05-2026.yolov8",
        aiq_photos_path="aiq photos",
        pseudo_labels_dir="aiq_pseudo_labels",
        output_dataset_path=r"E:\AImodel\Garbage_Combined_Dataset",
        split_ratio=0.90
    )

    print("\n>>> STEP 3: Training Road Cleanliness Detector (50 Epochs)...")
    train_garbage_detector()

    # Step 3: Prepare & Train Road Health Detector (50 Epochs)
    print("\n>>> STEP 4: Preparing Road Health Dataset (Potholes, Damaged Road, Debris, Vehicles)...")
    prepare_two_class_dataset(
        output_dir=r"E:\AImodel\Pothole_TwoClass_Dataset",
        aiq_photos_root="aiq photos/road health",
        pseudo_labels_dir="aiq_pseudo_labels",
        pothole_roboflow_dir="Roadhealthiness_ai model/modelssync-main/pothole",
        split_ratio=0.90
    )

    print("\n>>> STEP 5: Training Road Health Detector (50 Epochs)...")
    train_two_class_pothole_model(
        data_yaml="Roadhealthiness_ai model/modelssync-main/pothole_two_class.yaml",
        output_dir="pothole_detector/two_class_model",
        epochs=50,
        batch_size=16,
        img_size=640
    )

    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"      DUAL-APP PIPELINE EXECUTED SUCCESSFULLY IN {elapsed/60:.2f} MINUTES")
    print("=" * 70)

if __name__ == "__main__":
    main()
