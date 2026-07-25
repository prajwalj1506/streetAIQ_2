import os
import glob
from PIL import Image
import torch
from ultralytics import YOLO

def run_pseudo_labeling(
    aiq_root="aiq photos",
    weights_path="garbage_ai model/weights/garbage_detector.pt",
    output_dir="aiq_pseudo_labels",
    conf_thresh=0.50
):
    print("=" * 60)
    print("Starting Pseudo-Labeling Process...")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    
    # Load model if weights exist
    detector = None
    if os.path.exists(weights_path):
        print(f"Loading pre-trained detector from {weights_path}")
        detector = YOLO(weights_path)
    else:
        print(f"Warning: Pre-trained weights not found at {weights_path}. Using standard heuristics.")

    # Mappings:
    # 0: Garbage, 1: Pothole, 2: Debris, 3: Abandoned Vehicle
    # 4: Clean Road, 5: Slightly Dirty Road, 6: Very Dirty Road
    
    category_configs = {
        os.path.join(aiq_root, "road cleanliness", "clean roads"): {
            "global_cls": 4,  # Clean Road
            "local_cls": None
        },
        os.path.join(aiq_root, "road cleanliness", "natural waste"): {
            "global_cls": 5,  # Slightly Dirty Road
            "local_cls": 0,  # Garbage / Waste
            "use_model": True
        },
        os.path.join(aiq_root, "road cleanliness", "garbage"): {
            "global_cls": 6,  # Very Dirty Road
            "local_cls": 0,  # Garbage
            "use_model": True
        },
        os.path.join(aiq_root, "road health", "potholes"): {
            "global_cls": 5,  # Slightly Dirty Road
            "local_cls": 1,  # Pothole
            "default_box": [0.5, 0.6, 0.4, 0.3]
        },
        os.path.join(aiq_root, "road health", "debris"): {
            "global_cls": 5,  # Slightly Dirty Road
            "local_cls": 2,  # Debris
            "default_box": [0.5, 0.5, 0.5, 0.4]
        },
        os.path.join(aiq_root, "road health", "abandoned vehicles"): {
            "global_cls": 6,  # Very Dirty Road
            "local_cls": 3,  # Abandoned Vehicle
            "default_box": [0.5, 0.5, 0.6, 0.5]
        }
    }

    total_images_processed = 0
    total_boxes_generated = 0

    for folder_path, config in category_configs.items():
        if not os.path.exists(folder_path):
            print(f"Directory not found: {folder_path}, skipping...")
            continue
            
        image_files = []
        for root_dir, _, files in os.walk(folder_path):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    image_files.append(os.path.join(root_dir, f))

        print(f"\nProcessing {len(image_files)} images in '{folder_path}'...")
        
        for img_path in image_files:
            img_name = os.path.basename(img_path)
            base_name = os.path.splitext(img_name)[0]
            txt_path = os.path.join(output_dir, f"{base_name}.txt")

            labels = []
            # 1. Add global classification box [class_id 0.5 0.5 1.0 1.0]
            global_cls = config["global_cls"]
            labels.append(f"{global_cls} 0.500000 0.500000 1.000000 1.000000")

            # 2. Add local object boxes
            if config.get("use_model") and detector is not None:
                results = detector.predict(img_path, conf=conf_thresh, verbose=False)
                boxes_added = 0
                for r in results:
                    for box in r.boxes:
                        # Extract normalized xywh
                        xywh = box.xywhn[0].tolist()
                        cls_id = config["local_cls"]
                        labels.append(f"{cls_id} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f}")
                        boxes_added += 1
                total_boxes_generated += boxes_added

            elif config.get("local_cls") is not None and "default_box" in config:
                cls_id = config["local_cls"]
                box = config["default_box"]
                labels.append(f"{cls_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}")
                total_boxes_generated += 1

            # Save label file
            with open(txt_path, "w") as f:
                f.write("\n".join(labels) + "\n")

            total_images_processed += 1

    print(f"\nDone! Processed {total_images_processed} images.")
    print(f"Generated pseudo-label txt files in: {output_dir}")
    return total_images_processed

if __name__ == "__main__":
    run_pseudo_labeling()
