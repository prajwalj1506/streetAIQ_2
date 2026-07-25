import os
import shutil
import glob
import random
import numpy as np
import cv2
from PIL import Image, ImageEnhance

def apply_gaussian_noise(cv_img, mean=0, sigma=15):
    noise = np.random.normal(mean, sigma, cv_img.shape).astype(np.float32)
    noisy_img = np.clip(cv_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy_img

def apply_motion_blur(cv_img, size=9):
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[int((size - 1) / 2), :] = np.ones(size, dtype=np.float32)
    kernel = kernel / float(size)
    return cv2.filter2D(cv_img, -1, kernel)

def augment_image_and_labels(img_path, labels_list, aug_type):
    """
    Applies image and bounding box augmentations for 2-class pothole dataset.
    """
    try:
        cv_img = cv2.imread(img_path)
        if cv_img is None:
            return None, labels_list
    except Exception as e:
        print(f"Error reading image {img_path}: {e}")
        return None, labels_list

    new_labels = []
    if aug_type == 'hflip':
        aug_img = cv2.flip(cv_img, 1)
        for line in labels_list:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = parts[0]
                try:
                    xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    new_xc = max(0.0, min(1.0, 1.0 - xc))
                    new_labels.append(f"{cls_id} {new_xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                except ValueError:
                    new_labels.append(line)
            else:
                new_labels.append(line)
        return aug_img, new_labels

    elif aug_type == 'brightness_up':
        pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        aug_img = cv2.cvtColor(np.array(ImageEnhance.Brightness(pil_img).enhance(1.2)), cv2.COLOR_RGB2BGR)
        return aug_img, labels_list

    elif aug_type == 'brightness_down':
        pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        aug_img = cv2.cvtColor(np.array(ImageEnhance.Brightness(pil_img).enhance(0.8)), cv2.COLOR_RGB2BGR)
        return aug_img, labels_list

    elif aug_type == 'gaussian_noise':
        return apply_gaussian_noise(cv_img, sigma=15), labels_list

    elif aug_type == 'motion_blur':
        return apply_motion_blur(cv_img, size=9), labels_list

    return cv_img, labels_list

def convert_line_to_bbox_label(line, img_w=640, img_h=640):
    """
    Parses YOLO line (bbox or polygon) and categorizes into:
    Class 0: Pothole (Isolated ditch)
    Class 1: Damaged road (Whole road covered in potholes / large area bbox)
    """
    parts = line.strip().split()
    if not parts:
        return None

    # Handle polygon coordinates (cls_id x1 y1 x2 y2 ...)
    if len(parts) > 5:
        cls_id = int(parts[0])
        coords = [float(x) for x in parts[1:]]
        xs = coords[0::2]
        ys = coords[1::2]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        xc = (min_x + max_x) / 2.0
        yc = (min_y + max_y) / 2.0
        w = max_x - min_x
        h = max_y - min_y

        area = w * h
        target_cls = 1 if area >= 0.15 else 0
        return f"{target_cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"

    elif len(parts) == 5:
        try:
            raw_cls = int(parts[0])
            xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            area = w * h
            
            # If raw_cls is explicitly road level or bbox area is large (>0.15 area ratio), classify as Damaged road (1)
            if raw_cls == 1 or area >= 0.15:
                target_cls = 1
            else:
                target_cls = 0

            return f"{target_cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
        except ValueError:
            return None

    return None

def prepare_two_class_dataset(
    output_dir=r"E:\AImodel\Pothole_TwoClass_Dataset",
    aiq_photos_root="aiq photos/road health",
    pseudo_labels_dir="aiq_pseudo_labels",
    pothole_roboflow_dir="Roadhealthiness_ai model/modelssync-main/pothole",
    split_ratio=0.90,
    seed=42
):
    print("=" * 65)
    print("Preparing 4-Class Road Health & Defect Detection Dataset")
    print("Class 0: Pothole (isolated ditch)")
    print("Class 1: Damaged road (surface cracks/eroded road)")
    print("Class 2: Debris (construction rubble/stones)")
    print("Class 3: Abandoned Vehicle (road hazard)")
    print("=" * 65)

    random.seed(seed)
    np.random.seed(seed)

    # Ensure output dataset directories exist
    for s in ["train", "valid", "test"]:
        os.makedirs(os.path.join(output_dir, s, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, s, "labels"), exist_ok=True)

    collected_samples = []

    # 1. Process AIQ Photos by category
    if os.path.exists(aiq_photos_root):
        for root_dir, _, files in os.walk(aiq_photos_root):
            folder_name = os.path.basename(root_dir).lower()
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    img_path = os.path.join(root_dir, f)
                    base = os.path.splitext(f)[0]
                    lbl_path = os.path.join(pseudo_labels_dir, f"{base}.txt")

                    labels = []
                    if "debris" in folder_name:
                        labels.append("2 0.500000 0.500000 0.500000 0.500000")
                    elif "abandoned" in folder_name or "vehicle" in folder_name:
                        labels.append("3 0.500000 0.500000 0.600000 0.600000")
                    else:
                        # Pothole & Damaged Road folder
                        pothole_count = 0
                        if os.path.exists(lbl_path):
                            with open(lbl_path, "r") as lf:
                                for line in lf:
                                    conv = convert_line_to_bbox_label(line)
                                    if conv:
                                        labels.append(conv)
                                        if conv.startswith("0"):
                                            pothole_count += 1
                        if not labels:
                            labels.append("0 0.500000 0.500000 0.400000 0.350000")
                        elif pothole_count >= 2:
                            labels.append("1 0.500000 0.600000 0.850000 0.650000")

                    collected_samples.append((img_path, labels, f"aiq_{folder_name}_{f}"))

    print(f"Collected {len(collected_samples)} samples from AIQ photos.")

    # 2. Process existing Roboflow / ModelSync Pothole dataset labels & synthetic samples
    roboflow_labels_dir = os.path.join(pothole_roboflow_dir, "train", "labels")
    if os.path.exists(roboflow_labels_dir):
        rf_label_files = glob.glob(os.path.join(roboflow_labels_dir, "*.txt"))
        for lbl_file in rf_label_files:
            base_name = os.path.splitext(os.path.basename(lbl_file))[0]
            matched_img = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                cand = os.path.join(pothole_roboflow_dir, "train", "images", f"{base_name}{ext}")
                if os.path.exists(cand):
                    matched_img = cand
                    break

            labels = []
            pothole_count = 0
            with open(lbl_file, "r") as lf:
                for line in lf:
                    conv = convert_line_to_bbox_label(line)
                    if conv:
                        labels.append(conv)
                        if conv.startswith("0"):
                            pothole_count += 1

            if pothole_count >= 3:
                labels.append("1 0.500000 0.550000 0.900000 0.700000")

            if matched_img and labels:
                collected_samples.append((matched_img, labels, f"rf_{base_name}.jpg"))

    print(f"Total collected dataset samples: {len(collected_samples)}")

    if not collected_samples:
        print("Warning: No samples found! Generating synthetic baseline samples...")
        dummy_img = np.full((640, 640, 3), 128, dtype=np.uint8)
        dummy_img_path = os.path.join("Roadhealthiness_ai model", "modelssync-main", "dummy_road.jpg")
        cv2.imwrite(dummy_img_path, dummy_img)
        collected_samples.append((dummy_img_path, ["0 0.3 0.4 0.2 0.2", "1 0.5 0.6 0.8 0.7"], "dummy_sample.jpg"))

    # 3. Shuffle & Split 90% Train, 10% Valid
    random.seed(seed)
    random.shuffle(collected_samples)
    split_idx = int(len(collected_samples) * split_ratio)
    train_samples = collected_samples[:split_idx]
    val_samples = collected_samples[split_idx:]

    print(f"Dataset split: Train = {len(train_samples)}, Valid = {len(val_samples)}")

    # 4. Write Train Split + Offline Augmentations
    aug_types = ['hflip', 'brightness_up', 'brightness_down', 'gaussian_noise', 'motion_blur']
    train_count = 0
    aug_count = 0

    for src_img, labels_list, out_filename in train_samples:
        base, ext = os.path.splitext(out_filename)
        dst_img_path = os.path.join(output_dir, "train", "images", out_filename)
        dst_lbl_path = os.path.join(output_dir, "train", "labels", f"{base}.txt")

        shutil.copy2(src_img, dst_img_path)
        with open(dst_lbl_path, "w") as f:
            f.write("\n".join(labels_list) + "\n")
        train_count += 1

        selected_augs = random.sample(aug_types, k=2)
        for aug_type in selected_augs:
            aug_img, aug_labels = augment_image_and_labels(src_img, labels_list, aug_type)
            if aug_img is not None:
                aug_filename = f"{base}_aug_{aug_type}{ext}"
                aug_base = os.path.splitext(aug_filename)[0]
                aug_dst_img = os.path.join(output_dir, "train", "images", aug_filename)
                aug_dst_lbl = os.path.join(output_dir, "train", "labels", f"{aug_base}.txt")

                cv2.imwrite(aug_dst_img, aug_img)
                with open(aug_dst_lbl, "w") as f:
                    f.write("\n".join(aug_labels) + "\n")
                aug_count += 1

    print(f"Wrote {train_count} base training samples + {aug_count} augmented samples.")

    # 5. Write Validation Split
    val_count = 0
    for src_img, labels_list, out_filename in val_samples:
        base = os.path.splitext(out_filename)[0]
        dst_img_path = os.path.join(output_dir, "valid", "images", out_filename)
        dst_lbl_path = os.path.join(output_dir, "valid", "labels", f"{base}.txt")

        shutil.copy2(src_img, dst_img_path)
        with open(dst_lbl_path, "w") as f:
            f.write("\n".join(labels_list) + "\n")
        val_count += 1

    print(f"Wrote {val_count} validation samples.")

    # 6. Save dataset configuration YAML
    yaml_content = f"""path: {output_dir.replace('\\', '/')}
train: train/images
val: valid/images
test: valid/images

nc: 4
names: ['Pothole', 'Damaged road', 'Debris', 'Abandoned Vehicle']
"""
    yaml_path = os.path.join("Roadhealthiness_ai model", "modelssync-main", "pothole_two_class.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"Saved 4-class road health dataset configuration to: {yaml_path}")
    print("4-class dataset preparation complete!\n")

if __name__ == "__main__":
    prepare_two_class_dataset()
