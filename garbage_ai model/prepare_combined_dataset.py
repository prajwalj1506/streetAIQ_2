import os
import shutil
import glob
import random
import numpy as np
import cv2
from PIL import Image, ImageEnhance

def apply_gaussian_noise(cv_img, mean=0, sigma=20):
    noise = np.random.normal(mean, sigma, cv_img.shape).astype(np.float32)
    noisy_img = np.clip(cv_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy_img

def apply_salt_pepper_noise(cv_img, amount=0.01):
    noisy = cv_img.copy()
    h, w = cv_img.shape[:2]
    num_salt = int(amount * h * w * 0.5)
    num_pepper = int(amount * h * w * 0.5)

    y_salt = np.random.randint(0, h, num_salt)
    x_salt = np.random.randint(0, w, num_salt)
    noisy[y_salt, x_salt] = 255

    y_pep = np.random.randint(0, h, num_pepper)
    x_pep = np.random.randint(0, w, num_pepper)
    noisy[y_pep, x_pep] = 0

    return noisy

def apply_motion_blur(cv_img, size=11):
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[int((size - 1) / 2), :] = np.ones(size, dtype=np.float32)
    kernel = kernel / float(size)
    return cv2.filter2D(cv_img, -1, kernel)

def apply_defocus_blur(cv_img, kernel_size=7):
    return cv2.GaussianBlur(cv_img, (kernel_size, kernel_size), 0)

def apply_jpeg_compression(cv_img, quality=35):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', cv_img, encode_param)
    return cv2.imdecode(encimg, 1)

def apply_random_cutout(cv_img, num_holes=2, max_size=40):
    h, w = cv_img.shape[:2]
    img_copy = cv_img.copy()
    for _ in range(num_holes):
        y = np.random.randint(0, h)
        x = np.random.randint(0, w)
        h_size = np.random.randint(15, max_size)
        w_size = np.random.randint(15, max_size)

        y1 = max(0, y - h_size // 2)
        y2 = min(h, y + h_size // 2)
        x1 = max(0, x - w_size // 2)
        x2 = min(w, x + w_size // 2)

        if y2 > y1 and x2 > x1:
            img_copy[y1:y2, x1:x2] = np.random.randint(0, 256, (y2 - y1, x2 - x1, 3), dtype=np.uint8)
    return img_copy

def apply_rotation_and_box(cv_img, label_text, angle_deg=10):
    h, w = cv_img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated_img = cv2.warpAffine(cv_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    lines = [line.strip() for line in label_text.strip().split("\n") if line.strip()]
    new_lines = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 5:
            cls_id = parts[0]
            try:
                xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                xmin = (xc - bw / 2.0) * w
                xmax = (xc + bw / 2.0) * w
                ymin = (yc - bh / 2.0) * h
                ymax = (yc + bh / 2.0) * h

                corners = np.array([
                    [xmin, ymin],
                    [xmax, ymin],
                    [xmax, ymax],
                    [xmin, ymax]
                ], dtype=np.float32)

                ones = np.ones((4, 1), dtype=np.float32)
                corners_homo = np.hstack([corners, ones])
                new_corners = M.dot(corners_homo.T).T

                new_xmin = max(0.0, min(float(w), new_corners[:, 0].min()))
                new_xmax = max(0.0, min(float(w), new_corners[:, 0].max()))
                new_ymin = max(0.0, min(float(h), new_corners[:, 1].min()))
                new_ymax = max(0.0, min(float(h), new_corners[:, 1].max()))

                new_bw = max(0.001, (new_xmax - new_xmin) / float(w))
                new_bh = max(0.001, (new_ymax - new_ymin) / float(h))
                new_xc = max(0.0, min(1.0, (new_xmin + new_xmax) / (2.0 * float(w))))
                new_yc = max(0.0, min(1.0, (new_ymin + new_ymax) / (2.0 * float(h))))

                new_lines.append(f"{cls_id} {new_xc:.6f} {new_yc:.6f} {new_bw:.6f} {new_bh:.6f}")
            except ValueError:
                new_lines.append(line)
        else:
            new_lines.append(line)

    return rotated_img, "\n".join(new_lines) + "\n"

def augment_image_and_label(img_path, label_text, aug_type):
    """
    Applies offline image & bounding box augmentation.
    Supported aug_types:
    - 'hflip': Horizontal Flip
    - 'brightness_up' / 'brightness_down': Lighting variations
    - 'gaussian_noise': Low-light sensor grain
    - 'salt_pepper': Hardware dropout / dirty lens noise
    - 'motion_blur': Dashcam / camera movement blur
    - 'defocus_blur': Camera defocus
    - 'jpeg_compress': Network compression artifacts
    - 'cutout': Simulated object occlusion
    - 'rot_pos' / 'rot_neg': Affine rotation with box recalculation
    """
    try:
        cv_img = cv2.imread(img_path)
        if cv_img is None:
            return None, label_text
    except Exception as e:
        print(f"Error opening {img_path}: {e}")
        return None, label_text

    if aug_type == 'hflip':
        aug_img = cv2.flip(cv_img, 1)
        lines = [line.strip() for line in label_text.strip().split("\n") if line.strip()]
        new_lines = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                cls_id = parts[0]
                try:
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    new_x = max(0.0, min(1.0, 1.0 - x_center))
                    new_lines.append(f"{cls_id} {new_x:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
                except ValueError:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        return aug_img, "\n".join(new_lines) + "\n"

    elif aug_type == 'brightness_up':
        pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        aug_img = cv2.cvtColor(np.array(ImageEnhance.Brightness(pil_img).enhance(1.25)), cv2.COLOR_RGB2BGR)
        return aug_img, label_text

    elif aug_type == 'brightness_down':
        pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        aug_img = cv2.cvtColor(np.array(ImageEnhance.Brightness(pil_img).enhance(0.75)), cv2.COLOR_RGB2BGR)
        return aug_img, label_text

    elif aug_type == 'gaussian_noise':
        return apply_gaussian_noise(cv_img, sigma=20), label_text

    elif aug_type == 'salt_pepper':
        return apply_salt_pepper_noise(cv_img, amount=0.01), label_text

    elif aug_type == 'motion_blur':
        return apply_motion_blur(cv_img, size=11), label_text

    elif aug_type == 'defocus_blur':
        return apply_defocus_blur(cv_img, kernel_size=7), label_text

    elif aug_type == 'jpeg_compress':
        return apply_jpeg_compression(cv_img, quality=35), label_text

    elif aug_type == 'cutout':
        return apply_random_cutout(cv_img, num_holes=2, max_size=45), label_text

    elif aug_type == 'rot_pos':
        return apply_rotation_and_box(cv_img, label_text, angle_deg=10)

    elif aug_type == 'rot_neg':
        return apply_rotation_and_box(cv_img, label_text, angle_deg=-10)

    return cv_img, label_text

def prepare_combined_dataset(
    original_dataset_path=r"E:\AImodel\Garbage.v36-garbage-11-05-2026.yolov8",
    aiq_photos_path="aiq photos",
    pseudo_labels_dir="aiq_pseudo_labels",
    synthetic_dir="outputs/filtered_synthetic",
    output_dataset_path=r"E:\AImodel\Garbage_Combined_Dataset",
    split_ratio=0.90,  # 90% train, 10% validation
    seed=42,
    enable_augmentation=True
):
    print("=" * 60)
    print("Preparing Combined Dataset (90% Train / 10% Validation Split)")
    print("With Noise, Image Degradation & Advanced Structural Augmentations")
    print("=" * 60)

    random.seed(seed)
    np.random.seed(seed)

    # Output structure
    splits = ["train", "valid", "test"]
    for s in splits:
        os.makedirs(os.path.join(output_dataset_path, s, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dataset_path, s, "labels"), exist_ok=True)

    # 1. Process Original Test Set
    orig_test_img_dir = os.path.join(original_dataset_path, "test", "images")
    orig_test_lbl_dir = os.path.join(original_dataset_path, "test", "labels")

    test_count = 0
    if os.path.exists(orig_test_img_dir):
        for img_name in os.listdir(orig_test_img_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                continue
            base = os.path.splitext(img_name)[0]
            src_img = os.path.join(orig_test_img_dir, img_name)
            dst_img = os.path.join(output_dataset_path, "test", "images", img_name)
            shutil.copy2(src_img, dst_img)

            src_lbl = os.path.join(orig_test_lbl_dir, f"{base}.txt")
            dst_lbl = os.path.join(output_dataset_path, "test", "labels", f"{base}.txt")

            labels = []
            has_garbage = False
            if os.path.exists(src_lbl):
                with open(src_lbl, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        cls_id = int(parts[0])
                        if cls_id == 0:
                            has_garbage = True
                        labels.append(line.strip())

            global_cls = 6 if has_garbage else 4  # Very Dirty Road vs Clean Road
            labels.append(f"{global_cls} 0.500000 0.500000 1.000000 1.000000")

            with open(dst_lbl, "w") as f:
                f.write("\n".join(labels) + "\n")
            test_count += 1

    print(f"Copied {test_count} original test samples to test split.")

    # 2. Collect all real items
    real_items = []

    # 2a. Original Train & Valid
    for orig_split in ["train", "valid"]:
        img_dir = os.path.join(original_dataset_path, orig_split, "images")
        lbl_dir = os.path.join(original_dataset_path, orig_split, "labels")
        if os.path.exists(img_dir):
            for img_name in os.listdir(img_dir):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    continue
                base = os.path.splitext(img_name)[0]
                src_img = os.path.join(img_dir, img_name)
                src_lbl = os.path.join(lbl_dir, f"{base}.txt")

                labels = []
                has_garbage = False
                if os.path.exists(src_lbl):
                    with open(src_lbl, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if not parts:
                                continue
                            cls_id = int(parts[0])
                            if cls_id == 0:
                                has_garbage = True
                            labels.append(line.strip())

                if not has_garbage:
                    labels.append("2 0.500000 0.500000 1.000000 1.000000")
                label_text = "\n".join(labels) + "\n"

                real_items.append((src_img, label_text, f"orig_{orig_split}_{img_name}", False))

    # 2b. AIQ Photos + Pseudo Labels
    aiq_count = 0
    cleanliness_root = "aiq photos/road cleanliness"
    if os.path.exists(cleanliness_root):
        for root_dir, _, files in os.walk(cleanliness_root):
            folder_name = os.path.basename(root_dir).lower()
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    img_path = os.path.join(root_dir, f)
                    base = os.path.splitext(f)[0]
                    pseudo_lbl_path = os.path.join(pseudo_labels_dir, f"{base}.txt")

                    labels = []
                    if "natural" in folder_name or "leaves" in folder_name or "branches" in folder_name:
                        labels.append("1 0.500000 0.500000 0.500000 0.500000")
                    elif "clean" in folder_name:
                        labels.append("2 0.500000 0.500000 1.000000 1.000000")
                    else:
                        # Garbage folder
                        if os.path.exists(pseudo_lbl_path):
                            with open(pseudo_lbl_path, "r") as lf:
                                for line in lf:
                                    parts = line.strip().split()
                                    if parts and parts[0] == '0':
                                        labels.append(line.strip())
                        if not labels:
                            labels.append("0 0.500000 0.500000 0.400000 0.400000")

                    label_text = "\n".join(labels) + "\n"
                    real_items.append((img_path, label_text, f"aiq_{folder_name}_{f}", True))
                    aiq_count += 1

    print(f"Total real items collected: {len(real_items)} (including {aiq_count} AIQ cleanliness photos)")

    # 3. Shuffle & Split 90% Train / 10% Validation
    random.shuffle(real_items)
    num_train = int(len(real_items) * split_ratio)
    train_real = real_items[:num_train]
    val_real = real_items[num_train:]

    print(f"Split real items into: Train = {len(train_real)}, Validation = {len(val_real)}")

    aug_types = [
        'hflip', 'brightness_up', 'brightness_down',
        'gaussian_noise', 'salt_pepper', 'motion_blur',
        'defocus_blur', 'jpeg_compress', 'cutout',
        'rot_pos', 'rot_neg'
    ]

    aug_count = 0
    # Write Real Train items and create advanced offline augmented variants
    for src_img, lbl_text, out_name, is_aiq in train_real:
        base, ext = os.path.splitext(out_name)
        dst_img = os.path.join(output_dataset_path, "train", "images", out_name)
        dst_lbl = os.path.join(output_dataset_path, "train", "labels", f"{base}.txt")
        shutil.copy2(src_img, dst_img)
        with open(dst_lbl, "w") as f:
            f.write(lbl_text)

        if enable_augmentation and is_aiq:
            selected_augs = random.sample(aug_types, k=4)
            for aug_type in selected_augs:
                aug_img, aug_lbl = augment_image_and_label(src_img, lbl_text, aug_type)
                if aug_img is not None:
                    aug_out_name = f"{base}_aug_{aug_type}{ext}"
                    aug_base = os.path.splitext(aug_out_name)[0]
                    aug_dst_img = os.path.join(output_dataset_path, "train", "images", aug_out_name)
                    aug_dst_lbl = os.path.join(output_dataset_path, "train", "labels", f"{aug_base}.txt")
                    
                    cv2.imwrite(aug_dst_img, aug_img)
                    with open(aug_dst_lbl, "w") as f:
                        f.write(aug_lbl)
                    aug_count += 1

    print(f"Created {aug_count} advanced augmented image & label samples for Train split.")

    # Write Real Validation items
    for src_img, lbl_text, out_name, _ in val_real:
        base = os.path.splitext(out_name)[0]
        dst_img = os.path.join(output_dataset_path, "valid", "images", out_name)
        dst_lbl = os.path.join(output_dataset_path, "valid", "labels", f"{base}.txt")
        shutil.copy2(src_img, dst_img)
        with open(dst_lbl, "w") as f:
            f.write(lbl_text)

    # 4. Add GAN Synthetic images
    syn_count = 0
    if os.path.exists(synthetic_dir):
        syn_files = glob.glob(os.path.join(synthetic_dir, "**", "*.jpg"), recursive=True) + \
                    glob.glob(os.path.join(synthetic_dir, "**", "*.png"), recursive=True)
        for syn_path in syn_files:
            syn_name = f"synthetic_{syn_count:04d}_{os.path.basename(syn_path)}"
            base = os.path.splitext(syn_name)[0]

            dst_img = os.path.join(output_dataset_path, "train", "images", syn_name)
            dst_lbl = os.path.join(output_dataset_path, "train", "labels", f"{base}.txt")
            shutil.copy2(syn_path, dst_img)

            syn_label = "0 0.500000 0.500000 0.500000 0.500000\n"
            with open(dst_lbl, "w") as f:
                f.write(syn_label)
            syn_count += 1

    print(f"Added {syn_count} synthetic GAN samples to Train split.")

    # 5. Generate garbage_data.yaml
    yaml_content = f"""path: {output_dataset_path.replace('\\', '/')}
train: train/images
val: valid/images
test: test/images

nc: 3
names: ['Garbage', 'Natural Waste', 'Not Garbage']
"""
    yaml_path = os.path.join("garbage_ai model", "garbage_data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"Saved dataset yaml config to: {yaml_path}")
    print("Road cleanliness dataset preparation complete!\n")

if __name__ == "__main__":
    prepare_combined_dataset()
