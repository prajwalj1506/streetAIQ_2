import os
import shutil
import random
from sklearn.model_selection import train_test_split

def prepare_road_dataset():
    src_root = "E:/AImodel/RoadDataset/RoadDataset"
    dest_root = "C:/Users/yashw/.gemini/antigravity/scratch/trash_detection/road_data_split"
    
    classes = ["CleanRoad", "SligthlyDirty", "VeryDirty"]
    
    # Clean old split if exists
    if os.path.exists(dest_root):
        shutil.rmtree(dest_root)
        
    for cls in classes:
        src_cls_dir = os.path.join(src_root, cls)
        if not os.path.exists(src_cls_dir):
            print(f"Error: Source directory {src_cls_dir} does not exist!")
            continue
            
        images = [f for f in os.listdir(src_cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
        print(f"Found {len(images)} images in {cls}")
        
        # Ensure reproducibility
        random.seed(42)
        train_imgs, val_imgs = train_test_split(images, test_size=0.20, random_state=42)
        
        # Create directories
        os.makedirs(os.path.join(dest_root, "train", cls), exist_ok=True)
        os.makedirs(os.path.join(dest_root, "val", cls), exist_ok=True)
        
        # Copy files
        for img in train_imgs:
            shutil.copy(os.path.join(src_cls_dir, img), os.path.join(dest_root, "train", cls, img))
        for img in val_imgs:
            shutil.copy(os.path.join(src_cls_dir, img), os.path.join(dest_root, "val", cls, img))
            
        print(f"  Split {cls}: {len(train_imgs)} train, {len(val_imgs)} validation")

def prepare_garbage_yaml():
    yaml_content = """# Garbage Dataset absolute paths config
path: E:/AImodel/Garbage.v36-garbage-11-05-2026.yolov8
train: train/images
val: valid/images
test: test/images

nc: 2
names: ['Garbage', 'Not Garbage']
"""
    yaml_path = "C:/Users/yashw/.gemini/antigravity/scratch/trash_detection/garbage_data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"Created absolute path YAML config for YOLOv8 garbage detection at: {yaml_path}")

if __name__ == "__main__":
    print("Starting dataset preparation...")
    prepare_road_dataset()
    prepare_garbage_yaml()
    print("Dataset preparation completed!")
