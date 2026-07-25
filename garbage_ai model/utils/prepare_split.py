import os
import shutil
import random
from sklearn.model_selection import train_test_split

def split_dataset(src_dir, dest_dir, seed=42):
    # Mapping source folders (supporting both spelling formats)
    class_mapping = {
        "Clean": ["Clean", "CleanRoad"],
        "Slightly_Dirty": ["Slightly_Dirty", "SligthlyDirty"],
        "Very_Dirty": ["Very_Dirty", "VeryDirty"]
    }
    
    random.seed(seed)
    
    print(f"Splitting dataset from {src_dir} to {dest_dir}...")
    
    for target_class, src_options in class_mapping.items():
        src_class_path = None
        for option in src_options:
            path_check = os.path.join(src_dir, option)
            if os.path.exists(path_check):
                src_class_path = path_check
                break
                
        if not src_class_path:
            print(f"Warning: Could not find directory for class {target_class} (checked options: {src_options})")
            continue
            
        # Get all images
        images = [f for f in os.listdir(src_class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
        print(f"Found {len(images)} images for class {target_class} in {src_class_path}")
        
        # Split: 80% train, 20% temp (which will be split 50/50 into val and test)
        train_imgs, temp_imgs = train_test_split(images, test_size=0.20, random_state=seed)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.50, random_state=seed)
        
        splits = {
            "train": train_imgs,
            "validation": val_imgs,
            "test": test_imgs
        }
        
        # Copy to destinations
        for split_name, split_files in splits.items():
            split_dir_path = os.path.join(dest_dir, split_name, target_class)
            os.makedirs(split_dir_path, exist_ok=True)
            
            for img in split_files:
                src_img_path = os.path.join(src_class_path, img)
                dest_img_path = os.path.join(split_dir_path, img)
                shutil.copy2(src_img_path, dest_img_path)
                
            print(f"  Saved {len(split_files)} images to {split_name}/{target_class}")
            
    print("Dataset split completed successfully!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Split real road dataset into train/val/test splits.")
    parser.add_argument("--src", type=str, default="C:/Users/yashw/.gemini/antigravity/scratch/RoadDataset", help="Source directory")
    parser.add_argument("--dest", type=str, default="datasets/RoadDataset_Split", help="Destination split directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    split_dataset(args.src, args.dest, args.seed)
