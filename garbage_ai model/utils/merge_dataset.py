import os
import shutil

def merge_datasets(split_dir, gen_dir, final_dir):
    print(f"Creating final dataset folder at {final_dir}...")
    
    # 1. Clean old final dir if exists
    if os.path.exists(final_dir):
        print(f"  Clearing old final dataset folder...")
        shutil.rmtree(final_dir)
        
    # 2. Copy the split dataset (train, validation, test)
    print(f"  Copying split dataset (Train/Val/Test) to final folder...")
    shutil.copytree(split_dir, final_dir)
    
    # 3. Add generated images to the train folders of final dataset
    classes = ["Clean", "Slightly_Dirty", "Very_Dirty"]
    for cls in classes:
        cls_gen_dir = os.path.join(gen_dir, cls)
        cls_train_dest = os.path.join(final_dir, "train", cls)
        
        if not os.path.exists(cls_gen_dir):
            print(f"  Warning: Generated folder {cls_gen_dir} does not exist.")
            continue
            
        os.makedirs(cls_train_dest, exist_ok=True)
        
        gen_files = [f for f in os.listdir(cls_gen_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
        print(f"  Merging {len(gen_files)} synthetic images into train/{cls}...")
        
        for f in gen_files:
            src_file_path = os.path.join(cls_gen_dir, f)
            # Prefix synthetic images to prevent name collisions
            dest_file_path = os.path.join(cls_train_dest, f"syn_{f}")
            shutil.copy2(src_file_path, dest_file_path)
            
    print("Dataset merging completed successfully!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Merge real split dataset with synthetic generated images.")
    parser.add_argument("--split_dir", type=str, default="datasets/RoadDataset_Split", help="Path to split dataset")
    parser.add_argument("--gen_dir", type=str, default="datasets/RoadDataset_Generated", help="Path to generated synthetic images")
    parser.add_argument("--final_dir", type=str, default="datasets/RoadDataset_Final", help="Path to save merged final dataset")
    args = parser.parse_args()
    
    merge_datasets(args.split_dir, args.gen_dir, args.final_dir)
