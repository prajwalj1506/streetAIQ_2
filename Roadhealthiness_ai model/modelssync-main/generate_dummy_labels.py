import os
import cv2
import yaml
import shutil
import random

def create_yolo_dataset(source_dir, dest_dir):
    print(f"Creating YOLO formatted dataset in {dest_dir} from images in {source_dir}...")
    
    # Create YOLO directory structure
    for split in ['train', 'val']:
        os.makedirs(os.path.join(dest_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, split, 'labels'), exist_ok=True)
        
    # Get all jpg images
    images = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]
    random.shuffle(images)
    
    # Split 80% train, 20% val
    split_idx = int(len(images) * 0.8)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]
    
    def process_split(img_list, split_name):
        for img_name in img_list:
            src_path = os.path.join(source_dir, img_name)
            dst_img_path = os.path.join(dest_dir, split_name, 'images', img_name)
            dst_lbl_path = os.path.join(dest_dir, split_name, 'labels', img_name.replace('.jpg', '.txt'))
            
            # Copy image
            shutil.copy(src_path, dst_img_path)
            
            # Create a dummy label (class 0, x_center 0.5, y_center 0.5, width 0.2, height 0.2)
            # This allows YOLO to train just to prove the pipeline works end-to-end
            with open(dst_lbl_path, 'w') as f:
                f.write("0 0.5 0.5 0.2 0.2\n")

    process_split(train_imgs, 'train')
    process_split(val_imgs, 'val')
    
    # Create data.yaml
    yaml_content = {
        'path': os.path.abspath(dest_dir),
        'train': 'train/images',
        'val': 'val/images',
        'nc': 1,
        'names': ['Pothole']
    }
    
    yaml_path = os.path.join(dest_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f)
        
    print(f"Dataset created successfully! YAML file located at: {yaml_path}")
    return yaml_path

if __name__ == "__main__":
    create_yolo_dataset('Pothole_Image_Data', 'yolo_dataset')
