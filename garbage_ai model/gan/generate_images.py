import os
import sys
# Add parent directory to path so that models package can be imported when running from any folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
import torch
from torchvision.utils import save_image
from models.fastgan import Generator

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def is_valid_image(img_tensor, min_std=0.08, max_saturated_ratio=0.92):
    # img_tensor is expected to be in range [-1, 1] with shape [3, 128, 128]
    # Check 1: Calculate standard deviation of the image pixels
    std = img_tensor.std().item()
    if std < min_std:
        # Too flat (blank image)
        return False
        
    # Check 2: Check saturation (proportion of pixels extremely close to -1 or 1)
    # Scaled to [0, 1] for easier thresholding
    scaled_img = (img_tensor + 1.0) / 2.0
    saturated_pixels = (scaled_img < 0.02) | (scaled_img > 0.98)
    saturated_ratio = saturated_pixels.float().mean().item()
    
    if saturated_ratio > max_saturated_ratio:
        # Saturated/mode-collapsed image
        return False
        
    return True

def generate_synthetic_images(weights_dir, dest_root, count_per_class=200, nz=100, seed=42, min_std=0.08, max_saturated_ratio=0.92):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    classes = ["Clean", "Slightly_Dirty", "Very_Dirty"]
    
    print(f"\nGenerating {count_per_class} synthetic images per class using weights from {weights_dir}...")
    
    for cls in classes:
        cls_weights_path = os.path.join(weights_dir, cls, "generator_final.pt")
        if not os.path.exists(cls_weights_path):
            print(f"Warning: Final weights file not found for class {cls} at {cls_weights_path}")
            continue
            
        cls_dest_dir = os.path.join(dest_root, cls)
        os.makedirs(cls_dest_dir, exist_ok=True)
        
        # Load generator
        netG = Generator(nz=nz).to(device)
        netG.load_state_dict(torch.load(cls_weights_path, map_location=device))
        netG.eval()
        
        generated_count = 0
        attempts = 0
        max_attempts = count_per_class * 10  # increase attempts buffer
        
        print(f"Generating for class: {cls}...")
        
        with torch.no_grad():
            while generated_count < count_per_class and attempts < max_attempts:
                attempts += 1
                noise = torch.randn(1, nz, 1, 1, device=device)
                fake_img = netG(noise).squeeze(0)  # shape: [3, 128, 128]
                
                if is_valid_image(fake_img, min_std=min_std, max_saturated_ratio=max_saturated_ratio):
                    # Save image (scale to [0, 1] for saving)
                    save_path = os.path.join(cls_dest_dir, f"synthetic_{generated_count}.png")
                    save_image((fake_img + 1.0) / 2.0, save_path)
                    generated_count += 1
                
        print(f"  Successfully generated {generated_count} valid images for {cls} (discarded {attempts - generated_count} bad attempts).")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic images from trained generators.")
    parser.add_argument("--weights_dir", type=str, default="outputs/checkpoints", help="Path to GAN checkpoints")
    parser.add_argument("--dest", type=str, default="datasets/RoadDataset_Generated", help="Path to save generated dataset")
    parser.add_argument("--count", type=int, default=200, help="Number of images to generate per class (150-250)")
    parser.add_argument("--nz", type=int, default=100, help="Latent space dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--min_std", type=float, default=0.08, help="Minimum pixel standard deviation filter")
    parser.add_argument("--max_saturated", type=float, default=0.92, help="Maximum saturated pixel ratio filter")
    args = parser.parse_args()
    
    generate_synthetic_images(
        weights_dir=args.weights_dir,
        dest_root=args.dest,
        count_per_class=args.count,
        nz=args.nz,
        seed=args.seed,
        min_std=args.min_std,
        max_saturated_ratio=args.max_saturated
    )


if __name__ == "__main__":
    main()
