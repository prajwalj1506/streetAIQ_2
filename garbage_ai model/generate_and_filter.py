import os
import glob
import numpy as np
import torch
from PIL import Image
import imagehash
from torchvision.utils import save_image
import sys

# Import FastGAN generator
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.fastgan import Generator

def generate_and_filter_synthetic_data(
    weights_path,
    output_dir,
    num_samples=100,
    nz=100,
    variance_thresh=0.08,
    phash_distance_thresh=5,
    img_size=(128, 128)
):
    """
    Generates synthetic images from FastGAN checkpoint, filters low quality (low variance)
    and removes near-duplicates via perceptual hashing (pHash).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nGenerative Filtering Pipeline operating on device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)

    netG = Generator(nz=nz).to(device)
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        if isinstance(state_dict, dict) and 'generator_state_dict' in state_dict:
            state_dict = state_dict['generator_state_dict']
        netG.load_state_dict(state_dict)
        print(f"Loaded Generator weights from {weights_path}")
    else:
        print(f"Generator weights not found at {weights_path}. Generator will use random weights.")

    netG.eval()

    saved_hashes = []
    generated_count = 0
    passed_count = 0
    rejected_variance = 0
    rejected_duplicate = 0

    batch_size = 16
    total_batches = (num_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for b in range(total_batches):
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake_imgs = netG(noise)  # Output range [-1, 1]
            fake_imgs = (fake_imgs + 1.0) / 2.0  # Rescale to [0, 1]
            fake_imgs = torch.clamp(fake_imgs, 0.0, 1.0)

            for i in range(fake_imgs.size(0)):
                generated_count += 1
                img_tensor = fake_imgs[i]
                img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
                
                # Check 1: Pixel Variance / Standard Deviation Filter
                std_dev = np.std(img_np)
                if std_dev < variance_thresh:
                    rejected_variance += 1
                    continue

                # Convert to PIL Image for pHash
                img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
                
                # Check 2: Perceptual Hash Duplicate Filter
                curr_hash = imagehash.phash(img_pil)
                is_duplicate = False
                for prev_hash in saved_hashes:
                    if curr_hash - prev_hash < phash_distance_thresh:
                        is_duplicate = True
                        break

                if is_duplicate:
                    rejected_duplicate += 1
                    continue

                # Passed both filters -> save image
                saved_hashes.append(curr_hash)
                passed_count += 1
                save_path = os.path.join(output_dir, f"syn_{passed_count:04d}.jpg")
                img_pil.save(save_path, quality=95)

                if passed_count >= num_samples:
                    break

            if passed_count >= num_samples:
                break

    print(f"\n--- Generation & Filtering Results ---")
    print(f"Total Generated: {generated_count}")
    print(f"Passed Filters: {passed_count}")
    print(f"Rejected (Low Variance): {rejected_variance}")
    print(f"Rejected (Perceptual Duplicates): {rejected_duplicate}")
    print(f"Saved synthetic images to: {output_dir}")
    return passed_count

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="outputs/checkpoints/garbage/generator_final.pt")
    parser.add_argument("--out_dir", type=str, default="outputs/filtered_synthetic/garbage")
    parser.add_argument("--num_samples", type=int, default=50)
    args = parser.parse_args()

    generate_and_filter_synthetic_data(args.weights, args.out_dir, num_samples=args.num_samples)
