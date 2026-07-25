import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from models.fastgan import Generator, Discriminator

# Reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class RoadGANFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = []
        for root_dir, _, files in os.walk(folder_path):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    self.image_paths.append(os.path.join(root_dir, f))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

def crop_and_resize_real(real_imgs, size=32):
    # Downsample real images to match reconstruction resolution (e.g., 32x32)
    return F.interpolate(real_imgs, size=(size, size), mode='bilinear', align_corners=False)

def train_gan_for_class(class_name, train_dir, output_root, epochs=50, batch_size=16, nz=100, lr=0.0002, seed=42, sample_interval=5):
    set_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==================================================")
    print(f"Training GAN for class: {class_name} on device: {device}")
    # Paths
    class_train_dir = os.path.join(train_dir, class_name)
    generated_sample_dir = os.path.join(output_root, "generated_images", class_name)
    checkpoints_dir = os.path.join(output_root, "checkpoints", class_name)
    graphs_dir = os.path.join(output_root, "graphs")
    os.makedirs(generated_sample_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)
    
    # Dataset and Loader
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize to [-1, 1]
    ])
    
    dataset = RoadGANFolderDataset(class_train_dir, transform=transform)
    if len(dataset) == 0:
        print(f"Error: Training directory {class_train_dir} has 0 images!")
        return
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Models
    netG = Generator(nz=nz).to(device)
    netD = Discriminator().to(device)
    
    # Optimizers
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # Losses
    criterion = nn.BCELoss()
    mse_loss = nn.MSELoss()
    
    # Trackers
    d_losses = []
    g_losses = []
    
    # Training Loop
    for epoch in range(1, epochs + 1):
        epoch_d_loss = 0
        epoch_g_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for i, real_imgs in enumerate(progress_bar):
            real_imgs = real_imgs.to(device)
            current_batch_size = real_imgs.size(0)
            
            # Label smoothing / noisy labels for discriminator training stability
            real_label = torch.full((current_batch_size,), 0.9, dtype=torch.float32, device=device)
            fake_label = torch.full((current_batch_size,), 0.1, dtype=torch.float32, device=device)
            
            # ----------------------------------------
            # 1. Train Discriminator
            # ----------------------------------------
            optimizerD.zero_grad()
            
            # Forward pass real
            output_real, x8_real = netD(real_imgs)
            loss_D_real = criterion(output_real, real_label)
            
            # Self-Supervised Reconstruction Loss (reconstructing real from intermediate features)
            reconstructed_real = netD.reconstruct_features(x8_real)
            # Resize real image to 32x32 to match output of reconstruction decoder
            real_downsampled = torch.nn.functional.interpolate(real_imgs, size=(32, 32), mode="bilinear", align_corners=False)
            loss_reconstruct = mse_loss(reconstructed_real, real_downsampled)
            
            # Forward pass fake
            noise = torch.randn(current_batch_size, nz, 1, 1, device=device)
            fake_imgs = netG(noise)
            output_fake, _ = netD(fake_imgs.detach())
            loss_D_fake = criterion(output_fake, fake_label)
            
            # Total Discriminator Loss (combined adversarial and self-supervised reconstruction)
            loss_D = loss_D_real + loss_D_fake + 10.0 * loss_reconstruct
            loss_D.backward()
            optimizerD.step()
            
            # ----------------------------------------
            # 2. Train Generator
            # ----------------------------------------
            optimizerG.zero_grad()
            
            # Generator wants discriminator to output 1.0 (real) for fake images
            g_target_label = torch.full((current_batch_size,), 0.9, dtype=torch.float32, device=device)
            output_g, _ = netD(fake_imgs)
            loss_G = criterion(output_g, g_target_label)
            
            loss_G.backward()
            optimizerG.step()
            
            epoch_d_loss += loss_D.item()
            epoch_g_loss += loss_G.item()
            
            progress_bar.set_postfix({
                "D_Loss": f"{loss_D.item():.4f}", 
                "G_Loss": f"{loss_G.item():.4f}",
                "Recon": f"{loss_reconstruct.item():.4f}"
            })
            
        d_losses.append(epoch_d_loss / len(dataloader))
        g_losses.append(epoch_g_loss / len(dataloader))
        
        # Save sample images
        if epoch % sample_interval == 0 or epoch == 1 or epoch == epochs:
            netG.eval()
            with torch.no_grad():
                sample_noise = torch.randn(16, nz, 1, 1, device=device)
                sample_imgs = netG(sample_noise)
                # Rescale to [0, 1] for saving
                sample_imgs = (sample_imgs + 1.0) / 2.0
                save_path = os.path.join(generated_sample_dir, f"epoch_{epoch}.png")
                save_image(sample_imgs, save_path, nrow=4)
            netG.train()
            
        # Save checkpoint
        if epoch % 25 == 0 or epoch == epochs:
            checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'generator_state_dict': netG.state_dict(),
                'discriminator_state_dict': netD.state_dict(),
                'optimizerG_state_dict': optimizerG.state_dict(),
                'optimizerD_state_dict': optimizerD.state_dict(),
                'd_loss': d_losses[-1],
                'g_loss': g_losses[-1],
            }, checkpoint_path)
            
    # Save final weights separately
    final_weights_path = os.path.join(checkpoints_dir, "generator_final.pt")
    torch.save(netG.state_dict(), final_weights_path)
    
    # Save loss curve graph
    plt.figure()
    plt.plot(range(1, len(d_losses) + 1), d_losses, label="Discriminator Loss")
    plt.plot(range(1, len(g_losses) + 1), g_losses, label="Generator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"GAN Training Loss - {class_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(graphs_dir, f"gan_loss_{class_name}.png"))
    plt.close()
    
    print(f"Finished training GAN for {class_name}. Final Generator weights saved to {final_weights_path}.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train FastGAN on road cleanliness images.")
    parser.add_argument("--train_dir", type=str, default="datasets/RoadDataset_Split/train", help="Path to split train folder")
    parser.add_argument("--output_root", type=str, default="outputs", help="Root outputs directory")
    parser.add_argument("--class_name", type=str, default="all", choices=["all", "Clean", "Slightly_Dirty", "Very_Dirty"], help="Class to train GAN for (or 'all')")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--nz", type=int, default=100, help="Latent space dimension")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sample_interval", type=int, default=5, help="Sample generation frequency (epochs)")
    args = parser.parse_args()
    
    classes_to_train = ["Clean", "Slightly_Dirty", "Very_Dirty"] if args.class_name == "all" else [args.class_name]
    
    for cls in classes_to_train:
        train_gan_for_class(
            class_name=cls,
            train_dir=args.train_dir,
            output_root=args.output_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            nz=args.nz,
            lr=args.lr,
            seed=args.seed,
            sample_interval=args.sample_interval
        )

if __name__ == "__main__":
    main()
