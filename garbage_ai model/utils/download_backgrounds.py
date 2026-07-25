import os
import requests
import time

def download_images(dest_dir, count, start_id=0):
    os.makedirs(dest_dir, exist_ok=True)
    print(f"Downloading {count} images to {dest_dir}...")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    downloaded = 0
    attempts = 0
    max_attempts = count * 2
    
    while downloaded < count and attempts < max_attempts:
        attempts += 1
        img_id = start_id + attempts
        # Using a reliable random images API
        url = f"https://picsum.photos/224/224?random={img_id}"
        
        try:
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                filename = f"bg_{img_id}.jpg"
                filepath = os.path.join(dest_dir, filename)
                with open(filepath, "wb") as f:
                    f.write(response.content)
                downloaded += 1
                if downloaded % 20 == 0:
                    print(f"  Downloaded {downloaded}/{count}...")
                # Small sleep to prevent rate limits
                time.sleep(0.1)
        except Exception as e:
            # Silence connection errors and retry
            pass
            
    print(f"Finished downloading {downloaded} images.")
    return downloaded

def main():
    base_split_dir = "datasets/RoadDataset_Split"
    
    # Download 100 images for training
    download_images(os.path.join(base_split_dir, "train", "Not_a_Road"), 100, start_id=0)
    
    # Download 15 images for validation
    download_images(os.path.join(base_split_dir, "validation", "Not_a_Road"), 15, start_id=150)
    
    # Download 15 images for testing
    download_images(os.path.join(base_split_dir, "test", "Not_a_Road"), 15, start_id=200)
    
    print("\nBackground images downloaded and splits created successfully!")

if __name__ == "__main__":
    main()
