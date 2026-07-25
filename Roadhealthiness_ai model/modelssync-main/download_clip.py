import os
from transformers import CLIPProcessor, CLIPModel
print("Downloading CLIP Model from HuggingFace...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
print("Downloading Processor...")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("Download complete!")
