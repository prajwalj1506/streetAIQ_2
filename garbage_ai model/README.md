# Road Cleanliness Classifier Improvement Pipeline with FastGAN

This project implements a complete pipeline to improve a road cleanliness classification model by training a lightweight FastGAN to generate synthetic road images. The synthetic images are then merged with the original training set to improve classifier robustness and accuracy while avoiding overfitting.

---

## Project Folder Structure

```text
project/
├── models/
│   └── fastgan.py           # PyTorch implementation of FastGAN (Generator & Discriminator with SLE)
├── gan/
│   └── generate_images.py   # Script to generate synthetic images with blank/corrupted filters
├── classifier/
│   └── model.py             # TensorFlow/Keras classifier (EfficientNet-B0 + Preprocessing/Augmentation)
├── utils/
│   ├── prepare_split.py     # Script to split the original real dataset (80/10/10)
│   └── merge_dataset.py     # Script to merge real training split with GAN-generated images
│
├── train_gan.py             # Main PyTorch GAN training script (Trains one GAN per class)
├── train_classifier.py      # Main TensorFlow Keras classifier training script
├── convert_tflite.py        # Converts trained Keras model to a optimized, clean TFLite format
├── evaluate.py              # Evaluates model performance on untouched validation and test sets
├── requirements.txt         # Python dependencies
└── README.md                # Documentation
```

---

## Installation

### Local Execution (Windows/macOS/Linux)
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Google Colab Execution
Upload the project folder to Google Colab and install the requirements:
```python
!pip install -r requirements.txt
```

---

## Step-by-Step Execution Workflow

### Step 1: Split the Real Dataset
Run this to split the original real images (located at `C:/Users/Kokila/.antigravity-ide/aimodel/RoadDataset/RoadDataset` by default) into train (80%), validation (10%), and test (10%) sets:
```bash
python utils/prepare_split.py
```

### Step 2: Train the GANs
Train a separate FastGAN model for each class (`Clean`, `Slightly_Dirty`, `Very_Dirty`) using **only** the training real images. This will automatically use CUDA if available, otherwise it falls back to CPU:
```bash
# For a full run (e.g. 50 epochs)
python train_gan.py --epochs 50

# For a quick dry-run/verification (e.g. 1 epoch)
python train_gan.py --epochs 1
```

### Step 3: Generate Synthetic Images
Generate 150-250 synthetic images per class. Obviously flat, blank, or corrupted images are automatically filtered out using standard deviation and pixel saturation checks:
```bash
python gan/generate_images.py --count 200
```

### Step 4: Create Final Training Dataset
Merge the original real training images with the generated synthetic images into `RoadDataset_Final/`. The validation and test sets remain completely untouched:
```bash
python utils/merge_dataset.py
```

### Step 5: Train the Classifier
Train the **EfficientNet-B0** classifier using transfer learning and online TensorFlow augmentation. This uses a 2-stage training process: training the top classification head first, followed by fine-tuning the last layers of the backbone:
```bash
# For a full run
python train_classifier.py --epochs_stage1 20 --epochs_stage2 15

# For a quick dry-run/verification
python train_classifier.py --epochs_stage1 1 --epochs_stage2 1
```

### Step 6: Evaluate the Model
Evaluate the trained classifier against the untouched validation and test datasets:
```bash
python evaluate.py
```

### Step 7: Export to TensorFlow Lite
Convert the trained Keras model into an optimized, clean TFLite model (`outputs/road_cleanliness.tflite`). This script automatically extracts the core model and removes random augmentation layers to ensure complete compatibility and peak performance on mobile devices:
```bash
python convert_tflite.py
```

---

## Preprocessing & Data Augmentation Details
Augmentation is done dynamically inside the Keras model using preprocessing layers:
- Random Horizontal Flip
- Random Rotation (±15°)
- Random Zoom
- Random Translation
- Random Contrast & Brightness adjustment
- Gaussian Noise addition
- Random Crop (via Resizing + Random Crop)
