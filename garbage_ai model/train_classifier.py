import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from classifier.model import build_classifier

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def plot_history(history, output_path):
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    
    epochs_range = range(1, len(loss) + 1)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    if acc:
        plt.plot(epochs_range, acc, label='Training Accuracy')
    if val_acc:
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    if val_loss:
        plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def train_classifier(dataset_dir, output_dir, model_name="EfficientNet-B0", batch_size=32, epochs_stage1=25, epochs_stage2=25, seed=42):
    set_seed(seed)
    
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "validation")
    test_dir = os.path.join(dataset_dir, "test")
    
    print(f"\n==================================================")
    print(f"Training Cleanliness Classifier using {model_name}")
    print(f"==================================================")
    
    # Load dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        label_mode="categorical",
        batch_size=batch_size,
        image_size=(224, 224),
        seed=seed
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        label_mode="categorical",
        batch_size=batch_size,
        image_size=(224, 224),
        seed=seed
    )
    
    classes = train_ds.class_names
    num_classes = len(classes)
    print(f"Detected {num_classes} classes from training directories: {classes}")
    
    # Prefetch dataset to optimize CPU/GPU memory pipeline
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    
    # Build model
    model, base_model = build_classifier(model_name=model_name, num_classes=num_classes, img_size=224)
    model.summary()
    
    # Create outputs folder structure
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    reports_dir = os.path.join(output_dir, "reports")
    graphs_dir = os.path.join(output_dir, "graphs")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)
    
    # Compile model for stage 1 (only top layers trainable)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Callbacks for Stage 1
    callbacks_stage1 = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(checkpoints_dir, "best_stage1.keras"), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.CSVLogger(os.path.join(reports_dir, "training_stage1.log"))
    ]
    
    print("\nStarting Stage 1 training (Adversarial head)...")
    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_stage1,
        callbacks=callbacks_stage1
    )
    
    # ----------------------------------------
    # Stage 2: Fine-Tuning (Unfreeze last layers)
    # ----------------------------------------
    print("\nStarting Stage 2 training (Fine-Tuning base layers)...")
    
    # Unfreeze the base model
    base_model.trainable = True
    # Freeze all layers except the last 20 layers of the base model
    # (Unfreezing last 20 layers is highly effective for transfer learning tuning)
    for layer in base_model.layers[:-20]:
        layer.trainable = False
        
    # Recompile model with a lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Callbacks for Stage 2
    best_model_path = os.path.join(output_dir, "best_model.keras")
    callbacks_stage2 = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-7),
        tf.keras.callbacks.ModelCheckpoint(best_model_path, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.CSVLogger(os.path.join(reports_dir, "training_stage2.log"))
    ]
    
    history_stage2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_stage2,
        callbacks=callbacks_stage2
    )
    
    print(f"\nTraining completed! Best classifier model saved to {best_model_path}")
    
    # Save training graphs
    plot_history(history_stage2, os.path.join(graphs_dir, "training_history.png"))
    
    # Save class names label map
    label_path = os.path.join(output_dir, "labels.txt")
    with open(label_path, "w") as f:
        f.write("\n".join(classes))
        
    print(f"Label map saved to {label_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train road cleanliness classifier with transfer learning.")
    parser.add_argument("--dataset_dir", type=str, default="datasets/RoadDataset_Final", help="Path to final merged dataset")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Path to save checkpoints and outputs")
    parser.add_argument("--model_name", type=str, default="EfficientNet-B0", choices=["EfficientNet-B0", "MobileNetV3", "EfficientNetV2B0"], help="Backbone model name")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs_stage1", type=int, default=20, help="Epochs for training head")
    parser.add_argument("--epochs_stage2", type=int, default=15, help="Epochs for fine-tuning")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    train_classifier(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        epochs_stage1=args.epochs_stage1,
        epochs_stage2=args.epochs_stage2,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
