import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(split_dir, batch_size=32):
    # Load dataset using Keras
    return tf.keras.utils.image_dataset_from_directory(
        split_dir,
        label_mode="categorical",
        batch_size=batch_size,
        image_size=(224, 224),
        shuffle=False  # Keep order consistent for classification metrics evaluation
    )

def plot_confusion_matrix(y_true, y_pred, classes, output_path):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def evaluate_model(model_path, dataset_dir, output_dir, batch_size=32):
    print(f"\n==================================================")
    print(f"Evaluating Model: {model_path}")
    print(f"==================================================")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    model = tf.keras.models.load_model(model_path)
    
    # We will evaluate on both validation and test datasets
    splits = ["validation", "test"]
    # Load class names dynamically
    label_path = os.path.join(output_dir, "labels.txt")
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            classes = [line.strip() for line in f.read().split("\n") if line.strip()]
    else:
        classes = ["Clean", "Slightly_Dirty", "Very_Dirty"]
    
    reports_dir = os.path.join(output_dir, "reports")
    graphs_dir = os.path.join(output_dir, "graphs")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)
    
    eval_text = []
    
    for split in splits:
        split_path = os.path.join(dataset_dir, split)
        if not os.path.exists(split_path) or len(os.listdir(split_path)) == 0:
            print(f"Split folder {split} is empty or missing. Skipping...")
            continue
            
        print(f"\nEvaluating on {split} set...")
        ds = load_data(split_path, batch_size)
        
        # Predict
        y_pred_probs = model.predict(ds)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Extract true labels
        y_true = []
        for _, labels in ds:
            y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_true = np.array(y_true)
        
        # Metrics
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        # Per class accuracy
        cm = confusion_matrix(y_true, y_pred)
        class_accuracies = {}
        for idx, cls_name in enumerate(classes):
            class_total = cm[idx].sum()
            class_correct = cm[idx, idx]
            class_accuracies[cls_name] = class_correct / class_total if class_total > 0 else 0.0
            
        report = classification_report(y_true, y_pred, target_names=classes)
        
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        for cls_name, cls_acc in class_accuracies.items():
            print(f"  {cls_name} Accuracy: {cls_acc:.4f}")
            
        # Append split log text
        split_log = (
            f"==================================================\n"
            f"Split: {split.upper()}\n"
            f"==================================================\n"
            f"Accuracy:  {acc:.4f}\n"
            f"Precision: {precision:.4f}\n"
            f"Recall:    {recall:.4f}\n"
            f"F1 Score:  {f1:.4f}\n\n"
            f"Per-Class Accuracy:\n"
            + "\n".join([f"  {k}: {v:.4f}" for k, v in class_accuracies.items()])
            + f"\n\nClassification Report:\n{report}\n\n"
        )
        eval_text.append(split_log)
        
        # Save plots
        plot_confusion_matrix(y_true, y_pred, classes, os.path.join(graphs_dir, f"confusion_matrix_{split}.png"))
        
    # Write output reports
    report_file_path = os.path.join(reports_dir, "classification_report.txt")
    with open(report_file_path, "w") as f:
        f.write("\n".join(eval_text))
        
    print(f"\nFull evaluation report saved to {report_file_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate trained road cleanliness classifier.")
    parser.add_argument("--model", type=str, default="outputs/best_model.keras", help="Path to best classifier model file")
    parser.add_argument("--dataset_dir", type=str, default="datasets/RoadDataset_Final", help="Path to final dataset split folder")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Outputs root path")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
