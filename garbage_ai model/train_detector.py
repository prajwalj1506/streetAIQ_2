import os
import torch
from ultralytics import YOLO

def train_garbage_detector():
    yaml_config = "garbage_ai model/garbage_data.yaml"
    if not os.path.exists(yaml_config):
        yaml_config = "garbage_data.yaml"
        
    if not os.path.exists(yaml_config):
        raise FileNotFoundError(f"Configuration file {yaml_config} does not exist.")
        
    print("Loading pre-trained YOLOv8-medium detection model...")
    model = YOLO("yolov8m.pt")
    
    def on_epoch_end_auto_stop(trainer):
        metrics = getattr(trainer, 'metrics', {})
        map50 = metrics.get("metrics/mAP50(B)", 0)
        precision = metrics.get("metrics/precision(B)", 0)
        if map50 >= 0.80 or precision >= 0.80:
            print(f"\n🎯 TARGET REACHED: mAP50={map50:.4f}, Precision={precision:.4f} >= 80%! Stopping training immediately.")
            trainer.stop = True

    model.add_callback("on_fit_epoch_end", on_epoch_end_auto_stop)
    
    print("Starting target training of Road Cleanliness Detector (Garbage, Natural Waste, Not Garbage)...")
    
    train_args = dict(
        data=yaml_config,
        epochs=40,
        imgsz=640,
        batch=16,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=4 if torch.cuda.is_available() else 0,
        project="outputs_new",
        name="garbage_detector",
        exist_ok=True,
        save=True,
        plots=True,
        cos_lr=True,
        patience=15,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        cls=1.0,
        lr0=0.005,
        amp=True
    )

    results = model.train(**train_args)
    print("Garbage detector training completed successfully!")
    best_pt = os.path.join(results.save_dir, "weights", "best.pt")
    dst_pt = "garbage_ai model/weights/garbage_detector.pt"
    if os.path.exists(best_pt):
        os.makedirs(os.path.dirname(dst_pt), exist_ok=True)
        import shutil
        shutil.copy2(best_pt, dst_pt)
        print(f"Copied best detector weights to: {dst_pt}")

if __name__ == "__main__":
    train_garbage_detector()
