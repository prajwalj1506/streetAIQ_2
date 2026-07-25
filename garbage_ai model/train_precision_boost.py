import os
import torch
from ultralytics import YOLO

def boost_precision():
    yaml_config = "garbage_ai model/garbage_data.yaml"
    if not os.path.exists(yaml_config):
        yaml_config = "garbage_data.yaml"
        
    weights_path = "outputs_new/garbage_detector/weights/best.pt"
    if not os.path.exists(weights_path):
        weights_path = "garbage_ai model/weights/garbage_detector.pt"
        
    if not os.path.exists(weights_path):
        weights_path = "yolov8m.pt"
        
    print(f"Loading weights for Precision Boosting from: {weights_path}")
    model = YOLO(weights_path)
    
    def on_epoch_end_auto_stop(trainer):
        metrics = getattr(trainer, 'metrics', {})
        map50 = metrics.get("metrics/mAP50(B)", 0)
        precision = metrics.get("metrics/precision(B)", 0)
        if map50 >= 0.80 or precision >= 0.80:
            print(f"\n🎯 TARGET REACHED: mAP50={map50:.4f}, Precision={precision:.4f} >= 80%! Stopping training immediately.")
            trainer.stop = True

    model.add_callback("on_fit_epoch_end", on_epoch_end_auto_stop)
    
    print("Starting 10-Epoch Precision-Boosting Pass...")
    
    train_args = dict(
        data=yaml_config,
        epochs=10,
        imgsz=640,
        batch=16,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=4 if torch.cuda.is_available() else 0,
        project="outputs_new",
        name="garbage_detector_boosted",
        exist_ok=True,
        save=True,
        plots=True,
        cos_lr=True,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        cls=1.5,
        lr0=0.001,
        amp=True
    )

    results = model.train(**train_args)
    print("Precision Boosting training completed successfully!")
    best_pt = os.path.join(results.save_dir, "weights", "best.pt")
    dst_pt = "garbage_ai model/weights/garbage_detector.pt"
    if os.path.exists(best_pt):
        os.makedirs(os.path.dirname(dst_pt), exist_ok=True)
        import shutil
        shutil.copy2(best_pt, dst_pt)
        print(f"Copied boosted detector weights to: {dst_pt}")

if __name__ == "__main__":
    boost_precision()
