import os
import argparse
import time
import torch
import cv2
import numpy as np
from PIL import Image
from inference import load_pipeline, TRASH_CLASSES, CLIP_PROMPTS

def process_frame(cv_frame, classifier, detector, clip_model, clip_processor, device, conf_threshold=0.25):
    h_img, w_img, _ = cv_frame.shape
    
    # Convert BGR (OpenCV) to RGB (PIL) for models
    rgb_frame = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    
    # 1. Stage 1: Road Classification
    cls_results = classifier.predict(source=pil_image, verbose=False)
    probs = cls_results[0].probs
    class_id = probs.top1
    road_class = cls_results[0].names[class_id]
    confidence = probs.top1conf.item()
    
    road_class_map = {
        "CleanRoad": "Clean Road",
        "SligthlyDirty": "Slightly Dirty Road",
        "VeryDirty": "Very Dirty Road"
    }
    road_class_str = road_class_map.get(road_class, road_class)
    
    # 2. Stage 2: Garbage Detection (if not Clean)
    detections_summary = []
    if road_class != "CleanRoad":
        det_results = detector.predict(source=pil_image, conf=conf_threshold, verbose=False)
        boxes = det_results[0].boxes
        
        # 3. Stage 3: CLIP Zero-Shot classification
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            det_conf = box.conf[0].item()
            
            # Bound check
            x1_crop = max(0, x1)
            y1_crop = max(0, y1)
            x2_crop = min(w_img, x2)
            y2_crop = min(h_img, y2)
            
            if x2_crop > x1_crop and y2_crop > y1_crop:
                cropped_pil = pil_image.crop((x1_crop, y1_crop, x2_crop, y2_crop))
                
                # Run CLIP zero-shot classification
                inputs = clip_processor(text=CLIP_PROMPTS, images=cropped_pil, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    outputs = clip_model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs_clip = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
                    
                best_idx = np.argmax(probs_clip)
                trash_type = TRASH_CLASSES[best_idx]
                trash_conf = probs_clip[best_idx]
                
                detections_summary.append({
                    "box": [x1, y1, x2, y2],
                    "trash_type": trash_type,
                    "trash_confidence": trash_conf,
                    "det_confidence": det_conf
                })
                
    return road_class_str, confidence, detections_summary

def draw_overlays(cv_frame, road_status, road_conf, detections):
    h, w, _ = cv_frame.shape
    
    # 1. Draw overall road status in top-left
    status_colors = {
        "Clean Road": (0, 255, 0),         # Green
        "Slightly Dirty Road": (0, 165, 255), # Orange
        "Very Dirty Road": (0, 0, 255)      # Red
    }
    color_status = status_colors.get(road_status, (255, 255, 255))
    
    # Draw status background box
    cv2.rectangle(cv_frame, (10, 10), (450, 60), (0, 0, 0), -1)
    cv2.putText(cv_frame, f"ROAD: {road_status} ({road_conf:.1%})", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_status, 2)
                
    # 2. Draw bounding boxes around trash items
    trash_colors = {
        "plastic": (255, 0, 0),         # Blue
        "dry waste": (0, 255, 255),      # Yellow
        "wet waste": (0, 128, 0),        # Green
        "mixed waste": (0, 0, 255),      # Red
        "branches": (42, 42, 165),       # Brown
        "leaves": (0, 255, 0),           # Light Green
        "cement debris": (128, 128, 128) # Grey
    }
    
    counts = {tc: 0 for tc in TRASH_CLASSES}
    
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        ttype = det["trash_type"]
        tconf = det["trash_confidence"]
        
        counts[ttype] = counts.get(ttype, 0) + 1
        color = trash_colors.get(ttype, (255, 255, 255))
        
        # Bounding box
        cv2.rectangle(cv_frame, (x1, y1), (x2, y2), color, 2)
        
        # Label text
        label = f"{ttype} ({tconf:.1%})"
        label_y = y1 - 10 if y1 - 10 > 20 else y1 + 20
        cv2.putText(cv_frame, label, (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
    # 3. Draw Summary Panel in top-right
    panel_w = 280
    panel_h = 40 + (len(TRASH_CLASSES) * 25)
    px1, py1 = w - panel_w - 10, 10
    px2, py2 = w - 10, py1 + panel_h
    
    # Semi-transparent background
    overlay = cv_frame.copy()
    cv2.rectangle(overlay, (px1, py1), (px2, py2), (50, 50, 50), -1)
    cv2.addWeighted(overlay, 0.7, cv_frame, 0.3, 0, cv_frame)
    
    # Panel Title
    cv2.putText(cv_frame, "TRASH REPORT", (px1 + 15, py1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.line(cv_frame, (px1 + 10, py1 + 32), (px2 - 10, py1 + 32), (255, 255, 255), 1)
    
    # Draw counts for each category
    y_offset = py1 + 55
    for ttype in TRASH_CLASSES:
        count = counts[ttype]
        color = trash_colors.get(ttype, (255, 255, 255))
        
        # Draw category colored indicator dot
        cv2.circle(cv_frame, (px1 + 25, y_offset - 6), 5, color, -1)
        
        # Draw label and count
        txt = f"{ttype.capitalize()}: {count}"
        cv2.putText(cv_frame, txt, (px1 + 45, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
        
    return cv_frame

def main():
    parser = argparse.ArgumentParser(description="Real-Time Road Trash Video Application")
    parser.add_argument("--video", type=str, default="", help="Path to input video file. If empty, uses camera feed.")
    parser.add_argument("--cam_index", type=int, default=0, help="Index of camera to use if no video file provided.")
    parser.add_argument("--output", type=str, default="output/processed_video.mp4", help="Path to save output video.")
    parser.add_argument("--skip_frames", type=int, default=5, help="Number of frames to skip to maintain real-time speed.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for garbage detection.")
    args = parser.parse_args()
    
    # Load model pipeline
    try:
        classifier, detector, clip_model, clip_processor, device = load_pipeline()
    except Exception as e:
        print(f"Error loading models: {e}")
        return
        
    # Open video capture
    source = args.video if args.video else args.cam_index
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source: {source}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0
        
    print(f"\nVideo info: {width}x{height} @ {fps:.2f} FPS")
    print(f"Running pipeline. Press 'q' to quit.")
    
    # Setup Video Writer
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else "output", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    frame_idx = 0
    road_status = "Clean Road"
    road_conf = 1.0
    detections = []
    
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        
        # Process frame every skip_frames
        if frame_idx == 1 or frame_idx % args.skip_frames == 0:
            # Process frame
            road_status, road_conf, detections = process_frame(
                frame, classifier, detector, clip_model, clip_processor, device, args.conf
            )
            
        # Draw overlays using the latest available detection result
        annotated_frame = draw_overlays(frame.copy(), road_status, road_conf, detections)
        
        # Write to output file
        out_writer.write(annotated_frame)
        
        # Display frame in a window
        cv2.imshow("Road Trash Detector (Press 'q' to exit)", annotated_frame)
        
        # Keyboard interface (exit on 'q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Cleanup
    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()
    
    duration = time.time() - start_time
    print(f"\n==================================================")
    print(f"Processing finished!")
    print(f"Processed {frame_idx} frames in {duration:.1f} seconds ({frame_idx/duration:.1f} FPS)")
    print(f"Output saved to: {args.output}")
    print(f"==================================================")

if __name__ == "__main__":
    main()
