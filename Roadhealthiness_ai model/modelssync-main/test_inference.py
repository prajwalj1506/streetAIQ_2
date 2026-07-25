import cv2
from ultralytics import YOLO

def classify_pothole_size(bbox, frame_width, frame_height, threshold_ratio=0.05):
    """
    Classifies a pothole as 'small' or 'big' based on its bounding box area
    relative to the total frame area.
    
    bbox: [x1, y1, x2, y2]
    threshold_ratio: The fraction of the frame area that defines a 'big' pothole.
    """
    x1, y1, x2, y2 = bbox
    bbox_area = (x2 - x1) * (y2 - y1)
    frame_area = frame_width * frame_height
    
    ratio = bbox_area / frame_area
    if ratio >= threshold_ratio:
        return "Big Pothole", ratio
    else:
        return "Small Pothole", ratio

def test_inference(image_path, model_path="pothole_detector/mobile_model/weights/best.pt"):
    print(f"Loading model from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Failed to load image at {image_path}")
        return
        
    frame_height, frame_width = frame.shape[:2]
    
    print("Running inference...")
    results = model(frame)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get bounding box coordinates [x1, y1, x2, y2]
            xyxy = box.xyxy[0].cpu().numpy()
            
            # Get confidence
            conf = float(box.conf[0].cpu().numpy())
            
            # Classify size
            size_class, area_ratio = classify_pothole_size(xyxy, frame_width, frame_height)
            
            print(f"Detected Pothole - Confidence: {conf:.2f}, Size: {size_class} (Area Ratio: {area_ratio:.4f})")
            
            # Draw on image
            x1, y1, x2, y2 = map(int, xyxy)
            color = (0, 0, 255) if size_class == "Big Pothole" else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{size_class}: {conf:.2f}"
            cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save output
    output_path = "output_" + image_path.split("/")[-1]
    cv2.imwrite(output_path, frame)
    print(f"Saved output image to {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python test_inference.py <path_to_image>")
        sys.exit(1)
    
    test_inference(sys.argv[1])
