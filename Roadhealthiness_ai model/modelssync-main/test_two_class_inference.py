import os
import cv2
import sys
import torch
from ultralytics import YOLO

# 4 Class names and visual color scheme (BGR format)
CLASS_NAMES = {
    0: "Pothole",
    1: "Damaged road",
    2: "Debris",
    3: "Abandoned Vehicle"
}

CLASS_COLORS = {
    0: (0, 255, 0),      # Bright Green for isolated Pothole
    1: (0, 0, 255),      # Bright Red for Damaged road
    2: (0, 165, 255),    # Orange for Debris
    3: (255, 0, 255)     # Purple/Magenta for Abandoned Vehicle
}

def run_two_class_inference(
    image_path,
    model_path="pothole_detector/two_class_model/weights/best.pt",
    conf_threshold=0.25,
    output_save_path=None
):
    print(f"Loading 2-Class Pothole Model from: {model_path}")
    if not os.path.exists(model_path):
        # Fallback to synced weights path if two_class_model/weights/best.pt is in garbage_ai model
        fallback_path = "garbage_ai model/weights/pothole_two_class_detector.pt"
        if os.path.exists(fallback_path):
            model_path = fallback_path
        else:
            print(f"Error: Model weights not found at {model_path} or {fallback_path}")
            return None

    model = YOLO(model_path)

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not open image at {image_path}")
        return None

    h, w = frame.shape[:2]
    print(f"Running inference on image ({w}x{h})...")
    results = model(frame, conf=conf_threshold)

    detected_counts = {"Pothole": 0, "Damaged road": 0}

    for result in results:
        boxes = result.boxes
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())

            cls_name = CLASS_NAMES.get(cls_id, f"Class_{cls_id}")
            detected_counts[cls_name] = detected_counts.get(cls_name, 0) + 1
            color = CLASS_COLORS.get(cls_id, (255, 255, 0))

            x1, y1, x2, y2 = map(int, xyxy)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            # Draw label background box
            label = f"{cls_name}: {conf:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, max(y1 - text_h - 10, 0)), (x1 + text_w + 10, max(y1, text_h + 10)), color, -1)
            cv2.putText(frame, label, (x1 + 5, max(y1 - 5, text_h + 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            print(f" -> Detected [{cls_name}] Bounding Box: [{x1}, {y1}, {x2}, {y2}] Confidence: {conf:.2f}")

    if output_save_path is None:
        base_name = os.path.basename(image_path)
        output_save_path = f"output_twoclass_{base_name}"

    cv2.imwrite(output_save_path, frame)
    print(f"Saved visualization output image to: {output_save_path}")
    print(f"Summary: {detected_counts['Pothole']} isolated Pothole(s), {detected_counts['Damaged road']} Damaged Road section(s) detected.\n")

    return output_save_path

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_img = sys.argv[1]
    else:
        # Fallback to default dummy test image
        test_img = "Roadhealthiness_ai model/modelssync-main/dummy_test.jpg"
        if not os.path.exists(test_img):
            # Create a simple test image
            blank = np.full((640, 640, 3), 180, dtype=np.uint8)
            cv2.imwrite(test_img, blank)

    run_two_class_inference(test_img)
