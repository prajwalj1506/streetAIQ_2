import os
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tensorflow as tf
from transformers import CLIPProcessor, CLIPModel

# Define paths for the models
CLASSIFIER_PATH = "garbage_ai model/outputs/best_model.keras" if os.path.exists("garbage_ai model/outputs/best_model.keras") else "outputs/best_model.keras"
DETECTOR_PATH = "garbage_ai model/weights/garbage_detector.pt" if os.path.exists("garbage_ai model/weights/garbage_detector.pt") else "weights/garbage_detector.pt"

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Zero-shot classes and prompts
TRASH_CLASSES = ["plastic", "dry waste", "wet waste", "mixed waste", "branches", "leaves", "cement debris"]
CLIP_PROMPTS = [
    "a photo of plastic waste, plastic bottle, or plastic bag on the road",
    "a photo of dry waste, paper, cardboard, metal, or glass on the road",
    "a photo of wet waste, food waste, organic waste, or fruit peels on the road",
    "a photo of mixed household garbage, trash bags, or mixed waste on the road",
    "a photo of tree branches, twigs, or wood on the road",
    "a photo of leaves or foliage on the road",
    "a photo of cement debris, concrete, bricks, stones, or construction waste on the road"
]

def load_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading pipeline using device: {device.upper()}")
    
    # Load Stage 1 Keras Classifier
    if not os.path.exists(CLASSIFIER_PATH):
        raise FileNotFoundError(f"Classifier model not found at {CLASSIFIER_PATH}. Please train the classifier first.")
    print("Loading Road Classifier (Keras)...")
    
    # Custom initializer to handle Keras 3 -> Keras 2 backward compatibility bugs in Colab
    class SafeVarianceScaling(tf.keras.initializers.VarianceScaling):
        def __init__(self, scale=1.0, mode="fan_in", distribution="truncated_normal", seed=None, **kwargs):
            # Ignore extra kwargs like 'input_axes' or 'output_axes' that cause crashes
            super().__init__(scale=scale, mode=mode, distribution=distribution, seed=seed)
            
    classifier = tf.keras.models.load_model(
        CLASSIFIER_PATH, 
        custom_objects={'VarianceScaling': SafeVarianceScaling},
        compile=False
    )
    
    # Load Stage 2 Detector
    if not os.path.exists(DETECTOR_PATH):
        raise FileNotFoundError(f"Detector model not found at {DETECTOR_PATH}. Please train the detector first.")
    print("Loading Garbage Detector...")
    detector = YOLO(DETECTOR_PATH)
    
    # Load Stage 3 CLIP Zero-Shot Classifier
    print("Loading CLIP Zero-shot Classifier...")
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    
    return classifier, detector, clip_model, clip_processor, device

def run_pipeline(image_path, classifier, detector, clip_model, clip_processor, device, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Image
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
        
    print(f"\n==================================================")
    print(f"Processing image: {os.path.basename(image_path)}")
    print(f"==================================================")
    
    pil_image = Image.open(image_path).convert("RGB")
    cv_image = cv2.imread(image_path)
    h_img, w_img, _ = cv_image.shape
    
    # --------------------------------------------------------
    # [Verification Stage] Zero-shot Road vs Non-Road check
    # --------------------------------------------------------
    print("\n[Verification] Verifying if image contains a road/street...")
    road_verification_prompts = [
        "a photo of a road, street, highway, asphalt pavement, or public street view",
        "a photo of an indoor room, house interior, office, human face, sky view, indoor wall, furniture, tree close-up, animal, or anything other than a road"
    ]
    
    inputs_road = clip_processor(text=road_verification_prompts, images=pil_image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs_road = clip_model(**inputs_road)
        logits_per_image_road = outputs_road.logits_per_image
        probs_road = logits_per_image_road.softmax(dim=-1).cpu().numpy()[0]
        
    road_prob = probs_road[0]
    non_road_prob = probs_road[1]
    
    print(f"Road probability: {road_prob:.2%}, Non-road probability: {non_road_prob:.2%}")
    
    if non_road_prob > road_prob:
        print("Verification Failed: The input image does not appear to contain a road or street.")
        # Draw error message on visualization image
        cv2.putText(cv_image, "No road detected in the image", (20, h_img // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        out_path = os.path.join(output_dir, f"result_{os.path.basename(image_path)}")
        cv2.imwrite(out_path, cv_image)
        print(f"Saved visualization to: {out_path}")
        return {
            "error": "No road detected in the image",
            "road_status": None,
            "detections": []
        }
        
    print("Verification Passed: Road detected. Executing classification pipeline...")
    
    # 2. Stage 1: Road Classification (Keras Model)
    print("\n[Stage 1] Classifying road cleanliness...")
    img_resized = pil_image.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Predict using Keras model
    preds = classifier.predict(img_array, verbose=0)
    class_id = np.argmax(preds[0])
    confidence = preds[0][class_id]
    
    # Map alphabetical Keras class indexes: 0: Clean, 1: Slightly_Dirty, 2: Very_Dirty
    # Map to legacy class name keys for downstream stage compatibility
    road_class = ["CleanRoad", "SligthlyDirty", "VeryDirty"][class_id]
    
    # Standardize names for display
    road_class_map = {
        "CleanRoad": "Clean Road",
        "SligthlyDirty": "Slightly Dirty Road",
        "VeryDirty": "Very Dirty Road"
    }
    road_class_str = road_class_map.get(road_class, road_class)
    print(f"Result: {road_class_str} (Confidence: {confidence:.2%})")
    
    # If clean road, we stop
    if road_class == "CleanRoad":
        print("Road is clean. No garbage detection necessary.")
        # Save output image with classification label
        cv2.putText(cv_image, f"Road: {road_class_str} ({confidence:.1%})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        out_path = os.path.join(output_dir, f"result_{os.path.basename(image_path)}")
        cv2.imwrite(out_path, cv_image)
        print(f"Saved visualization to: {out_path}")
        return {
            "road_status": road_class_str,
            "detections": []
        }
        
    # 3. Stage 2: Garbage Detection
    print("\n[Stage 2] Detecting garbage on road...")
    det_results = detector.predict(source=pil_image, conf=0.35, verbose=False)
    boxes = det_results[0].boxes
    
    if len(boxes) == 0:
        print("No garbage items detected by the object detector.")
        cv2.putText(cv_image, f"Road: {road_class_str} ({confidence:.1%})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
        cv2.putText(cv_image, "No trash items detected", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        out_path = os.path.join(output_dir, f"result_{os.path.basename(image_path)}")
        cv2.imwrite(out_path, cv_image)
        return {
            "road_status": road_class_str,
            "detections": []
        }
        
    print(f"Detected {len(boxes)} garbage item(s).")
    
    # 4. Stage 3: Zero-shot classification with CLIP
    print("\n[Stage 3] Classifying garbage types...")
    detections_summary = []
    
    for i, box in enumerate(boxes):
        # Extract coordinates (xyxy)
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        det_conf = box.conf[0].item()
        
        # Crop garbage item (ensure crop is within image boundaries)
        x1_crop = max(0, x1)
        y1_crop = max(0, y1)
        x2_crop = min(w_img, x2)
        y2_crop = min(h_img, y2)
        
        cropped_pil = pil_image.crop((x1_crop, y1_crop, x2_crop, y2_crop))
        
        # Run CLIP
        inputs = clip_processor(text=CLIP_PROMPTS, images=cropped_pil, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image  # image-text similarity score
            probs_clip = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
            
        best_idx = np.argmax(probs_clip)
        trash_type = TRASH_CLASSES[best_idx]
        trash_conf = probs_clip[best_idx]
        
        print(f"  Item {i+1}: Detected at [{x1}, {y1}, {x2}, {y2}]")
        print(f"          Detection Conf: {det_conf:.2%}")
        print(f"          Classified as: {trash_type} ({trash_conf:.2%})")
        
        detections_summary.append({
            "box": [x1, y1, x2, y2],
            "detection_confidence": det_conf,
            "trash_type": trash_type,
            "trash_confidence": trash_conf
        })
        
        # Draw bounding boxes and text
        color_map = {
            "plastic": (255, 0, 0),         # Blue
            "dry waste": (0, 255, 255),      # Yellow
            "wet waste": (0, 128, 0),        # Green
            "mixed waste": (0, 0, 255),      # Red
            "branches": (42, 42, 165),       # Brown
            "leaves": (0, 255, 0),           # Light Green
            "cement debris": (128, 128, 128) # Grey
        }
        color = color_map.get(trash_type, (255, 255, 255))
        
        cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)
        label = f"{trash_type} ({trash_conf:.1%})"
        
        # Bounding box label
        label_y = y1 - 10 if y1 - 10 > 20 else y1 + 20
        cv2.putText(cv_image, label, (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
    # Draw overall road status
    cv2.putText(cv_image, f"Road: {road_class_str} ({confidence:.1%})", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                
    # Save the output image
    out_path = os.path.join(output_dir, f"result_{os.path.basename(image_path)}")
    cv2.imwrite(out_path, cv_image)
    print(f"\nSaved visualization image to: {out_path}")
    
    # Summary report
    print("\nSummary Report:")
    print(f"Road status: {road_class_str}")
    unique_types = {}
    for det in detections_summary:
        t = det["trash_type"]
        unique_types[t] = unique_types.get(t, 0) + 1
    for t, count in unique_types.items():
        print(f" - {t.capitalize()}: {count} item(s) found")
        
    return {
        "road_status": road_class_str,
        "detections": detections_summary
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Road Trash Detection Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to input road image")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save results")
    args = parser.parse_args()
    
    try:
        classifier, detector, clip_model, clip_processor, device = load_pipeline()
        run_pipeline(args.image, classifier, detector, clip_model, clip_processor, device, args.output_dir)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
