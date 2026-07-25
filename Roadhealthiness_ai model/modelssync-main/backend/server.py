import os
import cv2
import numpy as np
import requests
import time
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from PIL import Image
from geopy.geocoders import Nominatim

IMAGE_DIR = "detections"

# Initialize Reverse Geocoder
geolocator = Nominatim(user_agent="road_health_ai_app")

os.makedirs(IMAGE_DIR, exist_ok=True)

app = FastAPI(title="Road Health AI Backend (Pothole)")
app.mount("/detections", StaticFiles(directory="detections"), name="detections")

# In-memory store for the dashboard feed
recent_potholes = []

# ---------------------------------------------------------
# 1. Load YOLOv8 Pothole Model (2-Class Detector)
# ---------------------------------------------------------
possible_paths = [
    os.path.join("pothole_detector", "two_class_model", "weights", "best.pt"),
    os.path.join("Roadhealthiness_ai model", "modelssync-main", "pothole_detector", "two_class_model", "weights", "best.pt"),
    os.path.join("runs", "detect", "pothole_detector", "two_class_model", "weights", "best.pt")
]
yolo_model_path = next((p for p in possible_paths if os.path.exists(p)), possible_paths[0])

print(f"Loading Main YOLO Road Health Model from: {yolo_model_path}")
pothole_model = YOLO(yolo_model_path)



# The temporary notification backend URL
EXTERNAL_BACKEND_URL = "http://127.0.0.1:5001/notify_pothole"

def send_notification(data):
    """Sends the data to the external backend in the background."""
    try:
        print(f"Sending notification to {EXTERNAL_BACKEND_URL}: {data}")
        response = requests.post(EXTERNAL_BACKEND_URL, json=data)
        print(f"External backend responded with status: {response.status_code}")
    except Exception as e:
        print(f"Failed to send notification: {e}")

@app.post("/detect")
async def detect_anomalies(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...)
):
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    frame_height, frame_width = frame.shape[:2]
    frame_area = frame_height * frame_width
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Reverse Geocoding to get exact address
    try:
        location_obj = geolocator.reverse(f"{latitude}, {longitude}", timeout=3)
        address = location_obj.address if location_obj else "Unknown Address"
    except Exception as e:
        print(f"Geocoding error: {e}")
        address = f"Lat: {latitude}, Lng: {longitude} (Address unavailable)"
        
    pothole_detections = []
    
    # ---------------------------
    # A. 2-CLASS POTHOLE & DAMAGED ROAD INFERENCE
    # ---------------------------
    results = pothole_model(frame, verbose=False)
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0].cpu().numpy())
            if conf < 0.35:
                continue
                
            xyxy = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0].cpu().numpy())
            class_map = {0: "Pothole", 1: "Damaged road", 2: "Debris", 3: "Abandoned Vehicle"}
            color_map = {0: (0, 255, 0), 1: (0, 0, 255), 2: (0, 165, 255), 3: (255, 0, 255)}
            cls_name = class_map.get(cls_id, f"Class_{cls_id}")
            color = color_map.get(cls_id, (0, 255, 0))

            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            bbox_area = (x2 - x1) * (y2 - y1)
            ratio = bbox_area / frame_area
            
            pothole_data = {
                "classification": cls_name,
                "confidence": round(conf, 3),
                "location": {"lat": latitude, "lng": longitude},
                "address": address,
                "area_ratio": round(ratio, 4),
                "timestamp": int(time.time())
            }
            pothole_detections.append(pothole_data)

    # ---------------------------
    # AGGREGATION & SAVING
    # ---------------------------
    if pothole_detections:
        filename_only = f"event_{int(time.time())}.jpg"
        filename = f"detections/{filename_only}"
        cv2.imwrite(filename, frame)
        
        for d in pothole_detections:
            d['image_url'] = f"/detections/{filename_only}"
            recent_potholes.insert(0, d)
            background_tasks.add_task(send_notification, d)
            
        recent_potholes[:] = recent_potholes[:50]

    return {
        "message": f"Processed frame. Detected {len(pothole_detections)} pothole(s).",
        "potholes": pothole_detections
    }

@app.get("/api/feed")
async def get_feed():
    return {
        "potholes": recent_potholes
    }

@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    with open("backend/dashboard.html", "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(
        content=content, 
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
