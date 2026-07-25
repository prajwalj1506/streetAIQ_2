import os
import base64
import json
import torch
import cv2
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import requests
from flask import Flask, request, jsonify, render_template_string, Response
from flask_cors import CORS
from inference import load_pipeline, TRASH_CLASSES, CLIP_PROMPTS

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

classifier, detector, clip_model, clip_processor, device = load_pipeline()

print("AI Pipeline loaded successfully!")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Road Trash Detector - Mobile Test Client</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #121212;
            color: #ffffff;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-start;
            height: 100vh;
            overflow: hidden;
        }
        h1 {
            font-size: 1.2rem;
            margin: 10px 0;
            color: #00ffcc;
            text-shadow: 0 0 10px rgba(0, 255, 204, 0.3);
        }
        #container {
            position: relative;
            width: 100%;
            max-width: 500px;
            aspect-ratio: 3/4;
            background-color: #000000;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }
        video {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        canvas {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 10;
        }
        #status-panel {
            position: absolute;
            top: 10px;
            left: 10px;
            right: 10px;
            background-color: rgba(0, 0, 0, 0.7);
            padding: 10px;
            border-radius: 8px;
            z-index: 20;
            border: 1px solid rgba(255,255,255,0.1);
        }
        #road-status {
            font-weight: bold;
            font-size: 1.1rem;
            color: #00ff00;
        }
        #fps-display {
            font-size: 0.8rem;
            color: #888;
            margin-top: 2px;
        }
        #report-panel {
            width: 90%;
            max-width: 480px;
            margin-top: 15px;
            background-color: #1e1e1e;
            padding: 12px;
            border-radius: 8px;
            flex-grow: 1;
            overflow-y: auto;
            border: 1px solid #333;
            margin-bottom: 10px;
        }
        .report-title {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #00ffcc;
            margin-bottom: 8px;
            border-bottom: 1px solid #333;
            padding-bottom: 4px;
        }
        .report-item {
            display: flex;
            justify-content: space-between;
            margin: 4px 0;
            font-size: 0.9rem;
        }
        #controls {
            margin: 10px 0;
            display: flex;
            gap: 10px;
        }
        button {
            background-color: #00ffcc;
            color: #000000;
            border: none;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.2s;
        }
        button:active {
            transform: scale(0.95);
        }
        #stop-btn {
            background-color: #ff3366;
            color: white;
        }
    </style>
</head>
<body>
    <h1>ROAD TRASH DETECTOR (GPU)</h1>
    
    <div id="container">
        <video id="video" autoplay playsinline muted></video>
        <canvas id="canvas"></canvas>
        
        <div id="status-panel">
            <div id="road-status">Initializing Camera...</div>
            <div id="fps-display">Latency: -- ms | Processing: -- FPS</div>
        </div>
    </div>
    
    <div id="controls">
        <button id="start-btn">Start Stream</button>
        <button id="stop-btn">Stop</button>
    </div>
    
    <div id="report-panel">
        <div class="report-title">Active Trash Counts</div>
        <div id="trash-list">
            <div style="color: #666; font-style: italic;">No active detections</div>
        </div>
    </div>

    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const roadStatus = document.getElementById('road-status');
        const fpsDisplay = document.getElementById('fps-display');
        const trashList = document.getElementById('trash-list');
        const startBtn = document.getElementById('start-btn');
        const stopBtn = document.getElementById('stop-btn');
        
        let streaming = false;
        let streamInterval = null;
        let lastFrameTime = Date.now();
        
        // Bounding box colors
        const colors = {
            "plastic": "#0066ff",
            "dry waste": "#ffff00",
            "wet waste": "#008000",
            "mixed waste": "#ff0000",
            "branches": "#a52a2a",
            "leaves": "#00ff00",
            "cement debris": "#808080"
        };
        
        async function startCamera() {
            try {
                const constraints = {
                    video: {
                        facingMode: { ideal: "environment" },
                        width: { ideal: 640 },
                        height: { ideal: 480 }
                    },
                    audio: false
                };
                const stream = await navigator.mediaDevices.getUserMedia(constraints);
                video.srcObject = stream;
                
                video.onloadedmetadata = () => {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    roadStatus.textContent = "Camera Ready. Press Start Stream.";
                    roadStatus.style.color = "#00ffcc";
                };
            } catch (err) {
                console.error("Error accessing camera: ", err);
                roadStatus.textContent = "Camera Access Error";
                roadStatus.style.color = "#ff3366";
            }
        }
        
        function sendFrame() {
            if (!streaming) return;
            
            // Draw video frame to hidden canvas
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = canvas.width;
            tempCanvas.height = canvas.height;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
            
            // Get JPEG Base64
            const dataUrl = tempCanvas.toDataURL('image/jpeg', 0.7); // 0.7 quality to reduce upload bandwidth
            const sendTime = Date.now();
            
            fetch('/process_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: dataUrl })
            })
            .then(res => res.json())
            .then(data => {
                const latency = Date.now() - sendTime;
                const processingFps = (1000 / latency).toFixed(1);
                fpsDisplay.textContent = `Latency: ${latency} ms | ${processingFps} FPS`;
                
                drawResults(data);
                updateReport(data);
            })
            .catch(err => {
                console.error("Error sending frame: ", err);
            });
        }
        
        function drawResults(data) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // 1. Update Road status top bar
            roadStatus.textContent = `ROAD: ${data.road_status} (${(data.road_confidence * 100).toFixed(0)}%)`;
            if (data.road_status === "Clean Road") {
                roadStatus.style.color = "#00ff00";
            } else if (data.road_status === "Slightly Dirty Road") {
                roadStatus.style.color = "#ff9900";
            } else {
                roadStatus.style.color = "#ff3366";
            }
            
            // 2. Draw boxes
            data.detections.forEach(det => {
                const [x1, y1, x2, y2] = det.box;
                const color = colors[det.trash_type] || "#ffffff";
                
                // Draw box
                ctx.strokeStyle = color;
                ctx.lineWidth = 3;
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                
                // Draw label background
                ctx.fillStyle = color;
                ctx.font = "bold 14px sans-serif";
                const label = `${det.trash_type} (${(det.trash_confidence * 100).toFixed(0)}%)`;
                const textWidth = ctx.measureText(label).width;
                ctx.fillRect(x1, y1 - 22 > 0 ? y1 - 22 : y1, textWidth + 10, 20);
                
                // Draw label text
                ctx.fillStyle = "#000000";
                ctx.fillText(label, x1 + 5, y1 - 22 > 0 ? y1 - 7 : y1 + 15);
            });
        }
        
        function updateReport(data) {
            trashList.innerHTML = '';
            
            if (data.detections.length === 0) {
                trashList.innerHTML = '<div style="color: #666; font-style: italic;">No active detections</div>';
                return;
            }
            
            const counts = {};
            data.detections.forEach(det => {
                counts[det.trash_type] = (counts[det.trash_type] || 0) + 1;
            });
            
            Object.keys(counts).forEach(type => {
                const color = colors[type] || "#ffffff";
                const item = document.createElement('div');
                item.className = 'report-item';
                item.innerHTML = `
                    <span style="color: ${color}; font-weight: bold;">● ${type.toUpperCase()}</span>
                    <span>${counts[type]} item(s)</span>
                `;
                trashList.appendChild(item);
            });
        }
        
        startBtn.addEventListener('click', () => {
            if (streaming) return;
            streaming = true;
            startBtn.disabled = true;
            stopBtn.disabled = false;
            // Send frame every 200ms (~5 FPS upload loop)
            streamInterval = setInterval(sendFrame, 200);
        });
        
        stopBtn.addEventListener('click', () => {
            if (!streaming) return;
            streaming = false;
            startBtn.disabled = false;
            stopBtn.disabled = true;
            clearInterval(streamInterval);
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            roadStatus.textContent = "Streaming Paused";
            roadStatus.style.color = "#00ffcc";
        });
        
        // Start camera setup
        startCamera();
    </script>
</body>
</html>
"""

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

import time
from datetime import datetime
import uuid

# In-memory store for the dashboard feed
recent_snapshots = []
for label in ["Clean", "Not_a_Road", "Slightly_Dirty", "Very_Dirty"]:
    os.makedirs(f"static/images/{label}", exist_ok=True)

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

from flask import send_from_directory
from flask_cors import cross_origin

@app.route('/images/<path:filename>')
@cross_origin()
def serve_image(filename):
    """Serve images with CORS headers applied by flask_cors"""
    return send_from_directory('static/images', filename)

@app.route('/api/proxy_image')
def proxy_image():
    """Proxy Firebase Storage images to bypass Flutter Web CORS restrictions."""
    url = request.args.get('url')
    if not url:
        return jsonify({"error": "Missing url parameter"}), 400
        
    try:
        resp = requests.get(url)
        return Response(resp.content, content_type=resp.headers.get('Content-Type', 'image/jpeg'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/snapshots', methods=['GET'])
def get_snapshots():
    return jsonify({"snapshots": recent_snapshots})

@app.route('/process_frame', methods=['POST'])
def process_frame_route():
    try:
        data = request.json
        image_data = data['image']
        lat = data.get('lat', 0.0)
        lng = data.get('lng', 0.0)
        vehicle_number = data.get('vehicle_number', 'V-102')
        ward = data.get('ward', 'Ward A')
        session_id = data.get('session_id')
        point_type = data.get('point_type')
        
        image_b64 = image_data.split(',')[1] if ',' in image_data else image_data
        image_bytes = base64.b64decode(image_b64)
        
        import io
        from PIL import ImageOps
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        pil_image = ImageOps.exif_transpose(pil_image)
        
        rgb_frame = np.array(pil_image)
        cv_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        h_img, w_img, _ = cv_frame.shape
        
        img_resized = pil_image.resize((224, 224))
        input_data = np.expand_dims(np.array(img_resized, dtype=np.float32), axis=0)

        preds = classifier.predict(input_data, verbose=0)
        best_idx = np.argmax(preds[0])
        labels = ["Clean", "Not_a_Road", "Slightly_Dirty", "Very_Dirty"]
        road_class = labels[best_idx]
        confidence = float(preds[0][best_idx])
        
        road_class_map = {
            "Clean": "Clean Road",
            "Slightly_Dirty": "Slightly Dirty Road",
            "Very_Dirty": "Very Dirty Road"
        }
        road_class_str = road_class_map.get(road_class, road_class)
        
        detections = []
        
        if road_class != "Clean":
            det_results = detector.predict(source=pil_image, conf=0.20, verbose=False)
            boxes = det_results[0].boxes
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1_crop, y1_crop = max(0, x1), max(0, y1)
                x2_crop, y2_crop = min(w_img, x2), min(h_img, y2)
                
                if x2_crop > x1_crop and y2_crop > y1_crop:
                    cropped_pil = pil_image.crop((x1_crop, y1_crop, x2_crop, y2_crop))
                    inputs = clip_processor(text=CLIP_PROMPTS, images=cropped_pil, return_tensors="pt", padding=True).to(device)
                    with torch.no_grad():
                        outputs = clip_model(**inputs)
                        probs_clip = outputs.logits_per_image.softmax(dim=-1).cpu().numpy()[0]
                        
                    best_clip_idx = np.argmax(probs_clip)
                    trash_type = TRASH_CLASSES[best_clip_idx]
                    trash_conf = float(probs_clip[best_clip_idx])
                    
                    detections.append({
                        "box": [x1, y1, x2, y2],
                        "trash_type": trash_type,
                        "trash_confidence": trash_conf
                    })
                    
        annotated_frame = draw_overlays(cv_frame.copy(), road_class_str, confidence, detections)
        cv2.imshow("Live Phone Stream (Laptop View)", annotated_frame)
        cv2.waitKey(1)
        
        # Save frame and add to recent_snapshots for Dashboard
        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join("static/images", road_class, filename)
        cv2.imwrite(filepath, annotated_frame)
        
        # Convert annotated frame back to base64 to send to flutter
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        annotated_b64 = base64.b64encode(buffer).decode('utf-8')
        
        snapshot = {
            "filename": filename,
            "image_url": f"/images/{road_class}/{filename}",
            "class": road_class,
            "display_class": road_class_str,
            "confidence": float(confidence),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "lat": lat,
            "lng": lng,
            "vehicle_number": vehicle_number,
            "ward": ward,
            "session_id": session_id,
            "point_type": point_type
        }
        
        recent_snapshots.insert(0, snapshot)
        if len(recent_snapshots) > 50:
            oldest = recent_snapshots.pop()
            try:
                os.remove(os.path.join("static/images", oldest["class"], oldest["filename"]))
            except:
                pass
        
        return jsonify({
            "road_status": road_class_str,
            "road_confidence": float(confidence),
            "detections": detections,
            "annotated_image": annotated_b64,
            "image_url": snapshot["image_url"]
        })
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
