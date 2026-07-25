import os
import base64
import json
import time
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": "*"}})

# Config
IMAGE_DIR = "captured_images"
METADATA_FILE = os.path.join(IMAGE_DIR, "metadata.json")
os.makedirs(IMAGE_DIR, exist_ok=True)

# Load Keras Model
MODEL_PATH = "outputs/best_model.keras"

# Load labels dynamically
labels_file = "outputs/labels.txt"
if os.path.exists(labels_file):
    with open(labels_file, "r") as f:
        labels = [line.strip() for line in f.read().split("\n") if line.strip()]
else:
    labels = ["Clean", "Slightly_Dirty", "Very_Dirty", "Not_a_Road"]

label_map = {
    "Clean": "Clean Road",
    "Slightly_Dirty": "Slightly Dirty Road",
    "Very_Dirty": "Very Dirty Road",
    "Not_a_Road": "Not a Road"
}

print(f"Loading custom Keras Road Classifier from {MODEL_PATH}...")
classifier_model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# Load metadata helper
def load_metadata():
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

# Save metadata helper
def save_metadata(data):
    with open(METADATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

# Desktop HTML Dashboard
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Road Cleanliness Monitoring Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0f1015;
            --panel-bg: #161822;
            --border-color: #232738;
            --text-color: #f1f2f6;
            --primary: #00f2fe;
            --secondary: #4facfe;
            --clean-color: #00e676;
            --slightly-color: #ff9100;
            --very-color: #ff1744;
        }
        body {
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        header {
            width: 100%;
            max-width: 1200px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            padding-bottom: 15px;
            border-bottom: 2px solid var(--border-color);
        }
        h1 {
            font-weight: 800;
            margin: 0;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2rem;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            width: 100%;
            max-width: 1200px;
            margin-bottom: 40px;
        }
        .stat-card {
            background-color: var(--panel-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
            transition: transform 0.2s;
        }
        .stat-card:hover {
            transform: translateY(-2px);
        }
        .stat-card.clean { border-left: 5px solid var(--clean-color); }
        .stat-card.slightly { border-left: 5px solid var(--slightly-color); }
        .stat-card.very { border-left: 5px solid var(--very-color); }
        
        .stat-value {
            font-size: 2.2rem;
            font-weight: 800;
            margin: 5px 0 0 0;
        }
        .stat-label {
            font-size: 0.9rem;
            color: #8a8f9d;
            text-transform: uppercase;
            font-weight: 600;
        }
        .gallery-title {
            width: 100%;
            max-width: 1200px;
            font-size: 1.4rem;
            font-weight: 600;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .gallery-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 25px;
            width: 100%;
            max-width: 1200px;
        }
        .snapshot-card {
            background-color: var(--panel-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
            transition: all 0.3s ease;
            position: relative;
        }
        .snapshot-card:hover {
            transform: scale(1.02);
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        }
        .snapshot-img {
            width: 100%;
            aspect-ratio: 4/3;
            object-fit: cover;
            border-bottom: 1px solid var(--border-color);
        }
        .snapshot-info {
            padding: 15px;
        }
        .flag {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        .flag.clean { background-color: rgba(0, 230, 118, 0.15); color: var(--clean-color); }
        .flag.slightly { background-color: rgba(255, 145, 0, 0.15); color: var(--slightly-color); }
        .flag.very { background-color: rgba(255, 23, 68, 0.15); color: var(--very-color); }
        .flag.notaroad { background-color: rgba(138, 143, 157, 0.15); color: #8a8f9d; }
        
        .time-label {
            font-size: 0.8rem;
            color: #8a8f9d;
            margin-top: 5px;
            display: block;
        }
        .confidence-label {
            font-weight: 600;
            font-size: 1rem;
            margin: 0;
        }
        .empty-state {
            grid-column: 1 / -1;
            text-align: center;
            padding: 50px;
            color: #8a8f9d;
            font-size: 1.1rem;
        }
    </style>
</head>
<body>
    <header>
        <div>
            <h1>Road Cleanliness Analyzer</h1>
            <p style="margin: 5px 0 0 0; color: #8a8f9d;">Desktop Monitoring System (Live Feed)</p>
        </div>
        <div style="background-color: var(--panel-bg); border: 1px solid var(--border-color); padding: 8px 15px; border-radius: 20px; display: flex; align-items: center; gap: 8px;">
            <span style="display:inline-block; width:8px; height:8px; background-color:var(--clean-color); border-radius:50%; animation: pulse 1.5s infinite;"></span>
            <span style="font-size:0.9rem; font-weight:600;">ACTIVE CONNECTION</span>
        </div>
    </header>

    <div class="stats-grid">
        <div class="stat-card">
            <span class="stat-label">Total Captures</span>
            <span class="stat-value" id="val-total">0</span>
        </div>
        <div class="stat-card clean">
            <span class="stat-label">Clean Roads</span>
            <span class="stat-value" id="val-clean" style="color: var(--clean-color);">0</span>
        </div>
        <div class="stat-card slightly">
            <span class="stat-label">Slightly Dirty</span>
            <span class="stat-value" id="val-slightly" style="color: var(--slightly-color);">0</span>
        </div>
        <div class="stat-card very">
            <span class="stat-label">Very Dirty</span>
            <span class="stat-value" id="val-very" style="color: var(--very-color);">0</span>
        </div>
    </div>

    <div class="gallery-title">
        <span>Captured Road Snapshots</span>
        <span style="font-size: 0.9rem; font-weight: 400; color: #8a8f9d;">Auto-updating...</span>
    </div>

    <div class="gallery-grid" id="gallery">
        <div class="empty-state">No snapshots captured yet. Access /mobile on your phone to capture images!</div>
    </div>

    <script>
        let lastCount = -1;

        async function fetchSnapshots() {
            try {
                const response = await fetch('/api/snapshots');
                const data = await response.json();
                
                // Update Stats
                document.getElementById('val-total').innerText = data.stats.total;
                document.getElementById('val-clean').innerText = data.stats.Clean;
                document.getElementById('val-slightly').innerText = data.stats.Slightly_Dirty;
                document.getElementById('val-very').innerText = data.stats.Very_Dirty;
                
                // Only render if count has changed to prevent page flickering
                if (data.snapshots.length !== lastCount) {
                    lastCount = data.snapshots.length;
                    const gallery = document.getElementById('gallery');
                    
                    if (data.snapshots.length === 0) {
                        gallery.innerHTML = '<div class="empty-state">No snapshots captured yet. Access /mobile on your phone to capture images!</div>';
                        return;
                    }
                    
                    gallery.innerHTML = '';
                    data.snapshots.forEach(snap => {
                        const classClean = snap.class.toLowerCase().replace('_', '');
                        const card = document.createElement('div');
                        card.className = 'snapshot-card';
                        card.innerHTML = `
                            <img src="${snap.image_url}" class="snapshot-img" alt="Captured Road">
                            <div class="snapshot-info">
                                <span class="flag ${classClean}">${snap.display_class}</span>
                                <h3 class="confidence-label">Confidence: ${(snap.confidence * 100).toFixed(1)}%</h3>
                                <span class="time-label">${snap.timestamp}</span>
                            </div>
                        `;
                        gallery.appendChild(card);
                    });
                }
            } catch (err) {
                console.error("Error fetching snapshots:", err);
            }
        }

        // Poll every 1.5 seconds for instant dashboard updates
        setInterval(fetchSnapshots, 1500);
        fetchSnapshots();
    </script>
    
    <style>
        @keyframes pulse {
            0% { transform: scale(0.95); opacity: 0.5; }
            50% { transform: scale(1.05); opacity: 1; }
            100% { transform: scale(0.95); opacity: 0.5; }
        }
    </style>
</body>
</html>
"""

# Mobile HTML Client
MOBILE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Mobile Capture Client</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Outfit', sans-serif;
            margin: 0;
            padding: 15px;
            background-color: #0f1015;
            color: #f1f2f6;
            display: flex;
            flex-direction: column;
            align-items: center;
            height: 100vh;
            box-sizing: border-box;
            overflow: hidden;
        }
        h1 {
            font-size: 1.2rem;
            margin: 5px 0 15px 0;
            background: linear-gradient(135deg, #00f2fe, #4facfe);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }
        #camera-panel {
            position: relative;
            width: 100%;
            flex-grow: 1;
            background-color: #000;
            border-radius: 20px;
            overflow: hidden;
            border: 2px solid #232738;
            box-shadow: 0 8px 30px rgba(0,0,0,0.5);
            margin-bottom: 20px;
        }
        video {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        #capture-btn {
            width: 75px;
            height: 75px;
            background-color: #ffffff;
            border-radius: 50%;
            border: 6px solid #232738;
            outline: none;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 20px;
        }
        #capture-btn:active {
            transform: scale(0.9);
            background-color: #00f2fe;
        }
        #result-panel {
            position: absolute;
            bottom: 20px;
            left: 20px;
            right: 20px;
            background-color: rgba(22, 24, 34, 0.9);
            border: 1px solid #232738;
            border-radius: 15px;
            padding: 15px;
            text-align: center;
            z-index: 100;
            backdrop-filter: blur(10px);
            display: none;
            animation: slideUp 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }
        .spinner {
            border: 4px solid rgba(255, 255, 255, 0.1);
            width: 30px;
            height: 30px;
            border-radius: 50%;
            border-left-color: #00f2fe;
            animation: spin 1s linear infinite;
            display: inline-block;
            vertical-align: middle;
            margin-right: 10px;
        }
        .result-class {
            font-size: 1.3rem;
            font-weight: 800;
            margin-bottom: 5px;
        }
        .result-conf {
            font-size: 0.9rem;
            color: #8a8f9d;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        @keyframes slideUp {
            from { transform: translateY(50px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
    </style>
</head>
<body>
    <h1>ROAD SNAPSHOT CAPTURE</h1>
    
    <div id="camera-panel">
        <video id="video" autoplay playsinline></video>
        <div id="result-panel">
            <div id="loading">
                <div class="spinner"></div>
                <span>Analyzing snapshot...</span>
            </div>
            <div id="success" style="display: none;">
                <div class="result-class" id="res-class">CLEAN ROAD</div>
                <div class="result-conf" id="res-conf">Confidence: 98%</div>
            </div>
        </div>
    </div>
    
    <button id="capture-btn"></button>
    <canvas id="canvas" style="display: none;"></canvas>

    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const captureBtn = document.getElementById('capture-btn');
        const resultPanel = document.getElementById('result-panel');
        const loadingDiv = document.getElementById('loading');
        const successDiv = document.getElementById('success');
        const resClass = document.getElementById('res-class');
        const resConf = document.getElementById('res-conf');

        // Start camera
        async function startCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: { ideal: "environment" } },
                    audio: false
                });
                video.srcObject = stream;
            } catch (err) {
                console.error("Camera error: ", err);
                alert("Camera Access Error. Please check permissions and use HTTP (or Local Hotspot).");
            }
        }

        // Capture snapshot and send to server
        captureBtn.addEventListener('click', async () => {
            // Draw frame to canvas
            const ctx = canvas.getContext('2d');
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 480;
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            // Show loading panel
            resultPanel.style.display = 'block';
            loadingDiv.style.display = 'block';
            successDiv.style.display = 'none';
            captureBtn.disabled = true;

            const base64Image = canvas.toDataURL('image/jpeg', 0.85);

            try {
                const response = await fetch('/upload_snapshot', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: base64Image })
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    resClass.innerText = result.display_class.toUpperCase();
                    resConf.innerText = `Confidence: ${(result.confidence * 100).toFixed(1)}%`;
                    
                    // Set color based on classification
                    if (result.class === "Clean") {
                        resClass.style.color = "#00e676";
                    } else if (result.class === "Slightly_Dirty") {
                        resClass.style.color = "#ff9100";
                    } else if (result.class === "Very_Dirty") {
                        resClass.style.color = "#ff1744";
                    } else {
                        resClass.style.color = "#8a8f9d"; // Gray for Not a Road
                    }
                    
                    loadingDiv.style.display = 'none';
                    successDiv.style.display = 'block';
                    
                    // Hide panel after 3.5 seconds
                    setTimeout(() => {
                        resultPanel.style.display = 'none';
                        captureBtn.disabled = false;
                    }, 3500);
                } else {
                    alert("Error analyzing: " + result.error);
                    resultPanel.style.display = 'none';
                    captureBtn.disabled = false;
                }
            } catch (err) {
                console.error("Upload error:", err);
                alert("Upload failed. Make sure you are connected to the laptop hotspot.");
                resultPanel.style.display = 'none';
                captureBtn.disabled = false;
            }
        });

        startCamera();
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    return render_template_string(DASHBOARD_TEMPLATE)

@app.route('/mobile')
def mobile():
    return render_template_string(MOBILE_TEMPLATE)

# Serves saved images to the web dashboard
@app.route('/images/<filename>')
def serve_image(filename):
    from flask import send_from_directory
    return send_from_directory(IMAGE_DIR, filename)

# API endpoint to retrieve snapshots list and statistics
@app.route('/api/snapshots')
def api_snapshots():
    metadata = load_metadata()
    
    # Sort with newest first
    sorted_snapshots = sorted(metadata, key=lambda x: x['timestamp'], reverse=True)
    
    # Calculate stats
    stats = {"total": len(metadata), "Clean": 0, "Slightly_Dirty": 0, "Very_Dirty": 0}
    for item in metadata:
        cls = item['class']
        if cls in stats:
            stats[cls] += 1
            
    return jsonify({
        "snapshots": sorted_snapshots,
        "stats": stats
    })

# API endpoint for phone to upload snapshot
@app.route('/upload_snapshot', methods=['POST'])
def upload_snapshot():
    try:
        data = request.get_json()
        image_b64 = data['image'].split(',')[1]
        
        # Decode base64 bytes
        image_bytes = base64.b64decode(image_b64)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        cv_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Preprocess frame for TFLite road classifier (224x224, float32, BGR to RGB)
        rgb_frame = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(rgb_frame, (224, 224))
        input_data = np.expand_dims(np.array(resized_frame, dtype=np.float32), axis=0)

        # Run Keras inference
        predictions = classifier_model.predict(input_data, verbose=0)[0]

        # Calculate prediction details
        best_idx = np.argmax(predictions)
        pred_class = labels[best_idx]
        confidence = float(predictions[best_idx])
        display_class = label_map[pred_class]
        
        # Save snapshot image file
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snap_{timestamp_str}_{pred_class}.jpg"
        save_path = os.path.join(IMAGE_DIR, filename)
        cv2.imwrite(save_path, cv_frame)
        
        # Save metadata entry
        metadata = load_metadata()
        new_entry = {
            "filename": filename,
            "image_url": f"/images/{filename}",
            "class": pred_class,
            "display_class": display_class,
            "confidence": confidence,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        metadata.append(new_entry)
        save_metadata(metadata)
        
        print(f"Snapshot received: {pred_class} ({confidence:.1%}) -> Saved to {save_path}")
        
        return jsonify(new_entry)
        
    except Exception as e:
        print(f"Error saving snapshot: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/telemetry', methods=['POST', 'OPTIONS'])
def telemetry():
    return jsonify({"status": "success", "message": "Telemetry received"})

@app.route('/api/upload_chunk', methods=['POST', 'OPTIONS'])
def upload_chunk():
    return jsonify({"status": "success", "message": "Chunk received"})

if __name__ == "__main__":
    # Start secure public internet tunnel using pinggy
    public_url = None
    try:
        import pinggy
        print("\nStarting secure public HTTPS tunnel for mobile client...")
        tunnel = pinggy.start_tunnel(forwardto=5173)
        
        # Extract the public HTTPS URL from Pinggy
        for url in tunnel.urls:
            if url.startswith("https://"):
                public_url = url
                break
                
        if public_url:
            print("\n" + "="*80)
            print("                      PUBLIC INTERNET SHAREABLE LINK")
            print(" Share this secure HTTPS link with your teacher / friends to test on their phones:")
            print(f" {public_url}/mobile")
            print("="*80 + "\n")
        else:
            print("Warning: Could not extract public HTTPS URL from Pinggy.")
    except Exception as e:
        print(f"Could not start Pinggy tunnel: {e}")
        print("You can still use your local Wi-Fi network link.")

    # Run local Flask server on HTTP (Pinggy handles the HTTPS encryption externally)
    app.run(host="0.0.0.0", port=5000, debug=False)
