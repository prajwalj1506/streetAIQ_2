from fastapi import FastAPI, Request
import uvicorn
from datetime import datetime

app = FastAPI(title="Temporary Notification Backend")

# In-memory storage for notifications so we can see them
notifications = []

@app.post("/notify_pothole")
async def receive_notification(request: Request):
    data = await request.json()
    
    # Add a timestamp
    data["received_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    notifications.append(data)
    
    print("\n" + "="*50)
    print("🚨 NEW POTHOLE NOTIFICATION RECEIVED! 🚨")
    print(f"Time: {data['received_at']}")
    print(f"Severity: {data.get('size', 'Unknown')}")
    print(f"Confidence: {data.get('confidence', 0) * 100}%")
    
    address = data.get('address', 'Unknown Address')
    loc = data.get('location', {})
    print(f"Location: {address} (Lat: {loc.get('lat')}, Lng: {loc.get('lng')})")
    
    # Generate the full clickable link to the image on the main AI server
    image_path = data.get('image_url', '')
    if image_path:
        print(f"Image Link: http://localhost:8080{image_path}")
        
    print("="*50 + "\n")
    
    return {"status": "success", "message": "Notification logged successfully!"}

@app.get("/view_notifications")
async def view_notifications():
    """Endpoint to view all received notifications"""
    return {"total_received": len(notifications), "history": notifications}

if __name__ == "__main__":
    print("Starting Temporary Notification Backend on port 5001...")
    uvicorn.run(app, host="0.0.0.0", port=5001)
