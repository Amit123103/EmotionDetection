from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
import base64
import json
from emotion_model import EmotionDetector

app = FastAPI(title="Emotion Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
detector = EmotionDetector()

# Store latest speech text for context
latest_speech_text = ""

@app.get("/")
def read_root():
    return {"status": "active", "message": "Emotion & Age AI is running"}

@app.websocket("/ws/detect")
async def websocket_endpoint(websocket: WebSocket):
    global latest_speech_text
    await websocket.accept()
    print("Client connected")
    try:
        while True:
            # Receive message
            message = await websocket.receive_text()
            
            # Try to parse as JSON (New Protocol)
            try:
                data = json.loads(message)
                
                # Handle different message types
                if data.get('type') == 'audio_text':
                    latest_speech_text = data.get('text', '')
                    # print(f"Speech Received: {latest_speech_text}")
                    continue # Don't process as image
                
                if data.get('type') == 'image':
                    encoded_image = data.get('data')
                else:
                    # Fallback for old protocol or weird data
                    continue

            except json.JSONDecodeError:
                # Fallback: Assume it's just the raw base64 string (Old Protocol compaibility)
                if "," in message:
                    _, encoded_image = message.split(",", 1)
                else:
                    encoded_image = message

            # Decode image
            image_bytes = base64.b64decode(encoded_image)
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # Detect emotion & Age
            # We pass the latest speech text to generating summary
            results = detector.detect_all(frame, latest_speech_text)
            
            # Send results back
            await websocket.send_json({"results": results})

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
