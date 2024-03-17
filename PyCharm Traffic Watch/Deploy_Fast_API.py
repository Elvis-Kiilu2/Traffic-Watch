from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import tempfile
from collections import defaultdict

app = FastAPI()

# Mount static directory for HTML files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load your YOLO model
model = YOLO('content/runs/segment/train/weights/best.pt')

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Save the uploaded video temporarily
    with tempfile.NamedTemporaryFile(delete=False) as temp_video:
        temp_video.write(await file.read())

    # Process the uploaded video
    results = process_video(temp_video.name)

    return results

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    # Initialize variables for object counting and speed estimation
    last_frame_ids = set()
    time_interval = 1 / cap.get(cv2.CAP_PROP_FPS)  # Time interval between consecutive frames in seconds
    pixel_to_meter = 10  # Since 1 Pixel = 0.00026458333333719 Meter
    meter_to_km = 0.001  # 1 meter = 0.001 kilometer
    text_color = (78, 78, 78)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object tracking
        results = model.track(frame, persist=True, tracker="bytetrack.yaml")

        # Extract bounding boxes and track IDs from the results
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Annotate frame with bounding boxes and speed
        annotated_frame = frame.copy()  # Create a copy of the original frame for annotation

        # Plot the tracks and calculate speeds
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box

            # Draw bounding box
            cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)

            # Calculate speed
            speed_km_per_h = calculate_speed(track_id, (x, y), last_frame_ids, pixel_to_meter, time_interval, meter_to_km)
            if speed_km_per_h is not None:
                # Display speed
                cv2.putText(annotated_frame, f'Speed: {speed_km_per_h:.2f} km/h', (int(x - w / 2), int(y - h / 2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

        # Display count on the annotated frame
        object_count = len(boxes)
        cv2.putText(annotated_frame, f'Count: {object_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Convert the annotated frame to JPEG format and encode to base64
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        img_str = 'data:image/jpeg;base64,' + str(base64.b64encode(buffer), 'utf-8')
        frames.append(img_str)

        # Update last frame IDs
        last_frame_ids = set(track_ids)

    # Release the video capture object
    cap.release()

    return frames

def calculate_speed(track_id, current_position, last_frame_ids, pixel_to_meter, time_interval, meter_to_km):
    # Initialize variables for speed calculation
    last_positions = defaultdict(lambda: None)
    fps = 30  # Assuming the video frame rate is 30 frames per second

    # Check if the object was present in the previous frame
    if track_id in last_frame_ids:
        last_position = last_positions[track_id]
        if last_position:
            # Calculate displacement
            displacement = np.linalg.norm(np.array(current_position) - np.array(last_position))
            # Calculate speed (in km/h)
            speed = displacement * fps * pixel_to_meter * meter_to_km * 3600

            # Update last position
            last_positions[track_id] = current_position

            return
