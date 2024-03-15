from flask import Flask, request, render_template, jsonify, Response
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import tempfile
from collections import defaultdict

app = Flask(__name__)

# Load your YOLO model
model = YOLO('content/runs/segment/train/weights/best.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the uploaded video temporarily
    with tempfile.NamedTemporaryFile(delete=False) as temp_video:
        temp_video.write(file.read())

    # Process the uploaded video
    processed_video = process_video(temp_video.name)

    # Return the processed video data directly
    return Response(open(processed_video, 'rb'), mimetype='video/mp4')


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    track_history = defaultdict(lambda: [])

    # Get video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create a VideoWriter object for MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use the MP4 codec
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (w, h))

    # Initialize variables for object counting and speed estimation
    last_frame_ids = set()
    time_interval = 1 / fps  # Time interval between consecutive frames in seconds
    pixel_to_meter = 10  # Since 1 Pixel = 0.00026458333333719 Meter
    meter_to_km = 0.001  # 1 meter = 0.001 kilometer
    text_color = (78, 78, 78)

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            # Perform object tracking
            results = model.track(frame, persist=True, tracker="bytetrack.yaml")

            # Extract bounding boxes and track IDs from the results
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Annotate frame with bounding boxes and speed
            annotated_frame = frame.copy()

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]  # Access the track using track_id
                track.append((float(x), float(y)))
                # Draw bounding box
                cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),
                            (0, 255, 0), 2)

                # Calculate speed
                speed_km_per_h = calculate_speed(track_id, (x, y), last_frame_ids, pixel_to_meter, time_interval,
                                                meter_to_km)
                if speed_km_per_h is not None:
                    # Display speed
                    cv2.putText(annotated_frame, f'Speed: {speed_km_per_h:.2f} km/h',
                                (int(x - w / 2), int(y - h / 2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 0, 0), thickness=2)


            # Display count on the annotated frame
            object_count = len(boxes)
            cv2.putText(annotated_frame, f'Count: {object_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)

            # Write the annotated frame to the output video
            out.write(annotated_frame)

            # Update last frame IDs
            last_frame_ids = set(track_ids)
        else:
            break

    # Release the video capture and video writer objects
    cap.release()
    out.release()

    return 'output.mp4'



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

            return speed

    return None


if __name__ == '__main__':
    app.run()
