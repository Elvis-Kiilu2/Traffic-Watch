from flask import Flask, render_template, request, send_file
import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Loading YOLOv8-seg weights
model = YOLO('content/runs/segment/train/weights/best.pt')

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    track_history = defaultdict(lambda: [])

    #video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    #path to output video
    output_video_path = 'processed_video.mp4'

    # Defining the codec and create a VideoWriter object for MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    # Initialize variables for object counting and speed estimation
    last_frame_ids = set()
    # Taking fps as the frame rate of the video
    time_interval = 1 / fps

    # Assuming 1 pixel equals 1 meter
    pixel_to_meter = 1

    # 1 meter = 0.001 kilometer
    meter_to_km = 0.001

    text_color = (78, 78, 78)

    while cap.isOpened():
        # Reading a frame from the video
        success, frame = cap.read()

        if success:
            # Run bytetrack tracking on the frame
            results = model.track(frame, persist=True, tracker="bytetrack.yaml")

            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Initialize counts and speeds
            object_count = len(boxes)
            speeds_km_per_h = []

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Plot the tracks and calculate speeds
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 1:
                    # Calculating displacement
                    displacement = np.linalg.norm(np.array(track[-1]) - np.array(track[-2]))

                    # Converting displacement from pixels to kilometers
                    displacement_km = displacement * pixel_to_meter * meter_to_km

                    # Calculating speed in km/s
                    speed_km_per_s = displacement_km / time_interval

                    # Converting speed from km/s to km/h
                    speed_km_per_h = speed_km_per_s * 3600

                    # Displaying speed_km_per_h (km/h) for individual object
                    cv2.putText(annotated_frame, f'Speed: {speed_km_per_h:.2f} km/h', (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

                    # Drawing Lines the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 0, 0), thickness=2)

            # Display count on the annotated frame
            cv2.putText(annotated_frame, f'Count: {object_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Update last frame IDs
            current_frame_ids = set(track_ids)
            new_objects = current_frame_ids - last_frame_ids
            lost_objects = last_frame_ids - current_frame_ids
            object_count += len(new_objects) - len(lost_objects)

            # Write the annotated frame to the output video
            out.write(annotated_frame)

            # Update last frame IDs
            last_frame_ids = current_frame_ids

        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the output video
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return output_video_path

@app.route('/')
def index():
    return render_template('index.html')
    

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['video']  
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            return render_template('display_video.html', video_file=filename)
    return 'Error uploading file'


# Run Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5000)
