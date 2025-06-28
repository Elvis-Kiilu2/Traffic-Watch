from flask import Flask, render_template, request, send_file, flash, redirect
from werkzeug.utils import secure_filename
import os
import cv2, zipfile
import numpy as np
import math
from collections import defaultdict
from ultralytics import YOLO

app = Flask(__name__)

# Define the upload directory and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super_secret_key'

# Load the YOLOv8 model
#best_weights = r"/Traffic-Watch/FlawSense/seg_weights.zip"
model = YOLO(best_weights)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video(video_path, output_video_path):
    # Store the track history
    track_history = defaultdict(lambda: [])

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create a VideoWriter object for MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use the MP4 codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    # Initialize variables for object counting and speed estimation
    center_point = (w // 2, h)
    pixel_per_meter = 100
    last_frame_ids = set()
    last_frame_times = defaultdict(lambda: 0)
    total_object_count = 0  # Initialize total object count

    # Font parameters
    font_scale = 0.9
    font_thickness = 5

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Get current time
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.1)

            # Check if any objects are detected in the frame
            if results[0].boxes is not None:
                # Get the boxes and track IDs
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                # Plot the tracks and calculate distances
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 1:  # Calculate distance and speed when there are at least two points
                        # Calculate distance from center point
                        distance = math.sqrt((x - center_point[0]) ** 2 + (y - center_point[1]) ** 2) / pixel_per_meter

                        # Display distance information
                        cv2.putText(annotated_frame, f'Distance: {distance:.2f} m', (int(x), int(y) - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), font_thickness)

                        # Calculate time interval since last frame
                        last_frame_time = last_frame_times[track_id]
                        time_interval = current_time - last_frame_time

                        # Estimate speed in m/s
                        if time_interval != 0:  # Avoid division by zero
                            speed_m_per_s = distance / time_interval

                            # Convert speed from m/s to km/h
                            speed_km_per_h = speed_m_per_s * 5/18

                            # Display speed information
                            cv2.putText(annotated_frame, f'Speed: {speed_km_per_h:.2f} km/h', (int(x), int(y) + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

                    # Update last frame time for this track
                    last_frame_times[track_id] = current_time

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(211, 211, 211), thickness=2)

                # Update total object count
                total_object_count += len(boxes)

                # Display count on the annotated frame
                cv2.putText(annotated_frame, f'Total Count: {total_object_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness)

                # Write the annotated frame to the output video
                out.write(annotated_frame)
            else:
                # If no objects are detected, write the original frame to the output video
                out.write(frame)

        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the output video
    cap.release()
    out.release()
    cv2.destroyAllWindows()


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Create the uploads directory if it doesn't exist
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Process the uploaded video file
            output_video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.mp4')
            process_video(file_path, output_video_path)
            
            # Send the processed video file as a response
            return send_file(output_video_path, as_attachment=True)
    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)
