import zipfile
import os
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

# Specify the path to the zip file and the extraction directory
zip_file_path = r'C:\Users\KIILU\Desktop\CS\year 3\Y3Sem2\Project\PyCharm Traffic Watch\seg_weights.zip'
extraction_dir = r'C:\Users\KIILU\Desktop\CS\year 3\Y3Sem2\Project\PyCharm Traffic Watch'

# Create a ZipFile object
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Extract all the contents to the extraction directory
    zip_ref.extractall(extraction_dir)

# Print a message indicating that the extraction is complete
print("Extraction completed successfully.")




# Load the YOLOv8 model
model = YOLO('content/runs/segment/train/weights/best.pt')

# Open the video file
video_path = r"C:\Users\KIILU\Desktop\CS\year 3\Y3Sem2\Project\PyCharm Traffic Watch\traffic_-_27260 (540p).mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

# Get video properties
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the output video file path with .mp4 extension
output_video_path = 'ftestoutput_video.mp4'

# Define the codec and create a VideoWriter object for MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use the MP4 codec
out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))


# Initialize variables for object counting and speed estimation
last_frame_ids = set()

# Assuming fps is the frame rate of the video
time_interval = 1 / fps  # Time interval between consecutive frames in seconds

# Assuming pixel_to_meter is a conversion factor from pixels to meters
pixel_to_meter = 1 # Since 1 Pixel = 0.00026458333333719 Meter

# Assuming meter_to_km is a conversion factor from meters to kilometers
meter_to_km = 0.001  # 1 meter = 0.001 kilometer

text_color = (78, 78, 78)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run tracking on the frame, persisting tracks between frames
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
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 1:  # Calculate speed when there are at least two points
                # Calculate displacement
                displacement = np.linalg.norm(np.array(track[-1]) - np.array(track[-2]))

                # Convert displacement from pixels to kilometers
                displacement_km = displacement * pixel_to_meter * meter_to_km

                # Calculate speed in km/s
                speed_km_per_s = displacement_km / time_interval

                # Convert speed from km/s to km/h
                speed_km_per_h = speed_km_per_s * 3600

                # Display speed_km_per_h (km/h) for individual object
                cv2.putText(annotated_frame, f'Speed: {speed_km_per_h:.2f} km/h', (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)


                # Draw the tracking lines
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

