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




