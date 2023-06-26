# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 15:54:06 2023

@author: nicko
"""

import cv2
import cv2.aruco as aruco
import numpy as np

# Set the parameters for ArUco markers
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()

# Define the IDs of the four ArUco markers on the corners of the paper
marker_ids = [0, 1, 2, 3]

# Define the size of the ArUco markers in meters
marker_size = 0.05  # Assuming all markers have the same size

# Read the image from an arbitrary file location
image_path = 'path_to_image.jpg'  # Replace with the actual file path
frame = cv2.imread(image_path)

# Convert the image to grayscale for ArUco marker detection
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect the ArUco markers in the image
corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

if ids is not None and len(ids) >= 4:
    # Check if all four marker IDs are present in the detected markers
    if all(marker_id in ids for marker_id in marker_ids):
        # Estimate the pose of the ArUco markers
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_size, cameraMatrix, distortionCoeffs)

        # Get the position of the small flat object relative to the markers
        # Replace the following lines with your own logic for object position calculation
        object_x = 0.0  # Replace with your calculation
        object_y = 0.0  # Replace with your calculation
        object_z = 0.0  # Replace with your calculation

        # Print the object position relative to the markers
        print("Object position (x, y, z):", object_x, object_y, object_z)

# Draw detected ArUco markers on the image
frame = aruco.drawDetectedMarkers(frame, corners, ids)

# Display the image
cv2.imshow("Image", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
