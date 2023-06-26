 

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 12:29:23 2023

@author: nicko
"""

import cv2 as cv
import numpy as np
import glob
import csv
import sys

CHESSBOARD_WIDTH = 10  # 9
CHESSBOARD_HEIGHT = 7  # 6
RESIZE_IMAGES = True
SCALING = 0.1

cv.aruco.DetectorParameters.minMarkerPerimeterRate=0.01
cv.aruco.DetectorParameters.minMarkerDistanceRate=0.001
cv.aruco.DetectorParameters.minCornerDistanceRate=0.001

def resize_image(img):
    # adapted from from: https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
    width = int(img.shape[1] * SCALING)
    height = int(img.shape[0] * SCALING)
    dim = (width, height)

    # resize image
    return cv.resize(img, dim, interpolation=cv.INTER_AREA)


# Define the dimensions of the Charuco board
# Adjust these values according to your specific board
# You can specify the number of squares and the size of each square
charuco_rows = 6
charuco_cols = 7
square_length = 0.03329 # Length of each square in meters
marker_length = 0.02679  # Length of the marker in meters

# Create a Charuco board object
dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_50)
charuco_board = cv.aruco.CharucoBoard_create(
    charuco_cols, charuco_rows, square_length, marker_length, dictionary
)

# Create arrays to store object points and image points
object_points = []  # 3D points in the world coordinate system
image_points = []  # 2D points in the image plane
# Load the images for calibration
image_paths = glob.glob("Image*.jpeg") # Add your image paths here

for path in image_paths:
    # Read the image
    frame = cv.imread(path)
    frame = resize_image(frame) 

    # Detect markers and corners on the Charuco board
    corners, ids, _ = cv.aruco.detectMarkers(frame, dictionary)
    _, charuco_corners, charuco_ids = cv.aruco.interpolateCornersCharuco(
        corners, ids, frame, charuco_board
    )

    # If any Charuco corners are detected, add them to the calibration data
    if charuco_corners is not None and charuco_ids is not None:
        object_points.append(charuco_board.chessboardCorners)
        image_points.append(charuco_corners)

        # Draw detected markers and corners
        frame = cv.aruco.drawDetectedMarkers(frame, corners)
        frame = cv.aruco.drawDetectedCornersCharuco(
            frame, charuco_corners, charuco_ids
        )

    cv.imshow("Calibration", frame)
    print(len(image_points), sys.getsizeof(image_points))
    cv.waitKey(0)

# destroy OpenCV window
cv.destroyAllWindows()

# camera calibration using the collected data
ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv.calibrateCamera(
    object_points, image_points, frame.shape[::-1][1:], None, None
)

# Print the camera matrix and distortion coefficients
print("Camera Matrix:")
print(camera_matrix)
print("\nDistortion Coefficients:")
print(distortion_coeffs)

# Print the relative camera position and rotation
print("\nRelative Camera Position and Rotation:")
for rvec, tvec in zip(rvecs, tvecs):
    print("Rotation Vector:")
    print(rvec)
    print("Translation Vector:")
    print(tvec)
    print()

# Write the calibration data to a CSV file
csv_file = "calibration_data.csv"
header = ["Camera Matrix", "Distortion Coefficients", "Rotation Vector", "Translation Vector"]
data = [camera_matrix.flatten(), distortion_coeffs.flatten()]

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerow(data)
    for rvec, tvec in zip(rvecs, tvecs):
        writer.writerow(["", "", rvec.flatten(), tvec.flatten()])

print("Calibration data has been written to:", csv_file)