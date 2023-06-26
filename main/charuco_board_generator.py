# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 18:29:35 2023

@author: nicko
"""

import cv2
import cv2.aruco as aruco

# Define the dimensions of the ChArUco board
num_squares_x = 6  # Number of squares along the x-axis
num_squares_y = 7  # Number of squares along the y-axis
square_length = 100  # Length of each square in pixels
marker_length = 80  # Length of the marker in pixels
board = aruco.CharucoBoard_create(
    num_squares_x, num_squares_y, square_length, marker_length, aruco.Dictionary_get(aruco.DICT_5X5_50))

# Define the size of the output ChArUco board image
image_size = (num_squares_x * square_length, num_squares_y * square_length)

# Generate the ChArUco board image
charuco_board_image = board.draw(image_size)

# Save the generated ChArUco board image
cv2.imwrite("/Spyder_Aalto_Hook/iphone_Charuco_calib/charuco_board.png", charuco_board_image)

# Display the ChArUco board image
cv2.imshow("ChArUco Board", charuco_board_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
