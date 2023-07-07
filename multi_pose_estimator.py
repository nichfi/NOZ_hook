# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 17:49:55 2023

@author: nicko
"""

import cv2
import numpy as np
import glob
import sys

smalldim = 86
largedim = 166
aruco_scale = 56.53

SCALING = .4
# Define the dimensions of the Aruco markers


# Define the 3D coordinates of the Aruco markers
aruco_points = {
    8: np.array([[0, 0, 0], [aruco_scale, 0, 0], [aruco_scale, aruco_scale, 0], [0, aruco_scale, 0]], dtype='float32'),
    
    6: np.array([[smalldim + aruco_scale, 0, 0], [smalldim + 2 * aruco_scale, 0, 0],
                  [smalldim + 2 * aruco_scale, aruco_scale, 0], [smalldim + aruco_scale, aruco_scale, 0]], dtype='float32'),
    
    2: np.array([[0, largedim + aruco_scale , 0], [aruco_scale, largedim + aruco_scale, 0], [aruco_scale, largedim + 2 * aruco_scale, 0], [0, largedim + 2* aruco_scale, 0]], dtype='float32'),
    
    4: np.array([[smalldim + aruco_scale, largedim + aruco_scale , 0], [smalldim + 2 * aruco_scale, largedim + aruco_scale, 0],
                  [smalldim + 2 * aruco_scale, largedim + 2*aruco_scale, 0], [smalldim + aruco_scale, largedim + 2*aruco_scale, 0]], dtype='float32')
}

dummy_contours = [(70,100),(100,100),(100,130),(70,130)]

# Define the IDS to search for
key_list = []
for key in aruco_points:
    key_list.append(key)


#fns = glob.glob('C:/Users/nicko/Spyder_Aalto_Hook/hookIDtest/IMG20230703123545')                                                                               
#fns = glob.glob('C:/Users/nicko/PycharmProjects/pythonProject2/main/hook_data/aruco_test3/1*')
# Load the image
#image = cv2.imread('C:/Users/nicko/Spyder_Aalto_Hook/hookIDtest/IMG20230703123545.jg')           # Replace "image.jpg" with your actual image file
#image = cv2.imread('C:/Users/nicko/PycharmProjects/pythonProject2/main/hook_data/aruco_test3/1.jpg')      # Replace "image.jpg" with your actual image file
image = cv2.imread('C:/Users/nicko/Spyder_Aalto_Hook/Iphone_hook_IDtest/IMG_4789.jpg')           # Replace "image.jpg" with your actual image file


def resize_image(img):
    # adapted from from: https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
    width = int(img.shape[1] * SCALING)
    height = int(img.shape[0] * SCALING)
    dim = (width, height)

    # resize image
    image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return image
resize_image(image)
image = cv2.resize(image, (600, 800))




# Detect Aruco markers in the image
dictionary= cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
parameters = cv2.aruco.DetectorParameters()
#parameters.aprilTagMinWhiteBlackDiff = 1
parameters.minMarkerPerimeterRate = .03
parameters.cornerRefinementMethod = 1
detector = cv2.aruco.ArucoDetector(dictionary, parameters)
corners, ids, _ = detector.detectMarkers(image)

#draw in a small window / for debugging
cv2.aruco.drawDetectedMarkers(image, corners, ids)

# Initialize lists to store matched image points and correspondi0ng object points
matched_image_points = []
matched_object_points = []

# create image points and object point arrays based on aruco_id values
for i, value in enumerate(key_list):
    if value in ids:
        
        # Get the index of the ids array that matches the aruco_id list value
        marker_index = np.where(ids == value)[0][0]

        # extract the corners value corresponding to the ids array and aruco_id value
        marker_corners = corners[marker_index][0]
        print(ids[marker_index])
        print(marker_corners )
        # Get the 3D coordinates from the aruco_points dictionary treating the value as the key
        aruco_object_points = aruco_points[value]
        print (value)
        print(aruco_object_points)
        
        # Add matched points to the lists
        matched_image_points.append(marker_corners)
        matched_object_points.append(aruco_object_points)
        
#convert to float32 arrays. float 64 is optional for object points
matched_image_points = np.array(matched_image_points, dtype=np.float32)
matched_object_points = np.array(matched_object_points, dtype=np.float32)

#reshape to 2d arrays, points should still be in the correct order.  THIS  FORMAT IS 
#REQUIRED for solvePNP to work correctly and process all objects into the alg
shaper = (matched_image_points.shape[0])*4

reshaped_array_img = np.reshape(matched_image_points , (shaper, 2))
reshaped_array_obj = np.reshape(matched_object_points , (shaper, 3))

#define camera calibration params - from calibrator: current(iphone11)
CMTX = np.array([610.9408381232331, 0.0, 403.47211930995877, 0.0, 611.6991712878872, 306.3035453618453, 0.0, 0.0, 1.0], dtype='float32').reshape(3, 3)
DIST = np.array([0.25234352229514334, -1.3582990617062876, -0.0005754325467431295, 0.0011291053717762943, 2.278727183078464], dtype='float32')

#debugging
# print(len(matched_image_points),"ArUco's detected")
print('matched obj points', reshaped_array_obj)
print('matched img points', reshaped_array_img)



def desmos_cpr(array_str,array_str2):
    
    array_str = str(reshaped_array_img)
       # Replace the brackets with parentheses
    array_str = array_str.replace('[', '(').replace(']', ')').replace('.',',').replace(',)',')')
    print(array_str)
    
    array_str2 = str(reshaped_array_obj)
    
    # Replace the brackets with parentheses
    array_str2 = array_str2.replace('[', '(').replace(']', ')')#.replace('.',',').replace(',)',')')
    print(array_str2)
#desmos_cpr(reshaped_array_img , reshaped_array_obj )

# Perform camera pose estimation using solvePnP
success, rotation_vector, translation_vector = cv2.solvePnP(
    reshaped_array_obj , reshaped_array_img , CMTX, DIST)

rmat = cv2.Rodrigues(rotation_vector)[0]
cameraPosition = np.array(-np.matrix(rmat).T * np.matrix(translation_vector))
    
H = CMTX @ np.column_stack((rmat[:, 0], rmat[:, 1], translation_vector[:, 0]))
image_points_in_world = [(H @ np.array([i[0], i[1], 1]) / 1) for i in dummy_contours]

H_inv = np.linalg.inv(H)

# Convert the image points in the world back to the contours
#contour_points = [(H_inv @ np.append(image_point, 1))[:3] for image_point in image_points_in_world]

print("Rotation Vector:")
print(rotation_vector)

print("Translation Vector:")
print(translation_vector)
print("Success Y/N:")
print(success)
if success == 0:
    print("Camera pose estimation failed.")
# Draw coordinate axes on the image
axis_length = aruco_scale *.2  # Adjust the length of the coordinate axes as per your preference
axis_points, _ = cv2.projectPoints(axis_length * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                                  rotation_vector, translation_vector, CMTX, DIST)

image_with_axes = image.copy()

# Draw the coordinate axes
for i in range(4):
    cv2.drawFrameAxes(image_with_axes, CMTX, DIST, rotation_vector, translation_vector,
                      axis_length, 2)

#testing reverse projection
test3dcam = [[ 100.11556868],[ 16.06211138], [376.4731593]]
test_in_world= [(H @ np.array([i[0], i[1], 1]) / 1) for i in dummy_contours]
    
#  





#cv2.circle(img,(447,63), 63, (0,0,255), -1)

cv2.circle(image_with_axes ,(100,16), 20, (0,0,255), -1)   
# Display the image with coordinate axes
cv2.imshow("Image with Coordinate Axes", image_with_axes)
cv2.waitKey(0)
cv2.destroyAllWindows()

