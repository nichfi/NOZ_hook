# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 17:49:55 2023

@author: nicko
"""
from matplotlib import pyplot as plt

import cv2 as cv
import numpy as np
import glob
import sys

smalldim = 86
largedim = 166
aruco_scale = 56.53
SCALING = .4

#define image to read

imagename = 'C:/Users/nicko/Spyder_Aalto_Hook/Iphone_hook_IDtest/IMG_4811x.jpg'
contoursimage = glob.glob(imagename)
image = cv.imread(imagename)           # Replace "image.jpg" with your actual image file


#define camera calibration params - from calibrator: current(iphone11)
CMTX = np.array([1231.5409815966082, 0.0, 809.2274068386823, 0.0, 1233.8258214550817, 583.4597389570841, 0.0, 0.0, 1.0], dtype='float32').reshape(3, 3)
DIST = np.array([0.2604851842813399, -1.2686473647346477, -0.0027631214482377944, 0.0012125839297280072, 1.8982909964632204], dtype='float32')
aruco_points = {
    8: np.array([[0, 0, 0], [aruco_scale, 0, 0], [aruco_scale, aruco_scale, 0], [0, aruco_scale, 0]], dtype='float32'),
    
    6: np.array([[smalldim + aruco_scale, 0, 0], [smalldim + 2 * aruco_scale, 0, 0],
                  [smalldim + 2 * aruco_scale, aruco_scale, 0], [smalldim + aruco_scale, aruco_scale, 0]], dtype='float32'),
    
    2: np.array([[0, largedim + aruco_scale , 0], [aruco_scale, largedim + aruco_scale, 0], [aruco_scale, largedim + 2 * aruco_scale, 0], [0, largedim + 2* aruco_scale, 0]], dtype='float32'),
    
    4: np.array([[smalldim + aruco_scale, largedim + aruco_scale , 0], [smalldim + 2 * aruco_scale, largedim + aruco_scale, 0],
                  [smalldim + 2 * aruco_scale, largedim + 2*aruco_scale, 0], [smalldim + aruco_scale, largedim + 2*aruco_scale, 0]], dtype='float32'),
   
}


#resizing function
def resize_image(img,SCALING):
    # adapted from from: https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
    width = int(img.shape[1] * SCALING)
    height = int(img.shape[0] * SCALING)
    dim = (width, height)

    # resize image
    image = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    return image

image = resize_image(image,SCALING)

def contours(imagename):
    img = cv.imread(imagename, 0)
    img = resize_image(img,SCALING)
    img = cv.GaussianBlur(img, (5, 5), 5)
    # img = cv.medianBlur(img,15)
    # img = cv.medianBlur(img,25)

    thresholded = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 10)
    ret, thresholded = cv.threshold(img, 130, 255, cv.THRESH_BINARY)

    # Inversion needed for contour detection to work (detect contours from zero background)
    thresholded = cv.bitwise_not(thresholded)

    contours, hierarchy = cv.findContours(image=thresholded, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)

    # get largest contour only
    contours = [max(contours, key=cv.contourArea)]

    coords = list(zip((list(contours[0][:, 0][:, 0])), (list(contours[0][:, 0][:, 1]))))
    # print(list(coords)[::10])
    return coords


contour_points= (contours(imagename)[::40]) #uses contours function!!!

def cvdetector(image):
    # Parameters for detection
    dictionary= cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_50)
    parameters = cv.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = 1
    detector = cv.aruco.ArucoDetector(dictionary, parameters)
    
    #create image scaling (MUST BE SAME AS CALIBRAION PARAMS)

    
    # Detect Aruco markers in the image
    corners, ids, _ = detector.detectMarkers(image)
    return corners, ids

corners, ids = cvdetector(image)

def multi_pose_estimator(ids):
    # Initialize lists to store matched image points and correspondi0ng object points
    matched_image_points = []
    matched_object_points = []

    # Define the IDS to search for
    key_list = []
    for key in aruco_points:
        key_list.append(key)


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
    return reshaped_array_img, reshaped_array_obj

reshaped_array_img, reshaped_array_obj = multi_pose_estimator(ids)

def solvePNPer():
    # Perform camera pose estimation using solvePnP
    success, rotation_vector, translation_vector = cv.solvePnP(
        reshaped_array_obj , reshaped_array_img , CMTX, DIST)
    print("Rotation Vector:")
    print(rotation_vector)
    print("Translation Vector:")
    print(translation_vector)
    print("Success Y/N:",success)
    
    # Calculate rotation matrix using the rotation vector from solvePnp
    rmat = cv.Rodrigues(rotation_vector)[0]
    
    #Calculates camera position from rotation matrix and translation vector
    cameraPosition = np.array(-np.matrix(rmat).T * np.matrix(translation_vector))
    
    #creates Camera projection matrix from multiplying camera calibration matrix by the first two columns 
    #of the rotation matrix and 1st column of translation vector
    H = CMTX @ np.column_stack((rmat[:, 0], rmat[:, 1], translation_vector[:, 0]))
    return H,cameraPosition,rmat,translation_vector,rotation_vector

H,cameraPosition,rmat,translation_vector,rotation_vector = solvePNPer()

def imagedrawer(aruco_scale):
    
    
    # Draw coordinate axes on the image
    axis_length = aruco_scale *.2  # Adjust the length of the coordinate axes as per your preference
    axis_points, _ = cv.projectPoints(axis_length * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                                      rotation_vector, translation_vector, CMTX, DIST)
    
    # Draw the coordinate axes
    for i in range(4):
        cv.drawFrameAxes(image, CMTX, DIST, rotation_vector, translation_vector,
                          axis_length, 2)
    
    # Display the image with coordinate axes
    cv.namedWindow("Image with Coordinate Axes", cv.WINDOW_NORMAL)    # Create window with freedom of dimensions
    
    #draw in a small window / for debugging
    cv.aruco.drawDetectedMarkers(image, corners, ids)
    cv.imshow("Image with Coordinate Axes", image)
    cv.waitKey(0)
    cv.destroyAllWindows()
imagedrawer(aruco_scale)


def dimensional_transforms_contours(contour_points):
    list2dc = [(368.82687,  928.2491 ),(566.44434,  919.0917 ),( 567.1937 , 1121.746  ),(364.99164, 1133.5181)
    ,(382.55545, 205.10068),(563.11237, 205.60915),(564.61676, 375.5201 ),(379.1481 , 375.68643)]
    
    Camera_position_world = [95.1795, 154.647, 422.928]
    t = translation_vector
    R = rmat
    K = CMTX
    
    # Convert 2D points to 3D rays
    stackx=[]
    point3D = []
    d=1
    # Extract the (1, 2) arrays into a new tuple
    new_array = np.concatenate([array.reshape(-1, 2) for array in corners], axis=0)
    newtouple = [tuple(row) for row in new_array]#contour_points = contour_points.append(new_array)
    print(newtouple)
    #contour_points = contour_points.append(newtouple[0])
    
    for point in contour_points:
    
          
        u, v = point
        
    
        # Homogenize the pixel coordinate to a 3d array
        p = np.array([u, v, 1]).T.reshape((3, 1)) #ALL
    
        # Transform 3d pixel to camera coordinate frame with inverese of camera matrix
        pc = np.linalg.inv(K) @ p # 1_ and 2_
        #print (pc)
        # Transform pixel camera coordinate to World coordinate frame
        # pw = t + (R@pc) 
        # pw = -t + (R.T@pc) 
        pw = -R.T.dot(t) + (R.T@pc) 
    
        
        # Transform camera origin in World coordinate frame
        cam = np.array([0,0,0]).T; cam.shape = (3,1)
        # cam_world = t + R @ cam
        # cam_world = -t + R.T @ cam
        cam_world = -R.T.dot(t) + R.T @ cam
        
        vector = pw - cam_world
        unit_vector = vector / np.linalg.norm(vector)
        
        stackx.append(unit_vector)
        
        p3D = cam_world + d * unit_vector
        
        point3D.append(p3D)
        
    
        ########################################## stackexchange 2
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plotting the camera position
    ax.scatter(Camera_position_world[0], Camera_position_world[1], Camera_position_world[2], color='red', marker='o', label='Camera Position')
    return stackx,cam_world

stackx,cam_world = dimensional_transforms_contours(contour_points)
# Plotting the lines between the camera position and the unit vectors


def line_z_plane_intersection(point, direction):
    x, y, z = point
    dx, dy, dz = direction
    
    # Check if the line is parallel to the z-plane
    if dz == 0:
        return None  # Line is parallel to the z-plane, no intersection
    
    # Calculate the intersection point
    t = -z / dz
    x_intercept = x + dx * t
    y_intercept = y + dy * t
    z_intercept = 0
    
    world_outline = (x_intercept, y_intercept, z_intercept)
    return world_outline
    print(world_outline)
    
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

world_outline = []

for vector in stackx:
    wo = line_z_plane_intersection(cam_world,vector)
    world_outline.append(wo)
    
print(world_outline)
for value in world_outline:
        ax.scatter(*value, color="black", marker="x" )  


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

def arucos_world():
    ax.scatter(*cameraPosition[:, 0],color='pink',label='Camera Position OG',marker = '2')
    
    aruco_points_list=[]
    
    for value in aruco_points.values():
        for point in value:
            aruco_points_list.append(point)
            #aruco_points_list.append([500,500,0])
            ax.scatter(*point, color="black", marker="x" )  
arucos_world()


ax.elev = 90
ax.azim = -90
# plt.ylim(150, 170)
# plt.xlim(-90, 170)
# plt.ylim(150, 170)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_aspect('equal', 'box')
plt.show()