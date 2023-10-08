# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import glob
import os
import gmsh as msh
import sys
import logging
"""
Created on Tue Sep 26 15:55:17 2023
List of Error Handling ideas that are currently left out, mostly due to impact on runtime.
"""


"""
error handling samples: Filter out circumambulate contours by calculating how 
much of the contour is within a central area 
"""

# Assuming you already have the 'contours' variable containing all the contours
filtered_contours = []

# Define the radius of the circular region near the center
center_region_radius = 0.4  # Adjust based on your application (fraction of 
#image width/height)

# Calculate image center
height, width = image.shape[:2]
center_x, center_y = width // 2, height // 2

for contour in contours:
    total_area = cv.contourArea(contour)
    
    # Calculate the area within the circular region near the center
    area_within_center_region = 0
    for point in contour:
        x, y = point[0]
        distance_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        
        if distance_from_center < center_region_radius * min(width, height):
            area_within_center_region += 1

    # Calculate the ratio
    area_ratio = area_within_center_region / total_area

    # Adjust this threshold as needed
    if area_ratio > 0.5:  # You can adjust this threshold
        filtered_contours.append(contour)

    # Now, 'filtered_contours' contains the valid contours that primarily occupy 
    #the central region of the image.
    """
    Create polygons of white at aruco marker faces to prevent them being detected 
    as the object - 1.3.7 Vis_HullwBoolean
    """
    corners, ids, darkness, drkcrnrs = Aruco_Detector(image) #LOCATION OF TEST
  
    for i,value in enumerate(corners):
        start_point = tuple(map(int, tuple(corners[i][0][0])))
        print('check',start_point,'check')
        end_point = tuple(map(int, tuple(corners[i][0][2])))
        
        vertices = np.array([(tuple(map(int, tuple(corners[i][0][0])))),
                             tuple(map(int, tuple(corners[i][0][1]))),
                             tuple(map(int, tuple(corners[i][0][2]))),
                             tuple(map(int, tuple(corners[i][0][3])))])
        
        color = (255, 255, 255)
        imagepoly = cv.fillPoly(image, [vertices], color)
        cv.imshow('polytest', imagepoly) 
        
        
"""
Brute force search for largest contour, did not work very well, small part at 
the end requires that it is central
"""
    contours, hierarchy = cv.findContours(image=thresholded, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)

        # Calculate the areas of all contours
        contour_areas = [cv.contourArea(contour) for contour in contours]
        
        # Sort the contours by area in descending order and get the indices
        sorted_contour_indices = sorted(range(len(contour_areas)), key=lambda i: contour_areas[i], reverse=True)
        
        # Get the top 5 largest contours
        top_5_contours = [contours[i] for i in sorted_contour_indices[:5]]
        # print(top_5_contours)
        # retrieve largest contour only
        contoursBIG = [max(contours, key=cv.contourArea)]
        moments = cv.moments(contoursBIG[0])
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])
       
        #ensure largest contour is central to the image
        central_threshold = .5
        distance_from_center = np.sqrt((cX - width/2)**2 + (cY - height/2)**2)
        max_distance = np.sqrt((width/2)**2 + (height/2)**2)
        print('Contour centroid displacement:',distance_from_center,'Maximum allowed displacement:', max_distance*central_threshold)
        print()
    
        # Determine if the contour is off-center, if largest contour is offcenter return 0
        offcenter = distance_from_center > central_threshold * max_distance
        coords = list(zip((list(contoursBIG[0][:, 0][:, 0])), (list(contoursBIG[0][:, 0][:, 1]))))
        # image for debugging - optional
        if debugmode == 1:
            img_with_contours = cv.cvtColor(img, cv.COLOR_GRAY2BGR,cv.WINDOW_NORMAL)
            cv.drawContours(img_with_contours, [contoursBIG[0]], -1, (0, 255, 0), 2)
            cv.imshow('Largest Contour Highlight', img_with_contours)
            cv.waitKey(0)
            cv.destroyAllWindows()
        
        
        
        #only for testcenter - test
        offcenter = True
        
        if offcenter == False:
            return coords
        else: