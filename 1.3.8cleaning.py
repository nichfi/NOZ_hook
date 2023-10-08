# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 13:57:45 2023
@author: nicko

Imports + initial conditions and variables
"""
import os
import sys
import glob
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import gmsh as msh

msh.initialize(sys.argv)
aruco_scale = 56.76
LC = 20 #for GMSH
SCALING = 0.4
debug_mode=1
projection_list = []


# Define folder path and image name pattern 
folder_path = 'C:/Users/nicko/Downloads/3dt/3dtedtcube'
image_name = 'IMG*'

CMTX = np.array([738.4391110895575, 0.0, 391.10649709357614, 0.0,
                 737.6804692382966, 516.2104937135692, 0.0, 0.0, 1.0], 
                dtype='float32').reshape(3, 3)
DIST = np.array([-0.028550282679872235, -0.01788452651293315, 
                 -0.0002072655617462208, 0.0006906692231601624, 
                 0.5017724257813853], dtype='float32')

center_point = [0, 20, 0]

aruco_points = {
    39: np.array([[32.5,-32.5,-2.5],[32.5,32.5,-2.5],[32.5,-32.5,-62.5],
                  [32.5,32.5,-62.5]], dtype='float32'),
    40: np.array([[32.5,32.5,-2.5],[-32.5,32.5,-2.5],[32.5,32.5,-62.5],
                  [-32.5,32.5,-62.5]], dtype='float32'),
    41: np.array([[-32.5,32.5,-2.5],[-32.5,-32.5,-2.5],[-32.5,-32.5,-62.5],
                  [-32.5,32.5,-62.5]], dtype='float32'),
    42: np.array([[-32.5,-32.5,-2.5],[32.5,-32.5,-2.5],[32.5,-32.5,-62.5],
                  [-32.5,-32.5,-62.5]], dtype='float32')
    }


"""Functions"""

# darkness calculator
def Darkness_Calculator(image, contour):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv.drawContours(mask, [contour], 0, 255, -1)
    masked_image = cv.bitwise_and(image, image, mask=mask)
    darkness_value = np.mean(masked_image[mask > 0]) - 25
    return darkness_value

# resizing function
def Image_Resizer(img, SCALING):
    
    
    width = int(img.shape[1] * SCALING)
    height = int(img.shape[0] * SCALING)
    dim = (width, height)
    image = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    return image

# Detects aruco corners, IDS, and darkness values
def Aruco_Detector(image,debug_mode):
    
    print("Detecting Arucos with Aruco_Detector")
    print()
    # Parameters for detection
    dictionary= cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_50)
    parameters = cv.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = 1
    detector = cv.aruco.ArucoDetector(dictionary, parameters)

    # Detect Aruco markers in the image
    corners, ids, _ = detector.detectMarkers(image)
    # Display detected markers if debug_mode is 0
    if debug_mode == 1 or 2:
        if len(corners) > 0:
            image_with_markers = cv.aruco.drawDetectedMarkers(image.copy(), corners, ids)
            cv.imshow("Detected Markers", image_with_markers)
            cv.waitKey(0)
            cv.destroyAllWindows()
    
    # Calculate the darkness value of the black portion of the marker
    drkcrnrs = []
    for i in range(len(corners)):
        darkness = Darkness_Calculator(image, corners[i][0].astype(int))
        drkcrnrs.append(darkness)
        print(f"Marker {ids[i][0]} Darkness: {darkness:.2f}")
    
    return corners, ids, darkness, drkcrnrs

# Detects outlines of hook (uses independent resizing because of different Gamma requirements)
def Contour_Finder(imagename,debug_mode):
    print("Finding Contours")
    print()
    img = cv.imread(imagename, 0)
    img = Image_Resizer(img,SCALING)
    img = cv.GaussianBlur(img, (5, 5), 5)
    # img = cv.medianBlur(img,15)
    width = int(img.shape[1] )
    height = int(img.shape[0] )
    
    #currently thresholding twice 
    thresholded = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, -5)
    ret, thresholded = cv.threshold(img, min(drkcrnrs), 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    # Inversion needed for contour detection to work (detect contours from zero background)
    thresholded = cv.bitwise_not(thresholded)
    contours, hierarchy = cv.findContours(image=thresholded, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
    
    centroid_list = []
    
    for contour in contours:
        moments = cv.moments(contour)
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])
        distance_from_center = np.sqrt((cX - width/2)**2 + (cY - height/2)**2)
        #print(distance_from_center,'centroid distance from center')
        centroid_list.append(distance_from_center)
    
    
    mincentroid = min(centroid_list)
    x = centroid_list.index(mincentroid)
    print(min(centroid_list),f"Minimum centroid (ID: {x}) distance to center")
    coords = list(zip(contours[x][:, 0][:, 0], contours[x][:, 0][:, 1]))
    
    # Show the image with contours - optional
    if debug_mode == 1:
        img_with_contours = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        cv.drawContours(img_with_contours, [contours[x]], -1, (0, 255, 0), 2)
        cv.imshow('Central Contour Highlight', img_with_contours)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    return coords

# Performs solvePNP
def Pose_Estimator(ids):
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

            # Get the 3D coordinates from the aruco_points dictionary treating the value as the key
            aruco_object_points = aruco_points[value]
            
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

# optional - plots Aruco Markers in space 
def Plot_Arucos(aruco_scale):

# Finds real worls 

    
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

#
def Perspective_Solver():



    # Perform camera pose estimation using solvePnP
    success, rotation_vector, translation_vector = cv.solvePnP(
        reshaped_array_obj , reshaped_array_img , CMTX, DIST)
    #optional
    # print("Rotation Vector:")
    # print(rotation_vector)
    # print("Translation Vector:")
    # print(translation_vector)
    # print("Success Y/N:",success)
    
    # Calculate rotation matrix using the rotation vector from solvePnp
    rotation_matrix = cv.Rodrigues(rotation_vector)[0]
    
    #Calculates camera position from rotation matrix and translation vector
    camera_position = np.array(-np.matrix(rotation_matrix).T * np.matrix(translation_vector))
    
    #creates Camera projection matrix from multiplying camera calibration matrix by the first two columns 
    #of the rotation matrix and 1st column of translation vector
    H = CMTX @ np.column_stack((rotation_matrix[:, 0], rotation_matrix[:, 1], translation_vector[:, 0]))
    return H,camera_position,rotation_matrix,translation_vector,rotation_vector

def Dimensional_Transform_Manual(contour_points):
   
    # Convert 2D points to 3D rays
    stackx=[]
    point3D = []
    d=0
    pwlist = []
    for point in contour_points:
    
        u, v = point
    
        # Homogenize the pixel coordinate to a 3d array
        p = np.array([u, v, 1]).T.reshape((3, 1)) #ALL
    
        # Transform 3d pixel to camera coordinate frame with inverese of camera matrix
        pc = np.linalg.inv(CMTX) @ p # 1_ and 2_

        # Transform pixel camera coordinate to World coordinate frame 
        pw = -rotation_matrix.T.dot(translation_vector) + (rotation_matrix.T@pc) 
        # pw = 
        pwlist.append(pw)
        # Transform camera origin in World coordinate frame
        cam = np.array([0,0,0]).T; cam.shape = (3,1)
        cam_world = -rotation_matrix.T.dot(translation_vector) + rotation_matrix.T @ cam

        vector = pw - cam_world
        unit_vector = vector / np.linalg.norm(vector)
        # print('vector',vector)
        # print('unitranslation_vectortor',unit_vector)

        stackx.append(unit_vector)
        p3D = cam_world + d * unit_vector
        
        point3D.append(p3D)
        
    return stackx,cam_world,pwlist

def Set_Object_Plane(point, direction):
    x, y, z = point
    dx, dy, dz = direction
    
    # Calculate the intersection point
    t = -z / dz
    x_intercept = x + dx * t
    y_intercept = y + dy * t
    z_intercept = z + dz * t
    
    world_outline = (x_intercept, y_intercept, z_intercept)
    return world_outline

def arucos_world():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(*camera_position[:, 0],color='hotpink',label='Camera Position OG',marker = '1')
    
    aruco_points_list=[]
    
    for value in aruco_points.values():
        for point in value:
            aruco_points_list.append(point)
            #aruco_points_list.append([500,500,0])
            ax.scatter(*point, color="black", marker="x" ) 
    for value in world_outline:
        ax.scatter(*value, color="cyan", marker="." )  

          # Add the center_point here

    ax.scatter(*center_point, color="red", marker="o")  # Use a red circle marker for the center_point
    
    # Calculate the ray between the center_point and camera_position
    ray_vector = camera_position - center_point
    print(ray_vector, center_point,"RVRC")

    # Given ray vector
    ray_vector = np.array(ray_vector)

    # Plot the ray as a line from the center_point to camera_position
    ax.plot([center_point[0], camera_position[0]],
        [center_point[1], camera_position[1]],
        [center_point[2], camera_position[2]],
        color='green', linestyle='--')
    line_length = np.linalg.norm(center_point - camera_position)
    
            
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for vector in stackx:
        wo = Set_Object_Plane(cam_world,vector)
        world_outline.append(wo)
        
    #print(world_outline, image_path)
    for value in world_outline:
            ax.scatter(*value, color="cyan", marker="." )   
    ax.elev = 90
    ax.azim = -90
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_aspect('equal', 'box')
    plt.show()

    print("Length of the line:", line_length)

    ax.elev = 90
    ax.azim = -90
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_aspect('equal', 'box')
    plt.show()
    return(ray_vector)

def polymaker(world_points, campP):
    msh.initialize()
    msh.option.setNumber("General.AbortOnError", 1) #Exception: Could not get last error Error   : Gmsh has not been initialized
        
    testcp = [[i[0],i[1],0] for i in world_points]
    #testcamp = tuple([item for sublist in campP1 for item in sublist])
    
    x = round(campP[0][0])
    y = round(campP[1][0])
    z = round(campP[2][0])
    
    # Outline of object
    example_points = [[-3, 3, 0], [3, 3, 0], [3, -3, 0], [-3, -3, 0]]
    example_points = testcp
    # Camera position
    camera_point = msh.model.occ.add_point(x, y, z, LC)
    
    # Establish vertices
    point_list = []
    
    for i, point_coords in enumerate(example_points):
        # point_name = f'point_{i}'
        point_list.append(msh.model.occ.add_point(point_coords[0], point_coords[1], point_coords[2], LC))
    
    # Create a list to fill with all perimeter lines
    perimeter_list = []
    
    # Create outlines of the object
    for i in range(len(example_points)):
        line_handle = msh.model.occ.add_line(point_list[i], point_list[(i+1)%len(example_points)])
        perimeter_list.append(line_handle)
        if i == len(example_points) - 1:
            break
    
    # Create lines from each object perimeter point to the camera origin point
    plin_list = []
    
    for i in range(len(example_points)):
        plin_handle = msh.model.occ.add_line(point_list[i], camera_point)
        plin_list.append(plin_handle)
        if i == len(example_points) - 1:
            break
    
    # Create curve loops adjacent to the camera origin
    loop_list = []
    
    for i in range(len(example_points)):
        loop_handle = msh.model.occ.add_curve_loop([-plin_list[i], perimeter_list[i], plin_list[(i+1)%len(example_points)]])
        loop_list.append(loop_handle)
        if i == len(example_points) - 1:
            break
    
    # Create perimeter loop
    perimeter_handles = [line_handle for line_handle in perimeter_list]
    perimeter_loop = msh.model.occ.add_curve_loop(perimeter_handles)
    
    surface_loop_list = []
    mesh_perimeter = msh.model.occ.add_plane_surface([perimeter_loop])
    surface_loop_list.append(mesh_perimeter)
    
    
    
    for i in range(len(example_points)):
        mesh_test = msh.model.occ.add_plane_surface([loop_list[i]])
        surface_loop_list.append(mesh_test)
        if i == len(example_points) - 1:
            break
    #return mesh_test, mesh_perimeter
    

    #4840 4839 4834
    sl = msh.model.occ.addSurfaceLoop(surface_loop_list)
    # shells.append(sl)
    v = msh.model.occ.addVolume([sl]) 
    
    return mesh_test,v

image_paths = glob.glob(os.path.join(folder_path, image_name))

#begin processing loop
for index, image_path in enumerate(image_paths):
    image = cv.imread(image_path)
    #resize function called
    image = Image_Resizer(image, SCALING)
    #print the image in the sequence that was scanned
    print()
    print(f"Scanning image #:{index}")
    print(f"Image adress:{image_path}")
    print()
    
    corners, ids, darkness, drkcrnrs = Aruco_Detector(image,debug_mode)
    
    # Error handling for 0 contour result, continues loop. - check lighting and avoid dark objects in background or periphery')
    try:
        contour_points= (Contour_Finder(image_path,debug_mode)[::5]) #uses contours function!!!
    except:
        print('contours(',image_path,'): contrast outline too far from center, moving to next picture')
        print()
        continue

    #creating plane of arucos
    reshaped_array_img, reshaped_array_obj = Pose_Estimator(ids)

    H,camera_position,rotation_matrix,translation_vector,rotation_vector = Perspective_Solver()
    
       
    if debug_mode ==1:
        Plot_Arucos(aruco_scale)

    stackx,cam_world,pwlist  = Dimensional_Transform_Manual(contour_points)

    # Plotting the lines between the camera position and the unit vectors
    world_outline = []
    arucos_world() #Comment to erase aruco real world values
    
    
    
    # STL SCRIPT begins
    _,v =polymaker(world_outline , camera_position)
    projection_list.append(v)


# two = msh.model.occ.intersect([(3, 1)], [(3, 2)], 3,removeObject=True, removeTool=True)

booled_object_list = [ [element] for element in projection_list ]
for index, value in enumerate (projection_list) :
    value = msh.model.occ.intersect([(3,1)],[(3,value+1)],removeObject= True, removeTool = True)
    


# # Create the relevant msh data structures from the msh model
msh.model.occ.synchronize()

# Set visibility options to hide points, lines, and 3D faces
msh.option.setNumber("General.Verbosity", 1)  # 1=Show messages and warnings
msh.option.setNumber("Geometry.Points", 1)  #1=Show points
msh.option.setNumber("Geometry.Lines", 0)   #1=Show lines
msh.option.setNumber("Mesh.SurfaceFaces", 0)    #1=Show mesh faces
msh.option.setNumber("Mesh.SurfaceEdges", 0)    #1=Show mesh edges
msh.option.setNumber("Mesh.VolumeEdges", 0)    #1=Show volume edges

# Generate mesh
msh.model.mesh.generate()

# Write mesh data
msh.write("GFG.msh")

# Run the graphical user interface event loop
msh.fltk.run()
#msh.hide(all) hide meshs
#msh.optimize_threshold

# Finalize the msh API
msh.finalize()

        
