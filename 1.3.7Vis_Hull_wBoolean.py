# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 19:46:17 2023

@author: nicko
"""

from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import glob
import os
import gmsh as msh
import sys
import logging

smalldim = 66.31
largedim = 82.19
aruco_scale = 56.76
SCALING = 0.4
debugmode=1
vlist = []
errorlist=[]

#define folder path and image name pattern 
folder_path = 'C:/Users/nicko/Downloads/3dtedtcube'
imagename_pattern = 'IMG*'

#define camera calibration params - from calibrator script: currently using Joose iphone11 folder
# CMTX = np.array([1231.5409815966082, 0.0, 809.22740[-68386823, 0.0, 1233.8258214550817, 583.4597389570841, 0.0, 0.0, 1.0], dtype='float32').reshape(3, 3)
# DIST = np.array([0.2604851842813399, -1.2686473647346477, -0.0027631214482377944, 0.0012125839297280072, 1.8982909964632204], dtype='float32')
#for procam checker 1 .4
# CMTX = np.array([1156.8361842962481, 0.0, 602.5193809906307, 0.0, 1155.4587429108044, 802.5231337051961, 0.0, 0.0, 1.0], dtype='float32').reshape(3, 3)
# DIST = np.array([-0.07340364563454689, 0.3237118230528758, 0.0015036752884026255, -0.00010017544216576494, -0.40144988224708344], dtype='float32')

# Procam small
# CMTX = np.array([1156.8361842962481, 0.0, 602.5193809906307, 0.0, 1155.4587429108044, 802.5231337051961, 0.0, 0.0, 1.0], dtype='float32').reshape(3, 3)
# DIST = np.array([-0.07340364563454689, 0.3237118230528758, 0.0015036752884026255, -0.00010017544216576494, -0.40144988224708344], dtype='float32')
#procam_remeasured
# CMTX = np.array([363.40274411526906, 0.0, 195.3764106761795, 0.0, 363.72093472114784, 256.5031233114503, 0.0, 0.0, 1.0], dtype='float32').reshape(3, 3)
# DIST = np.array([-0.058551058619618845, 0.33480868732908253, 0.0010543745032795722, -0.001023935262334852, -0.5430476006001702], dtype='float32')
#procam_remeasured_no_glare
# CMTX = np.array([738.9090683753249, 0.0, 390.4196137797938, 0.0, 739.1264740734568, 511.32994803059637, 0.0, 0.0, 1.0], dtype='float32').reshape(3, 3)
# DIST = np.array([-0.03882346140102668, 0.20717019978089832, -0.000911731929343776, -8.051361341482986e-05, -0.3432625783785088], dtype='float32')

CMTX = np.array([738.4391110895575, 0.0, 391.10649709357614, 0.0, 737.6804692382966, 516.2104937135692, 0.0, 0.0, 1.0], dtype='float32').reshape(3, 3)
DIST = np.array([-0.028550282679872235, -0.01788452651293315, -0.0002072655617462208, 0.0006906692231601624, 0.5017724257813853], dtype='float32')

center_point = [((aruco_scale*2)+smalldim)/2 , (aruco_scale*2+largedim)/2, 0]

# aruco_points = {
#     8: np.array([[0, 0, 0], [aruco_scale, 0, 0], [aruco_scale, aruco_scale, 0], [0, aruco_scale, 0]], dtype='float32'),
#     6: np.array([[smalldim + aruco_scale, 0, 0], [smalldim + 2 * aruco_scale, 0, 0],
#                  [smalldim + 2 * aruco_scale, aruco_scale, 0], [smalldim + aruco_scale, aruco_scale, 0]], dtype='float32'),
#     2: np.array([[0, largedim + aruco_scale, 0], [aruco_scale, largedim + aruco_scale, 0],
#                  [aruco_scale, largedim + 2 * aruco_scale, 0], [0, largedim + 2 * aruco_scale, 0]], dtype='float32'),
#     4: np.array([[smalldim + aruco_scale, largedim + aruco_scale, 0], [smalldim + 2 * aruco_scale, largedim + aruco_scale, 0],
#                  [smalldim + 2 * aruco_scale, largedim + 2 * aruco_scale, 0], [smalldim + aruco_scale, largedim + 2 * aruco_scale, 0]], dtype='float32'),
# }

aruco_points = {
    39: np.array([[32.5,-32.5,-2.5],[32.5,32.5,-2.5],[32.5,-32.5,-62.5],[32.5,32.5,-62.5]], dtype='float32'),
    40: np.array([[32.5,32.5,-2.5],[-32.5,32.5,-2.5],[32.5,32.5,-62.5],[-32.5,32.5,-62.5]], dtype='float32'),
    41: np.array([[-32.5,32.5,-2.5],[-32.5,-32.5,-2.5],[-32.5,-32.5,-62.5],[-32.5,32.5,-62.5]], dtype='float32'),
    42: np.array([[-32.5,-32.5,-2.5],[32.5,-32.5,-2.5],[32.5,-32.5,-62.5],[-32.5,-32.5,-62.5]], dtype='float32')
    }

#options for msh - move later
msh.initialize(sys.argv)

msh.option.setNumber("General.AbortOnError", 0)


#darkness calculator
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


# Return a list of image file paths that match the pattern
image_paths = glob.glob(os.path.join(folder_path, imagename_pattern))

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

    # Detects aruco corners, IDS, and darkness values
    def Aruco_Detector(image):
        print("Detecting Arucos with Aruco_Detector")
        print()
        # Parameters for detection
        dictionary= cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_50)
        parameters = cv.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = 0
        detector = cv.aruco.ArucoDetector(dictionary, parameters)
 
        # Detect Aruco markers in the image
        corners, ids, _ = detector.detectMarkers(image)
        
        # Calculate the darkness value of the black portion of the marker
        drkcrnrs = []
        for i in range(len(corners)):
            darkness = Darkness_Calculator(image, corners[i][0].astype(int))
            drkcrnrs.append(darkness)
            print(f"Marker {ids[i][0]} Darkness: {darkness:.2f}")
        print('Darkest corner:', min(drkcrnrs))  
        
        return corners, ids, darkness, drkcrnrs


    corners, ids, darkness, drkcrnrs = Aruco_Detector(image)


    #detects outlines of hook (uses independent resizing because of different Gamma requirements)
    def Contour_Finder(imagename):
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
        
        return coords
        # # Show the image with contours - optional
        if debugmode ==1:
            img_with_contours = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            cv.drawContours(img_with_contours, [contours[x]], -1, (0, 255, 0), 2)
            cv.imshow('Central Contour Highlight', img_with_contours)
            cv.waitKey(0)
            cv.destroyAllWindows()
    
    # Error handling for 0 contour result, continues loop. - check lighting and avoid dark objects in background or periphery')
    try:
        contour_points= (Contour_Finder(image_path)[::5]) #uses contours function!!!
    except:
        print('contours(',image_path,'): contrast outline too far from center, moving to next picture')
        print()
        continue


    #creating plane of arucos
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
                # print(ids[marker_index])
                # print(marker_corners )
                # Get the 3D coordinates from the aruco_points dictionary treating the value as the key
                aruco_object_points = aruco_points[value]
                # print (value)
                # print(aruco_object_points)
                
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

    reshaped_array_img, reshaped_array_obj = Pose_Estimator(ids)

    
    def Perspective_Solver():
        # Perform camera pose estimation using solvePnP
        success, rotation_vector, tvec = cv.solvePnP(
            reshaped_array_obj , reshaped_array_img , CMTX, DIST)
        #optional
        # print("Rotation Vector:")
        # print(rotation_vector)
        # print("Translation Vector:")
        # print(tvec)
        # print("Success Y/N:",success)
        
        # Calculate rotation matrix using the rotation vector from solvePnp
        rmat = cv.Rodrigues(rotation_vector)[0]
        
        #Calculates camera position from rotation matrix and translation vector
        cameraPosition = np.array(-np.matrix(rmat).T * np.matrix(tvec))
        
        #creates Camera projection matrix from multiplying camera calibration matrix by the first two columns 
        #of the rotation matrix and 1st column of translation vector
        H = CMTX @ np.column_stack((rmat[:, 0], rmat[:, 1], tvec[:, 0]))
        return H,cameraPosition,rmat,tvec,rotation_vector

    H,cameraPosition,rmat,tvec,rotation_vector = Perspective_Solver()
    if np.sqrt((cameraPosition[0]**2+cameraPosition[1]**2+cameraPosition[2]**2))> 900 or cameraPosition[2]<0 :
        print('Perspective_Solver(: erroneous or distant camera position detected (x,y,z:,',cameraPosition,')')
        print()
        continue #error handling
       
    #optional   
    def Plot_Arucos(aruco_scale):
        
        # Draw coordinate axes on the image
        axis_length = aruco_scale *.2  # Adjust the length of the coordinate axes as per your preference
        axis_points, _ = cv.projectPoints(axis_length * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                                          rotation_vector, tvec, CMTX, DIST)
        
        # Draw the coordinate axes
        for i in range(4):
            cv.drawFrameAxes(image, CMTX, DIST, rotation_vector, tvec,
                              axis_length, 2)
        
        # Display the image with coordinate axes
        cv.namedWindow("Image with Coordinate Axes", cv.WINDOW_NORMAL)    # Create window with freedom of dimensions
        
        #draw in a small window / for debugging
        cv.aruco.drawDetectedMarkers(image, corners, ids)
        cv.imshow("Image with Coordinate Axes", image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    if debugmode ==1:
        Plot_Arucos(aruco_scale)


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
            pw = -rmat.T.dot(tvec) + (rmat.T@pc) 
            # pw = 
            pwlist.append(pw)
            # Transform camera origin in World coordinate frame
            cam = np.array([0,0,0]).T; cam.shape = (3,1)
            cam_world = -rmat.T.dot(tvec) + rmat.T @ cam
 
            vector = pw - cam_world
            unit_vector = vector / np.linalg.norm(vector)
            # print('vector',vector)
            # print('unitvector',unit_vector)

            stackx.append(unit_vector)
            p3D = cam_world + d * unit_vector
            
            point3D.append(p3D)
            
        return stackx,cam_world,pwlist

    stackx,cam_world,pwlist  = Dimensional_Transform_Manual(contour_points)
    
    
    # Plotting the lines between the camera position and the unit vectors
    world_outline = []
    def Set_Object_Plane(point, direction,D):
        x, y, z = point
        dx, dy, dz = direction
        
        # Check if the line is parallel to the z-plane
        if dz == 0:
            return None  # Line is parallel to the z-plane, no intersection
        
        # Calculate the intersection point
        t = -z / dz
        x_intercept = x + dx * t
        y_intercept = y + dy * t
        z_intercept = D
        
        world_outline = (x_intercept, y_intercept, z_intercept)
        return world_outline


    #optional
    def arucos_world():
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(*cameraPosition[:, 0],color='hotpink',label='Camera Position OG',marker = '2')
        
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
        
        # Calculate the ray between the center_point and cameraPosition
        ray_vector = cameraPosition - center_point
        print(ray_vector, center_point,"RVRC")
        # Given point on the plane
        point_on_plane = np.array(center_point)

        # Given ray vector
        ray_vector = np.array(ray_vector)
        
        # Calculate the normal vector by normalizing the ray vector
        normal_vector = ray_vector / np.linalg.norm(ray_vector)

        # Calculate the constant 'D' using the dot product of the normal vector and the point
        D = np.dot(normal_vector, point_on_plane)

        # Extract components of the normal vector (A, B, C)
        A1, B2, C3 = normal_vector

        # Create a variable for the plane equation
        plane_equation = f"{A1:.4f}x + {B2:.4f}y + {C3:.4f}z = {D:.4f}"
        
        # Plot the ray as a line from the center_point to cameraPosition
        ax.plot([center_point[0], cameraPosition[0]],
            [center_point[1], cameraPosition[1]],
            [center_point[2], cameraPosition[2]],
            color='green', linestyle='--')
        line_length = np.linalg.norm(center_point - cameraPosition)
        
                
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        for vector in stackx:
            wo = Set_Object_Plane(cam_world,vector,D)
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

    arucos_world() #Comment to erase aruco real world values
    
    lc = 20
    if debugmode==1:
        continue
    # STL SCRIPT begins
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
        camera_point = msh.model.occ.add_point(x, y, z, lc)
        
        # Establish vertices
        point_list = []
        
        for i, point_coords in enumerate(example_points):
            # point_name = f'point_{i}'
            point_list.append(msh.model.occ.add_point(point_coords[0], point_coords[1], point_coords[2], lc))
        
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

    
    _,v =polymaker(world_outline , cameraPosition)
    vlist.append(v)
    print()
    #optional  
    if len(vlist) >  10: #test
        break


# two = msh.model.occ.intersect([(3, 1)], [(3, 2)], 3,removeObject=True, removeTool=True)

booled_object_list = [ [element] for element in vlist ]
for index, value in enumerate (vlist) :
    value = msh.model.occ.intersect([(3,1)],[(3,value+1)],removeObject= True, removeTool = True)
    


# # Create the relevant msh data structures from the msh model
msh.model.occ.synchronize()
# Set visibility options to hide points, lines, and 3D faces
msh.option.setNumber("General.Verbosity", 1)  # 1=Show all messages and warnings
msh.option.setNumber("Geometry.Points", 1)   
msh.option.setNumber("Geometry.Lines", 0)   
msh.option.setNumber("Mesh.SurfaceFaces", 0)      # Hide all points, lines, and 3D faces in the GUI
msh.option.setNumber("Mesh.SurfaceEdges", 0)      # Hide all points, lines, and 3D faces in the GUI
msh.option.setNumber("Mesh.VolumeFaces", 1)      # Hide all points, lines, and 3D faces in the GUI
msh.option.setNumber("Mesh.VolumeEdges", 0)      # Hide all points, lines, and 3D faces in the GUI


# #msh.model.mesh.Triangles(0)


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

        
