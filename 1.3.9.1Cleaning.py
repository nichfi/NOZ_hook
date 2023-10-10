from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import glob
import os
import gmsh as msh
import sys
import logging

msh.initialize(sys.argv)

aruco_scale = 56.76
LC = 20 #for GMSH
SCALING = 0.4
debug_mode = 0
projection_list = []

#define folder path and image name pattern 
folder_path = 'C:/Users/skippy/Downloads/'
imagename_pattern = 'IMG_20231002_034*'


#define camera calibration params - from calibrator script: currently using Joose iphone11 folder
# CMTX = np.array([1231.5409815966082, 0.0, 809.2274068386823, 0.0, 1233.8258214550817, 583.4597389570841, 0.0, 0.0, 1.0], dtype='float32').reshape(3, 3)
# DIST = np.array([0.2604851842813399, -1.2686473647346477, -0.0027631214482377944, 0.0012125839297280072, 1.8982909964632204], dtype='float32')
#for procam checker 1 .4
CMTX = np.array([738.4391110895575, 0.0, 391.10649709357614, 0.0,
                 737.6804692382966, 516.2104937135692, 0.0, 0.0, 1.0], 
                dtype='float32').reshape(3, 3)
DIST = np.array([-0.028550282679872235, -0.01788452651293315, 
                 -0.0002072655617462208, 0.0006906692231601624, 
                 0.5017724257813853], dtype='float32')


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


# msh.option.setNumber("General.AbortOnError", 1)

# darkness caLCulator
def caLCulate_darkness(image, contour):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv.drawContours(mask, [contour], 0, 255, -1)
    masked_image = cv.bitwise_and(image, image, mask=mask)
    darkness_value = np.mean(masked_image[mask > 0]) - 25
    return darkness_value

# resizing function
def resize_image(img, SCALING):
    width = int(img.shape[1] * SCALING)
    height = int(img.shape[0] * SCALING)
    dim = (width, height)
    image = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    return image

# Detects aruco corners, IDS, and darkness values
def detect_markers(image):
    # Parameters for detection
    dictionary= cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_50)
    parameters = cv.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = 0
    detector = cv.aruco.ArucoDetector(dictionary, parameters)
    
    
    # Detect Aruco markers in the image
    corners, ids, _ = detector.detectMarkers(image)
    aruco_corner_darkness = []
    for i in range(len(corners)):
        # CaLCulate the darkness value of the black portion of the marker
        darkness = caLCulate_darkness(image, corners[i][0].astype(int))
        aruco_corner_darkness.append(darkness)
        if debug_mode == 1:
            print(f"Marker {ids[i][0]} Darkness: {darkness:.2f}")  
    
    return corners, ids, darkness, aruco_corner_darkness

# Detects outlines of hook (uses independent resizing because of different Gamma requirements)
def find_contours(imagename,debug_mode):
    print("Finding Contours")
    print()
    img = cv.imread(imagename, 0)
    img = resize_image(img,SCALING)
    img = cv.GaussianBlur(img, (5, 5), 5)
    # img = cv.medianBlur(img,15)
    width = int(img.shape[1] )
    height = int(img.shape[0] )
    
    #currently thresholding twice 
    thresholded = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, -5)
    ret, thresholded = cv.threshold(img, min(aruco_corner_darkness), 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
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
    if debug_mode == 1 or 2:
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

# Combines aruco location data 
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

# Solves perspective and point
def pnp_solver():
    # Perform camera pose estimation using solvePnP
    success, rotation_vector, tvec = cv.solvePnP(
        reshaped_array_obj , reshaped_array_img , CMTX, DIST)
    #optional
    # print("Rotation Vector:")
    # print(rotation_vector)
    # print("Translation Vector:")
    # print(tvec)
    # print("Success Y/N:",success)
    
    # CaLCulate rotation matrix using the rotation vector from solvePnp
    rmat = cv.Rodrigues(rotation_vector)[0]
    
    #CaLCulates camera position from rotation matrix and translation vector
    camera_position_pnp = np.array(-np.matrix(rmat).T * np.matrix(tvec))
    
    #creates Camera projection matrix from multiplying camera calibration matrix by the first two columns 
    #of the rotation matrix and 1st column of translation vector
    H = CMTX @ np.column_stack((rmat[:, 0], rmat[:, 1], tvec[:, 0]))
    return H,camera_position_pnp,rmat,tvec,rotation_vector

# Coordinate visualizer - optional
def coordinate_visualizer(aruco_scale):
    
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
    
def dimensional_transforms_contours(contour_points):
   
    # Convert 2D points to 3D rays
    unit_vector_list=[]
    point3D = []
    d=0
    world_coordinates_list = []
    vector_list = []
    
    for image_coordinates in contour_points:
    
        u, v = image_coordinates
    
        # Homogenize the pixel coordinate to a 3d array
        homogenized_coordinates = np.array([u, v, 1]).T.reshape((3, 1)) #ALL
    
        # Transform 3d pixel to camera coordinate frame with inverese of camera matrix
        camera_frame_coordinates = np.linalg.inv(CMTX) @ homogenized_coordinates # 1_ and 2_

        # Transform pixel camera coordinate to World coordinate frame 
        world_frame_coordinates = -rmat.T.dot(tvec) + (rmat.T@camera_frame_coordinates) 
        world_coordinates_list.append(world_frame_coordinates)
        # Transform camera origin in World coordinate frame
        camera_0 = np.array([0,0,0]).T; camera_0.shape = (3,1)
        cam_world = -rmat.T.dot(tvec) + rmat.T @ camera_0
        
        vector = world_frame_coordinates - cam_world
        unit_vector = vector / np.linalg.norm(vector)
        
        unit_vector_list.append(unit_vector)
        
        p3D = cam_world + d * unit_vector
        point3D.append(p3D)
        vector_list.append(vector)
        
    return unit_vector_list,cam_world,vector_list,world_coordinates_list

 
# Creates a plane normal to the vector between the object centroid and the camera
def norm_plane_estimator(point, direction):
    
    x, y, z = point
    dx, dy, dz = direction
    
    # Check if the line is parallel to the z-plane
    if dz == 0:
        print('parallel camera, check for pnp error' ) # Line is parallel to the z-plane, no intersection
    
    # CaLCulate the intersection point
    t = -z / dz
    x_intercept = x + dx * t
    y_intercept = y + dy * t
    z_intercept = 0
    
    world_outline = (x_intercept, y_intercept, z_intercept)
    return world_outline


# STL creation script begins
def polyhedron_obj_converter(world_points, campP):
    msh.initialize()
    msh.option.setNumber("General.AbortOnError", 1) #Exception: Could not get last error Error   : Gmsh has not been initialized
        
    testcp = [[i[0],i[1],i[2]] for i in world_points]
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



'''
Main
'''

# Return a list of image file paths that match the pattern
image_paths = glob.glob(os.path.join(folder_path, imagename_pattern))

#begin processing loop
for index, image_path in enumerate(image_paths):
    image = cv.imread(image_path)
    image = resize_image(image, SCALING)
    print()
    print(f"Scanning image #:{index}")
    print(f"Image adress:{image_path}")
    print()

    corners, ids, darkness, aruco_corner_darkness = detect_markers(image)

    # Error handling for 0 contour result, continues loop. - check lighting and avoid dark objects in background or periphery')
    try:
        contour_points= (find_contours(image_path,debug_mode)[::10]) #uses contours function!!!
    except:
        print('(',image_path,'):find_contour error')
        print()
        continue


    reshaped_array_img, reshaped_array_obj = multi_pose_estimator(ids)

    H,camera_position_pnp,rmat,tvec,rotation_vector = pnp_solver()

    unit_vector_list,cam_world,vector_list,world_frame_coordinate_list  = dimensional_transforms_contours(contour_points)
 

    
        
    tuple_list = []
    # Iterate through the arrays in the tuple
    for array in corners:
        # Iterate through the rows in the array
        for row in array[0]:
            # Extract the two values from each row and convert them into a tuple
            value_tuple = tuple(row)
            # Append the tuple to the tuple_list
            tuple_list.append(value_tuple)
    unit_vector_list_arucos,cam_world_arucos,_,_  = dimensional_transforms_contours(tuple_list)
    
    
    
    
    # CaLCulate the ray between the centerpoint and cam_world
    centerpoint = [[0], [0], [0]]

    ray_vector = cam_world - centerpoint
    far_point = -(cam_world)
    D_far = ray_vector[0]*far_point[0]+ray_vector[1]*far_point[1]+ray_vector[2]*far_point[2]

    origin_point = [cam_world[0],cam_world[1],cam_world[2]]
    # Initialize a list to store intersection points
    intersection_point_list = []
    
    # Iterate through each unit vector
    for unit_vector in unit_vector_list:
        # Calculate the intersection point with the plane
        # The equation of the line is x = x0 + t * u, y = y0 + t * v, z = z0 + t * w
        # Solve for t using the plane equation and the equation of the line
        t = (D_far - ray_vector[0] * origin_point[0] - ray_vector[1] * origin_point[1] - ray_vector[2] * origin_point[2]) / (ray_vector[0] * unit_vector[0] + ray_vector[1] * unit_vector[1] + ray_vector[2] * unit_vector[2])
    
        # Calculate the intersection point
        intersection_point = (origin_point[0] + t * unit_vector[0], origin_point[1] + t * unit_vector[1], origin_point[2] + t * unit_vector[2])
    
        intersection_point_list.append(intersection_point)
       
        
       
        
       
        
    #optional
    def arucos_world(cam_world,unit_vector_list):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(*cam_world[:, 0],color='hotpink',label='Camera Position OG',marker = '2')
        
        aruco_points_list=[]
        
        for value in aruco_points.values():
            for point in value:
                aruco_points_list.append(point)
                #aruco_points_list.append([500,500,0])
                ax.scatter(*point, color="black", marker="x" ) 
        # for value in world_outline:
        #     ax.scatter(*value, color="cyan", marker="." )  

              # Add the centerpoint here
        centerpoint = [[0], [0], [0]]

        ax.scatter(*centerpoint, color="red", marker="o")  # Use a red circle marker for the centerpoint
        
        # CaLCulate the ray between the centerpoint and cam_world
        ray_vector = cam_world - centerpoint
        far_point = -(cam_world)
        D_far = ray_vector[0]*far_point[0]+ray_vector[1]*far_point[1]+ray_vector[2]*far_point[2]
        
        # Plot the ray as a line from the centerpoint to cam_world
        ax.plot([centerpoint[0], cam_world[0]],
            [centerpoint[1], cam_world[1]],
            [centerpoint[2], cam_world[2]],
            color='green', linestyle='--')
        line_length = np.linalg.norm(centerpoint - cam_world)

        print("Length of the line:", line_length)

        ax.elev = 45
        ax.azim = 0 
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_aspect('equal', 'box')
        plt.show()
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Define a common color for all lines
        line_color = 'b'  # You can change 'b' to any valid Matplotlib color

        # Plot the line between cam_world and unit vectors in unit_vector_list
        for unit_vector in unit_vector_list:
            # Create points for the line
            x_points = [cam_world[0, 0], cam_world[0, 0] + unit_vector[0, 0]*line_length]
            y_points = [cam_world[1, 0], cam_world[1, 0] + unit_vector[1, 0]*line_length]
            z_points = [cam_world[2, 0], cam_world[2, 0] + unit_vector[2, 0]*line_length]

            # Use the common color for all lines
            ax.plot(x_points, y_points, z_points, color=line_color)
            
        for value in aruco_points.values():
            for point in value:
                aruco_points_list.append(point)
                #aruco_points_list.append([500,500,0])
                ax.scatter(*point, color="black", marker="x" ) 
        # Customize the plot as 

        
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_title('Lines between cam_world and unit vectors')
        ax.set_aspect('equal', 'box')
        ax.elev = 0
        ax.azim = -30
        # Show the plot
        plt.show()
            
        return ray_vector,x_points #,world_coordinates_list,vectorlist,point3D
        
    ray_vector,x_points = arucos_world(cam_world,unit_vector_list) #Comment to erase aruco real world values
    ray_vector_aruco,x_points_aruco = arucos_world(cam_world_arucos ,unit_vector_list_arucos) #Comment to erase aruco real world values

    # coordinate_visualizer(aruco_scale)




    _,v =polyhedron_obj_converter(intersection_point_list , cam_world)
    projection_list.append(v)
    print()
    #optional  
    if len(projection_list) >  40: #test
        break


for index, value in enumerate (projection_list) :
    value = msh.model.occ.intersect([(3,1)],[(3,value+1)],removeObject= True, removeTool = True)


# # Create the relevant msh data structures from the msh model
msh.model.occ.synchronize()
# Set visibility options to hide points, lines, and 3D faces
msh.option.setNumber("General.Verbosity", 1)  # 1=Show all messages and warnings
msh.option.setNumber("Geometry.Points", 0)   
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

        
