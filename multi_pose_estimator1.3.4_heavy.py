from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import glob
import os
import gmsh as msh
import sys
import logging

smalldim = 86
largedim = 166
aruco_scale = 56.53
SCALING = 0.4
vlist = []
errorlist=[]


#define folder path and image name pattern
folder_path = 'C:/Users/nicko/Spyder_Aalto_Hook/iphone_11_library_moment/iphone11_117_Joose/'
imagename_pattern = 'IMG_*.jpg'

#define camera calibration params - from calibrator script: currently using Joose iphone11 folder
CMTX = np.array([1231.5409815966082, 0.0, 809.2274068386823, 0.0, 1233.8258214550817, 583.4597389570841, 0.0, 0.0, 1.0], dtype='float32').reshape(3, 3)
DIST = np.array([0.2604851842813399, -1.2686473647346477, -0.0027631214482377944, 0.0012125839297280072, 1.8982909964632204], dtype='float32')

aruco_points = {
    8: np.array([[0, 0, 0], [aruco_scale, 0, 0], [aruco_scale, aruco_scale, 0], [0, aruco_scale, 0]], dtype='float32'),
    6: np.array([[smalldim + aruco_scale, 0, 0], [smalldim + 2 * aruco_scale, 0, 0],
                 [smalldim + 2 * aruco_scale, aruco_scale, 0], [smalldim + aruco_scale, aruco_scale, 0]], dtype='float32'),
    2: np.array([[0, largedim + aruco_scale, 0], [aruco_scale, largedim + aruco_scale, 0],
                 [aruco_scale, largedim + 2 * aruco_scale, 0], [0, largedim + 2 * aruco_scale, 0]], dtype='float32'),
    4: np.array([[smalldim + aruco_scale, largedim + aruco_scale, 0], [smalldim + 2 * aruco_scale, largedim + aruco_scale, 0],
                 [smalldim + 2 * aruco_scale, largedim + 2 * aruco_scale, 0], [smalldim + aruco_scale, largedim + 2 * aruco_scale, 0]], dtype='float32'),
}


#options for msh - move later
msh.option.setNumber("General.AbortOnError", 1)
msh.initialize(sys.argv)


#darkness calculator
def calculate_darkness(image, contour):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv.drawContours(mask, [contour], 0, 255, -1)
    masked_image = cv.bitwise_and(image, image, mask=mask)
    darkness_value = np.mean(masked_image[mask > 0])
    return darkness_value


# resizing function
def resize_image(img, SCALING):
    # adapted from from: https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
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
    image = resize_image(image, SCALING)
    #print the image in the sequence that was scanned
    print()
    print(f"Scanning image #:{index}")
    print(f"Image adress:{image_path}")
    print()

    # Detects aruco corners, IDS, and darkness values
    def cvdetector(image):
        # Parameters for detection
        dictionary= cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_50)
        parameters = cv.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = 1
        detector = cv.aruco.ArucoDetector(dictionary, parameters)
        
        #create image scaling (MUST BE SAME AS CALIBRAION PARAMS)

        
        # Detect Aruco markers in the image
        corners, ids, _ = detector.detectMarkers(image)
        drkcrnrs = []
        for i in range(len(corners)):
            # Calculate the darkness value of the black portion of the marker
            darkness = calculate_darkness(image, corners[i][0].astype(int))
            drkcrnrs.append(darkness)
            print(f"Marker {ids[i][0]} Darkness: {darkness:.2f}")
        print('darkest corner:', min(drkcrnrs))  
        
        return corners, ids, darkness, drkcrnrs

    corners, ids, darkness, drkcrnrs = cvdetector(image)

    #detects outlines of hook (uses independent resizing because of different Gamma requirements)
    def contoursfinder(imagename):
        img = cv.imread(imagename, 0)
        img = resize_image(img,SCALING)
        img = cv.GaussianBlur(img, (5, 5), 5)
        # img = cv.medianBlur(img,15)
        # img = cv.medianBlur(img,25)
        width = int(img.shape[1] )
        height = int(img.shape[0] )
        
        #currently thresholding twice - fix
        
        thresholded = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, -5)
        ret, thresholded = cv.threshold(img, min(drkcrnrs), 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        #ret, thresholded = cv.threshold(img, min(drkcrnrs), 255, cv.THRESH_BINARY)

        # Inversion needed for contour detection to work (detect contours from zero background)
        thresholded = cv.bitwise_not(thresholded)
        contours, hierarchy = cv.findContours(image=thresholded, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
    
        
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
        img_with_contours = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        cv.drawContours(img_with_contours, [contoursBIG[0]], -1, (0, 255, 0), 2)
    
        

        
        if offcenter == False:
            # Show the largest contours
            # cv.imshow('Largest Contour Highlight', img_with_contours)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
            return coords
        
        else:
            # retrieve central contour
            momentlist = []
            
            for i in contours:
                moments = cv.moments(i)
                momentlist.append(moments)
                
            centroid_list = []
            for i in momentlist:
                cX = int(i["m10"] / i["m00"])
                cY = int(i["m01"] / i["m00"])
                distance_from_center = np.sqrt((cX - width/2)**2 + (cY - height/2)**2)
                print(distance_from_center,'centroid distance from center')
                centroid_list.append(distance_from_center)
            
            mincentroid = (min(centroid_list))
            print(min(centroid_list),'min centroid distance')
            x = centroid_list.index(mincentroid)
            print(x)
            
            coords = list(zip((list(contours[x][:, 0][:, 0])), (list(contours[x][:, 0][:, 1]))))
            img_with_contours = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            cv.drawContours(img_with_contours, [contours[x]], -1, (0, 255, 0), 2)
        
            # # Show the image with contours
            # cv.imshow('Central Contour Highlight', img_with_contours)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
        
 
    # Error handling for 0 contour result, continues loop. - check lighting and avoid dark objects in background or periphery')
    try:
        contour_points= (contoursfinder(image_path)[::10]) #uses contours function!!!
    except:
        print('contours(',image_path,'): contrast outline too far from center, moving to next picture')
        print()
        continue


#previois spot

    #
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

    reshaped_array_img, reshaped_array_obj = multi_pose_estimator(ids)

    def solvePNPer():
        # Perform camera pose estimation using solvePnP
        success, rotation_vector, tvec = cv.solvePnP(
            reshaped_array_obj , reshaped_array_img , CMTX, DIST)
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

    H,cameraPosition,rmat,tvec,rotation_vector = solvePNPer()
    if np.sqrt((cameraPosition[0]**2+cameraPosition[1]**2+cameraPosition[2]**2))> 500 or cameraPosition[2]<0 :
        print('solvePNPer(: erroneous or distant camera position detected (x,y,z:,',cameraPosition,')')
        print()
        continue #error handling
        
    def imagedrawer(aruco_scale):
        
        
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
        # cv.aruco.drawDetectedMarkers(image, corners, ids)
        # cv.imshow("Image with Coordinate Axes", image)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
    #imagedrawer(aruco_scale)


    def dimensional_transforms_contours(contour_points):
       
        # Convert 2D points to 3D rays
        stackx=[]
        point3D = []
        d=0
        # Extract the (1, 2) arrays into a new tuple
        # new_array = np.concatenate([array.reshape(-1, 2) for array in corners], axis=0)
        # newtouple = [tuple(row) for row in new_array]#contour_points = contour_points.append(new_array)
        # print(newtouple)
        #contour_points = contour_points.append(newtouple[0])
        
        for point in contour_points:
        
            u, v = point
        
            # Homogenize the pixel coordinate to a 3d array
            p = np.array([u, v, 1]).T.reshape((3, 1)) #ALL
        
            # Transform 3d pixel to camera coordinate frame with inverese of camera matrix
            pc = np.linalg.inv(CMTX) @ p # 1_ and 2_

            # Transform pixel camera coordinate to World coordinate frame 
            pw = -rmat.T.dot(tvec) + (rmat.T@pc) 
        
            # Transform camera origin in World coordinate frame
            cam = np.array([0,0,0]).T; cam.shape = (3,1)
            cam_world = -rmat.T.dot(tvec) + rmat.T @ cam
            
            vector = pw - cam_world
            unit_vector = vector / np.linalg.norm(vector)
            
            stackx.append(unit_vector)
            p3D = cam_world + d * unit_vector
            
            point3D.append(p3D)
            
        return stackx,cam_world

    stackx,cam_world = dimensional_transforms_contours(contour_points)
    # Plotting the lines between the camera position and the unit vectors
    
    world_outline = []
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
        z_intercept = 40
        
        world_outline = (x_intercept, y_intercept, z_intercept)
        return world_outline
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    world_outline = []

    for vector in stackx:
        wo = line_z_plane_intersection(cam_world,vector)
        world_outline.append(wo)
        
    #print(world_outline, image_path)
    for value in world_outline:
            ax.scatter(*value, color="cyan", marker="." )   


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

        ax.elev = 45
        ax.azim = -90
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_aspect('equal', 'box')
        plt.show()

    arucos_world() #Comment to erase aruco real world values

   

    lc = 20

    # #NOW THE SCRIPT begins
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
            # point_name = f'point_{i}'
            # next_point_name = f'point_{(i+1)%len(example_points)}'
            # line_name = f'line_{i}'
            line_handle = msh.model.occ.add_line(point_list[i], point_list[(i+1)%len(example_points)])
            perimeter_list.append(line_handle)
            if i == len(example_points) - 1:
                break
        
        # Create lines from each object perimeter point to the camera origin point
        plin_list = []
        
        for i in range(len(example_points)):
            # point_name = f'point_{i}'
            # plin_name = f'plin_{i}'
            # next_line_name = f'line_{(i+1)%len(example_points)}'
            plin_handle = msh.model.occ.add_line(point_list[i], camera_point)
            plin_list.append(plin_handle)
            if i == len(example_points) - 1:
                break
        
        # Create curve loops adjacent to the camera origin
        loop_list = []
        
        for i in range(len(example_points)):
            # plin_name = f'plin_{i}'
            # next_plin_name = f'plin_{(i+1)%len(example_points)}'
            # next_line_name = f'line_{(i+1)%len(example_points)}'
            # loop_name = f'loop_{i}'
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
        #test = polymaker(world_points2,campP2)
        #test3 = polymaker(world_points3,campP3)
        #test4 = polymaker(world_points4,campP4)
        #booltest = msh.model.occ.intersect([(3,vlist[0])],[(3,vlist[1])], removeObject= True, removeTool = True)
        #booltest2 = msh.model.occ.cut([(3, test1)], [(3, test3)], removeObject=False)
    
    _,v =polymaker(world_outline , cameraPosition)
    vlist.append(v)
    print()
    if len(vlist) >  2: #test
        break

#print('v',v)


# #Now, perform boolean intersection with subsequent volumes

#bool_result = msh.model.occ.intersect([(3, 3)], [(3, 8)], 8, removeObject=True, removeTool=True)
#print(bool_result,'bool result')
msh.model.occ.synchronize()
msh.model.mesh.generate()
two = msh.model.occ.intersect([(3, 3)], [(3, 2)], 8,removeObject=False, removeTool=False)




# #Working Example of Boolean - note: booleans must be done one by one.
# booled_object_list = [ [element] for element in vlist ]
# for i,value in booled_object_list :
#     booled_object_list[i+1] = msh.model.occ.intersect([(3,booled_object_list[value])],[(3,1)],removeObject= False, removeTool = True)

# print(booled_object_list)
# #print(vlist)


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

        