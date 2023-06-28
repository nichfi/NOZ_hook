# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 15:15:29 2023

@author: nicko
"""

import gmsh as msh
import numpy as np
import sys

# gmsh 4.11.1
# Initialize gmsh:
msh.initialize(sys.argv)

lc = 20
#def polymaker(contour_points,campP);

#here are the inputs to be fed into the function, this is just defining stuff
contour_points1 = [(1218, 225), (1110, 367), (1036, 520), (1017, 716), (973, 898), (920, 1085), (720, 1197), (533, 1372), (447, 1572), (380, 1658), (328, 1798), (384, 1915), (442, 2016), (537, 2132), (665, 2236), (790, 2344), (911, 2409), (1098, 2429), (1282, 2388), (1460, 2270), (1614, 2085), (1690, 1885), (1704, 1685), (1719, 1485), (1540, 1559), (1415, 1746), (1297, 1917), (1097, 1893), (953, 1714), (1047, 1561), (1198, 1375), (1336, 1221), (1493, 1073), (1391, 902), (1374, 702), (1343, 504), (1354, 304)]
campP1 = [[  88.67113925],[ 497.95212976],[-815.51804156]]
#contour_points2 = [(1218, 225), (1110, 367), (1036, 520), (1017, 716), (973, 898), (920, 1085), (720, 1197), (533, 1372), (447, 1572), (380, 1658), (328, 1798), (384, 1915), (442, 2016), (537, 2132), (665, 2236), (790, 2344), (911, 2409), (1098, 2429), (1282, 2388), (1460, 2270), (1614, 2085), (1690, 1885), (1704, 1685), (1719, 1485), (1540, 1559), (1415, 1746), (1297, 1917), (1097, 1893), (953, 1714), (1047, 1561), (1198, 1375), (1336, 1221), (1493, 1073), (1391, 902), (1374, 702), (1343, 504), (1354, 304)]
#campP2 = [[  95.30699387],[ 495.7407782 ],[-814.71851572]]
contour_points3 = [(1274, 142), (1150, 260), (1072, 392), (1061, 586), (1025, 773), (984, 972), (785, 1086), (630, 1244), (511, 1374), (442, 1547), (473, 1680), (519, 1764), (468, 1777), (531, 1910), (669, 2049), (792, 2114), (945, 2177), (1123, 2225), (1318, 2210), (1495, 2103), (1681, 1940), (1749, 1740), (1753, 1540), (1724, 1349), (1540, 1455), (1428, 1650), (1290, 1727), (1145, 1546), (1169, 1364), (1338, 1169), (1482, 1015), (1505, 843), (1437, 647), (1391, 448), (1404, 248)]
campP3 = [[-174.04032837],[ 817.12439841],[-795.36277238]]
contour_points4 = [(1181, 344), (1064, 422), (1070, 469), (1070, 538), (1096, 654), (1035, 816), (1031, 898), (1026, 997), (1010, 1070), (1013, 1097), (993, 1173), (951, 1212), (885, 1312), (782, 1434), (604, 1600), (503, 1799), (480, 1999), (518, 2199), (634, 2399), (828, 2552), (1028, 2610), (1228, 2601), (1428, 2513), (1600, 2327), (1670, 2127), (1683, 1927), (1709, 1727), (1529, 1727), (1432, 1868), (1343, 2038), (1235, 2175), (1051, 2153), (983, 1957), (1102, 1758), (1235, 1558), (1341, 1361), (1498, 1212), (1357, 1044), (1359, 844), (1308, 688), (1315, 508), (1307, 359)]
campP4 = [[ 126.58470887],[ 377.73211762],[-888.75726601]]

vlist = []

# #NOW THE SCRIPT begins
def polymaker(contour_points, campP):
        
    testcp = [[i[0],i[1],0] for i in contour_points]
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
        point_name = f'point_{i}'
        point_list.append(msh.model.occ.add_point(point_coords[0], point_coords[1], point_coords[2], lc))
    
    # Create a list to fill with all perimeter lines
    perimeter_list = []
    
    # Create outlines of the object
    for i in range(len(example_points)):
        point_name = f'point_{i}'
        next_point_name = f'point_{(i+1)%len(example_points)}'
        line_name = f'line_{i}'
        line_handle = msh.model.occ.add_line(point_list[i], point_list[(i+1)%len(example_points)])
        perimeter_list.append(line_handle)
        if i == len(example_points) - 1:
            break
    
    # Create lines from each object perimeter point to the camera origin point
    plin_list = []
    
    for i in range(len(example_points)):
        point_name = f'point_{i}'
        plin_name = f'plin_{i}'
        next_line_name = f'line_{(i+1)%len(example_points)}'
        plin_handle = msh.model.occ.add_line(point_list[i], camera_point)
        plin_list.append(plin_handle)
        if i == len(example_points) - 1:
            break
    
    # Create curve loops adjacent to the camera origin
    loop_list = []
    
    for i in range(len(example_points)):
        plin_name = f'plin_{i}'
        next_plin_name = f'plin_{(i+1)%len(example_points)}'
        next_line_name = f'line_{(i+1)%len(example_points)}'
        loop_name = f'loop_{i}'
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
    

    
    sl = msh.model.occ.addSurfaceLoop(surface_loop_list)
    # shells.append(sl)
    v = msh.model.occ.addVolume([sl]) 
    vlist.append(v)
    return mesh_test,v

test1 = polymaker(contour_points1,campP1)
#test2 = polymaker(contour_points2,campP2)
test3 = polymaker(contour_points3,campP3)
#test4 = polymaker(contour_points4,campP4)
booltest = msh.model.occ.intersect([(3,vlist[0])],[(3,vlist[1])], removeObject= True, removeTool = True)
#booltest2 = msh.model.occ.cut([(3, test1)], [(3, test3)], removeObject=False)

print(vlist)
# Create the relevant msh data structures from the msh model
msh.model.occ.synchronize()

# Generate mesh
msh.model.mesh.generate()

# Write mesh data
msh.write("GFG.msh")

# Run the graphical user interface event loop
msh.fltk.run()

# Finalize the gmsh API
msh.finalize()
