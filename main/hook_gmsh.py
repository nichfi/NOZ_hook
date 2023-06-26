import gmsh as msh
import numpy as np
import sys

# gmsh   4.11.1
# Initialize gmsh:
msh.initialize(sys.argv)


lc = 1e-2

#outline of object
example_points = [[-3, 3, 0], [3, 3, 0], [3, -3, 0], [-3, -3, 0]]

#camera position
camera_point = msh.model.geo.add_point(4, 4, 10, lc)

#establish vertices 
for i, point_coords in enumerate(example_points):
    point_name = f'point_{i}'
    globals()[point_name] = msh.model.geo.add_point(point_coords[0], point_coords[1], point_coords[2], lc)

#create a list to fill with all perimeter lines
perimeter_list =[]

#create outlines of object
for i in range(len(example_points)):
    point_name = f'point_{i}'
    next_point_name = f'point_{(i+1)%len(example_points)}'
    line_name = f'line_{i}'
    globals()[line_name] = msh.model.geo.add_line(globals()[point_name], globals()[next_point_name])
    perimeter_list.append(line_name)
    if i == len(example_points) - 1:
        break
    
    
#create lines from each object perimeter point to the camera origin point
for i in range(len(example_points)):
    point_name = f'point_{i}'
    plin_name = f'plin_{i}'
    next_line_name = f'line_{(i+1)%len(example_points)}'
    globals()[plin_name] = msh.model.geo.add_line(globals()[point_name], camera_point)

    if i == len(example_points) - 1:
        break

print (range(len(example_points)))

#Create curve loops adjacent to the camera origin
for i in range(len(example_points)):
    plin_name = f'plin_{i}'
    next_plin_name = f'plin_{(i+1)%len(example_points)}'
    next_line_name = f'line_{(i+1)%len(example_points)}'
    loop_name = f'loop_{i}'
    globals()[loop_name] = msh.model.geo.add_curve_loop([globals()[plin_name], globals()[next_line_name], -globals()[next_plin_name]])
    
    if i == len(example_points) - 1:
        break

# #Create perimeter loop
perimeter_handles = [globals()[line_name] for line_name in perimeter_list]
perimeter_loop = msh.model.geo.add_curve_loop(perimeter_handles)



mesh_test = msh.model.geo.add_plane_surface(globals()[loop_1])


#DISREGARD THESE COMMENTED OUT TESTS
# print(perimeter_handles)  
#perimeter_face_test = msh.model.geo.add_plane_surface([perimeter_loop])
# = msh.model.geo.add_plane_surface([loop_3])
#face_test2 = msh.model.geo.add_plane_surface([perimeter_loop])
#nodes = msh.model.mesh.getNodes(includeBoundary=False, 
                                    # returnParametricCoord=False)



# Create the relevant msh data structures from the msh model
msh.model.geo.synchronize()

# Generate mesh
msh.model.mesh.generate()

# Write mesh data
msh.write("GFG.msh")

# Run the graphical user interface event loop
msh.fltk.run()

# Finalize the gmsh API
msh.finalize()
