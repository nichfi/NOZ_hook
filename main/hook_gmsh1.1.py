import gmsh as msh
import numpy as np
import sys

# gmsh 4.11.1
# Initialize gmsh:
msh.initialize(sys.argv)

lc = 1e-1

# Outline of object
example_points = [[-3, 3, 0], [3, 3, 0], [3, -3, 0], [-3, -3, 0]]

# Camera position
camera_point = msh.model.geo.add_point(4, 4, 10, lc)

# Establish vertices
point_list = []

for i, point_coords in enumerate(example_points):
    point_name = f'point_{i}'
    point_list.append(msh.model.geo.add_point(point_coords[0], point_coords[1], point_coords[2], lc))

# Create a list to fill with all perimeter lines
perimeter_list = []

# Create outlines of the object
for i in range(len(example_points)):
    point_name = f'point_{i}'
    next_point_name = f'point_{(i+1)%len(example_points)}'
    line_name = f'line_{i}'
    line_handle = msh.model.geo.add_line(point_list[i], point_list[(i+1)%len(example_points)])
    perimeter_list.append(line_handle)
    if i == len(example_points) - 1:
        break

# Create lines from each object perimeter point to the camera origin point
plin_list = []

for i in range(len(example_points)):
    point_name = f'point_{i}'
    plin_name = f'plin_{i}'
    next_line_name = f'line_{(i+1)%len(example_points)}'
    plin_handle = msh.model.geo.add_line(point_list[i], camera_point)
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
    loop_handle = msh.model.geo.add_curve_loop([-plin_list[i], perimeter_list[i], plin_list[(i+1)%len(example_points)]])
    loop_list.append(loop_handle)
    if i == len(example_points) - 1:
        break

# Create perimeter loop
perimeter_handles = [line_handle for line_handle in perimeter_list]
perimeter_loop = msh.model.geo.add_curve_loop(perimeter_handles)

mesh_test = msh.model.geo.add_plane_surface([loop_list[1]])
mesh_test = msh.model.geo.add_plane_surface([loop_list[2]])
mesh_test = msh.model.geo.add_plane_surface([loop_list[3]])
mesh_test = msh.model.geo.add_plane_surface([loop_list[0]])


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
