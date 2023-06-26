import gmsh as msh
import numpy as np
import sys

# Initialize msh:
msh.initialize()

lc = 1e-2
example_points = [(-1, 1, 3), (1, 1, 3), (1, -1, 3)]

for i, point_coords in enumerate(example_points):
    variable_name = f'point_{i}'
    globals()[variable_name] = msh.model.geo.add_point(point_coords[0], point_coords[1], point_coords[2], lc)

# Example usage of the created variables
print(point_0)  # Output: The handle/ID of the first point
print(point_1)  # Output: The handle/ID of the second point
print(point_2)  # Output: The handle/ID of the third point

for i in (variable_name):
    line_name = f'line_{i}'
    globals()[line_name] = msh.model.geo.add_line([variable_name],point_2)


print (variable_name)
# Create the relevant msh data structures
# from msh model.
msh.model.geo.synchronize()
 
# Generate mesh:
msh.model.mesh.generate()
 
# Write mesh data:
msh.write("GFG.msh")
 
# Creates  graphical user interface
if 'close' not in sys.argv:
    msh.fltk.run()
 
# It finalize the msh API
msh.finalize()