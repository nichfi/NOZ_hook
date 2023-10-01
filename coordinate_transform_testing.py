# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 17:54:29 2023

@author: nicko
"""

import numpy as np

# Given point on the plane
point_on_plane = np.array([89.915, 97.855, 0])

# Given ray vector
ray_vector = np.array([145.76496088, 137.82496088, 235.67996088])

# Calculate the normal vector by normalizing the ray vector
normal_vector = ray_vector / np.linalg.norm(ray_vector)

# Calculate the constant 'D' using the dot product of the normal vector and the point
D = np.dot(normal_vector, point_on_plane)

# Extract components of the normal vector (A, B, C)
A, B, C = normal_vector

# Create a variable for the plane equation
plane_equation = f"{A:.4f}x + {B:.4f}y + {C:.4f}z = {D:.4f}"

# Print the plane equation
print("Plane Equation:", plane_equation)

# You can return the 'plane_equation' variable if needed
# return plane_equation
