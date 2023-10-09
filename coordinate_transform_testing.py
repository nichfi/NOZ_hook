import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Given point on the plane
point_on_plane = np.array([0, 30, 0])

# Given ray vector
ray_vector = np.array([-.461, 3.2534, -374.4751])

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

# Define a meshgrid of points in the x-y plane
x_range = np.linspace(-100, 200, 100)  # Adjust the range as needed
y_range = np.linspace(-100, 200, 100)  # Adjust the range as needed
X, Y = np.meshgrid(x_range, y_range)

# Calculate the corresponding z-values for each point on the meshgrid using the plane equation
Z = (-A * X - B * Y + D) / C

# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create the surface plot for the plane
ax.plot_surface(X, Y, Z, cmap='viridis')

# Set labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('Plane Plot')

# Show the plot
plt.show()
