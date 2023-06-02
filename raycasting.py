import json
import torch
from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

"""
Simple raycasting

1. From one camera send out ray(s) #check out the data, how do they represent the camera
2. Sample points along the ray(s)
3. compare the sampled points with other cameras

"""

def get_camera (n):
    """
    Gets the transform matrix for the first n cameras from the dataset

    Args:
        n: int

    Returns:
        tensor with the camera data
    """
    
    with open("quadrants.json", "r") as read_file:
        cameras = json.load(read_file)
    camera_arr = cameras['frames'][n]['transform_matrix']
    return torch.tensor(camera_arr)

def get_rotation_matrix(camera):
    return camera[:3, :3]

def calculate_viewing_direction(camera_view):
    """
    Calculates the viewing direction of a camera
    with matrix multiplication 

    Args:
        camera_view:3x3 tensor

    Returns:
        3x3 tensor
    """

    null_vec = torch.tensor([0, 0, 1], dtype=torch.float32)
    return torch.matmul(get_rotation_matrix(camera_view), null_vec * -1)

def visualize_camera(camera):
    """
    Args:
        camera:4x4 tensor
    """
    x = camera[0][3]
    y = camera[1][3]
    z = camera[2][3]
    u, v, w = calculate_viewing_direction(camera)
    ax.quiver(x, y, z, u, v, w, length=2, normalize=True)

for n in range(5):
    visualize_camera(get_camera(n))

# Set the labels of the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set the x, y, and z limits
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_zlim(-4, 4)

plt.show()