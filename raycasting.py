import json
import random
import torch
from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

IMG_HEIGHT = 800
IMG_WIDTH = 800
RATIO = IMG_WIDTH / IMG_HEIGHT

"""
Simple raycasting

1. From one camera send out ray(s) #check out the data, how do they represent the camera
2. Sample points along the ray(s)
3. compare the sampled points with other cameras
"""

def get_camera_data ():
    """
    Gets the camera angle for the intrinsic matrix
    aswell as the transform matrices for the cameras from the dataset

    Returns:
        array
        float
    """
    
    with open("quadrants.json", "r") as read_file:
        cameras = json.load(read_file)
    camera_array = cameras['frames']
    return camera_array, cameras['camera_angle_x']

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
    

def calculate_intrinsic(camera_angle_x):
    """
    """
    camera_angle_y = RATIO * camera_angle_x
    intrinsic = torch.tensor(np.zeros(shape=(3, 3), dtype=np.float32))
    focal_x = (IMG_WIDTH / 2) / np.tan(camera_angle_x / 2)
    focal_y = (IMG_HEIGHT / 2) / np.tan(camera_angle_y / 2)
    intrinsic[0][0] = focal_x
    intrinsic[1][1] = focal_y
    intrinsic[2][2] = 1
    return intrinsic

def get_rays_np(H, W, focal, c2w):
    """
    Get ray origins, directions from a pinhole camera.
    """
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d

def visualize_rays(focal, c2w):
    """
    Args:
    focal length of the camera
    transformation matrix of the camera
    """
    rays_o, rays_d = get_rays_np(IMG_HEIGHT, IMG_WIDTH, focal, torch.tensor(c2w).numpy())
    for m in range(1, 801,100):
        for n in range(1, 801,100):
            ax.quiver(rays_o[m][n][0], rays_o[m][n][1], rays_o[m][n][2], rays_d[m][n][0], rays_d[m][n][1], rays_d[m][n][2], length=5, normalize=True)


camera_data, camera_angle_x = get_camera_data()
for n in range(5):
    camera = camera_data[n]['transform_matrix']
    intrinsic = calculate_intrinsic(camera_angle_x)
    #visualize_rays(intrinsic[0][0], camera)
    visualize_camera(torch.tensor(camera))

# Set the labels of the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set the x, y, and z limits
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_zlim(-4, 4)
plt.show()