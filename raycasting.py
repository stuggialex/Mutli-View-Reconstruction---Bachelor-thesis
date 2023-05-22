import json
from mpl_toolkits import mplot3d

%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')



"""
Simple raycasting

1. From one camera send out ray(s) #check out the data, how do they represent the camera
2. Sample points along the ray(s)
3. compare the sampled points with other cameras

"""

"""
 The first 3 columns are the +X, +Y, and +Z defining the camera orientation, and the X, Y, Z values define the origin. 
 The last row is to be compatible with homogeneous coordinates.
         [+X0 +Y0 +Z0 X]
         [+X1 +Y1 +Z1 Y]
         [+X2 +Y2 +Z2 Z]
         [0.0 0.0 0.0 1]

work with pytorch tensor
"""

"""
function that gets the transform matrix for one camera from the dataset
"""
def get_camera ():
    with open("quadrants.json", "r") as read_file:
        cameras = json.load(read_file)
    camera_arr = cameras['frames'][0]['transform_matrix']
    return camera_arr

def get_camera_origin(camera):
    arr = []
    for x in range(3):
        arr.append(camera[x][3])
    return arr

print(get_camera())

def visualize_camera(camera):
    x = camera[0][3]
    y = camera[1][3]
    z = camera[2][3]