from raycasting import calculate_viewing_direction
import numpy as np
from camera import Camera

CameraSet = Camera()
"""

function to process camera data
input: camera, rays (with camera values generate rays)
check the dataset which is being used
what type of camera or depth maps?




"""


"""
calculate for each camera/depth map the distance to point x

Args:
    x: (Art von Vektor) point in 3D space
    camera: camera data like position and rotation
    dmaps: (Menge von Bilder bzw. verarbeitet) depth maps which belongs
           to a camera

Returns:
    srdf: (Menge von Vektoren) signed distances to the cameras from the 
          depth maps
"""

#probably need a function that searches for the point on the depth map
def get_point_from_depthmap(camera, dmap):
    view_dir = calculate_viewing_direction(camera)
    #calculate intersection between camera and depth map
    
"""
Args:
    x: sampled point of one of the selected rays
    camera: origin of camera, doesn't necessarily need to be the one which the ray belongs to
    dmap: guessed 3D point, that should be close to the real surface
"""
#maybe needs a check if srdf gets negative, then we dont have to sample further
def calculate_srdf(x, camera, dmap):
    #calculate the distance between camera and point Z

    #with the now calculated vector Z use normed direction to calculate
    #distance to surface D
    camera_tuple = (camera[0][3], camera[1][3], camera[2][3])

    #calculate signed distance
    z = np.subtract(x, camera_tuple)
    srdf = np.subtract(dmap, z)
    return srdf
"""
helper function to check if all the elements in a list are the same 
"""
def check_list_elements_identical(list):
   return list.count(list[0]) == len(list)

#should be something with pytorch, I could also implement it myself cause
#its not that hard
#watch out for matrix multiplication, if something looks off, this might cause the problem
#maybe put the inverse into the class Camera to save processing time
def calculate_2d_point(point_3d, camera_matrix):
    w2c = np.linalg.inv(camera_matrix)
    #point_homogeneous = np.append(point_3d, 1)
    #the output is a 3d point, how do I get the 2d point?
    c2w_point = np.dot(w2c, point_3d)
    x = c2w_point[0]/-c2w_point[2]
    y = c2w_point[1]/-c2w_point[2]
    return x, y


"""
checks if the cameras sampling the same point are on a surface or not

Args:
    cameras: an array of cameras(selected through the group function)
"""
#with this function sample some points and check if the sum is the lowest
#before this function can work, I need to find a way to calculate the point on 
#the depth map that corresponds to the sampled point
#check phind folder
#check pytorch doc
#instead of using the camera data directly, better is to use the idx
#and with that call the necessary data
def check_surface(cameras, camera_idxs, sampled_point, dmaps):
    srdfs = []
    relative_srdfs = []
    n = 0
    for camera_idx in camera_idxs:
        intrinsic, poses = CameraSet.__getitem__(camera_idx)
        camera_poses = cameras[camera_idx]
        point2d_x, point2d_y = calculate_2d_point(sampled_point, intrinsic)
        estimated_distance = dmaps[n]
        estimated_distance = estimated_distance[int(point2d_x)]
        estimated_distance = estimated_distance[int(point2d_y)]
        estimated_distance = estimated_distance[0]
        #need to get the camera id, so you can get the dmap value

        srdfs.append(calculate_srdf(sampled_point, camera_poses, estimated_distance))
        n+=1
    for srdf in srdfs:
        relative_srdfs.append(srdf[2] - srdfs[0][2])
    return sum(relative_srdfs)

#tests

#testing get_point_from_depthmap

#testing srdf function
#print(calculate_srdf((1,2,3),[[-0.9072327017784119,0.12652991712093353,-0.40114709734916687,-1.805161952972412],[-0.4206290543079376,-0.27290573716163635,0.8652130961418152,3.893458843231201],
#                    [0.0,0.9536836743354797,0.30081117153167725,1.3536502122879028],[0.0,0.0,0.0,1.0]], (0,0,0)))

#testing check_list_elements_identical

#testing check_surface