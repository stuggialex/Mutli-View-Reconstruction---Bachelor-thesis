from raycasting import calculate_viewing_direction
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
    

#maybe needs a check if srdf gets negative, then we dont have to sample further
def srdf(x, camera, dmap):
    #calculate the distance between camera and point Z

    #with the now calculated vector Z use normed direction to calculate
    #distance to surface D

    camera_tuple = (camera[0][3], camera[1][3], camera[2][3])

    #calculate signed distance
    z = x - camera_tuple
    srdf = dmap - z
    return srdf
"""
helper function to check if all the elements in a list are the same 
"""
def check_list_elements_identical(list):
   return list.count(list[0]) == len(list)

"""
checks if the cameras sampling the same point are on a surface or not

Args:
    cameras: an array of cameras(selected through the group function)
"""
#with this function sample some points and check if the sum is the lowest
def check_surface(cameras, sampled_point, dmaps):
    srdfs = []
    relative_srdfs = []
    for camera in cameras:
        srdfs.append(srdf(sampled_point, camera, dmaps))
    for srdf in srdfs:
        relative_srdfs.append(srdf - srdfs[0])
    return sum(relative_srdfs)

#tests

#testing get_point_from_depthmap

#testing srdf function
print(srdf((1,2,3),[[-0.9072327017784119,0.12652991712093353,-0.40114709734916687,-1.805161952972412],[-0.4206290543079376,-0.27290573716163635,0.8652130961418152,3.893458843231201],
                    [0.0,0.9536836743354797,0.30081117153167725,1.3536502122879028],[0.0,0.0,0.0,1.0]], (0,0,0)))

#testing check_list_elements_identical

#testing check_surface
