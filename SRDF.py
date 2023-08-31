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
#maybe needs a check if srdf gets negative, then we dont have to sample further
def srdf(x, camera, dmaps):
    #calculate the distance between camera and point Z

    #with the now calculated vector Z use normed direction to calculate
    #distance to surface D

    #calculate signed distance
    z = x - camera
    srdf = dmaps - z
    return srdf
"""
helper function to check if all the elements in a list are the same 
"""
def check_list__elements_identical(list):
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
