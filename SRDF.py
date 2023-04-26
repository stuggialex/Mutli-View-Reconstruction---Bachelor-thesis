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
def srdf(x, camera, dmaps):
    #calculate the distance between camera and point Z

    #with the now calculated vector Z use normed direction to calculate
    #distance to surface D

    #calculate signed distance
