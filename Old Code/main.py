import raycasting
import SRDF
import camera
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np


ITERATION_NUM = 1
SAMPLING_INTERVALL = 0.5
SAMPLING_AMOUNT = 10
CAMERAS = [0,1,10,23] #hatte irgendeine Funktion die Probleme hatte
#einfach array length von CAMERAS nehmen

def main():
    #main pipeline loop

    #load files, create class

    #little bit hard coded

    ImageSet = camera.Image()
    #random_camera_id = np.random.randint(ImageSet.length)
    cameraSet = camera.Camera()
    camera_idx = 0
    intrinsic, poses = cameraSet.__getitem__(camera_idx)

    #maybe add a function that helps getting the camera position

    #hardcoded image size -> change later
    rays_o, rays_d = raycasting.get_rays_np(800, 800, intrinsic[0][0], poses)
    origin = rays_o[0][0]
    #loop starts
    for x in range(ITERATION_NUM):
        #raycasting combined with sampling + SRDF
        #do i need to always generate new rays?
        #make the raycasting and overall function 
        #how do I efficiently search for rays? Alot of rays mean alot of computing time
        #Any way to optimize it?
        sampled_points = raycasting.raysampling(rays_d, origin, SAMPLING_INTERVALL, SAMPLING_AMOUNT)
        potential_surface_points = torch.empty(0, 25)
        for sampled_point in sampled_points:
            surface_num = SRDF.check_surface(cameraSet.poses, CAMERAS, sampled_point, ImageSet.dmaps)
            tensor_element = torch.zeros(2,3)
            tensor_element[0] = torch.from_numpy(sampled_point)
            tensor_element[1] = surface_num
            potential_surface_points = torch.cat((potential_surface_points, tensor_element), dim=0)
        #only have the srdf, but i need to know which point the srdf corresonds to
        #print(torch.sort(potential_surface_points, dim=1))
        #update depthmaps
        ImageSet.updatedepthmaps()

        ImageSet.depthmapfusion()

    #output

#main()

#result = cv2.imread("Images/omni_depth/23.exr",  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) 
#result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

#f = plt.figure()
#ax = f.add_subplot()
#ax.imshow(result[:, :, 0])
#plt.show()

Camera = camera.Camera() 
print(Camera.get_ordered_dotproduct_list_cameras(0))