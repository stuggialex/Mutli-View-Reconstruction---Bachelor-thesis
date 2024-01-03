import json
import torch
import numpy as np
import torch
import matplotlib.pyplot as plt
from raycasting import calculate_viewing_direction
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

class Camera:

     def __init__(self):
        self.IMG_HEIGHT = 800
        self.IMG_WIDTH = 800
        self.RATIO = self.IMG_WIDTH / self.IMG_HEIGHT
        with open("quadrants.json", "r") as read_file:
          self.json = json.load(read_file)
        self.poses = []
        for frame in self.json['frames']:
             self.poses.append(frame['transform_matrix'])
        self.camera_angle_x = self.json['camera_angle_x']
        self.camera_angle_y = self.RATIO * self.camera_angle_x
        self.intrinsic = torch.tensor(np.zeros(shape=(3, 3), dtype=np.float32))
        focal_x = (self.IMG_WIDTH / 2) / np.tan(self.camera_angle_x / 2)
        focal_y = (self.IMG_HEIGHT / 2) / np.tan(self.camera_angle_y / 2)
        self.intrinsic[0][0] = focal_x
        self.intrinsic[1][1] = focal_y
        self.intrinsic[2][2] = 1

        self.length = len(self.poses)

     def  __getitem__(self, idx):
          return self.intrinsic, self.poses[idx]
    
     def get_group(self):
         group = []
         for element in self.poses:
              if np.dot(self.poses[0],element)>0.75:
                    group.append(element)

     """
     orders a list in ascending order. The function picks a camera and checks the dot product between other cameras while ordering it 
     """
     def get_ordered_dotproduct_list_cameras(self, idx):
          count = 0
          dprod_list = torch.Tensor()
          viewing_dir = calculate_viewing_direction(self.poses[idx])
          for pose in self.poses:
               if count != idx:
                    n_cameras = calculate_viewing_direction(pose)
                    dotproduct = np.dot(viewing_dir, n_cameras)
                    #we save the idx first and after that the dot product
                    ordered_list = torch.stack((ordered_list, [count, dotproduct]))
                    count+= 1
                    #index has to come with this list, otherwise its hard to see
                    #which camera the value belongs to
                    #i can use the "get group" function, it kinda calculates
                    #the dot product, or i can rename it
          return torch.sort(dprod_list, dim=1)
               
     def get_n_closest_cameras(self, idx, num_cam):
          ordered_list = self.get_ordered_dotproduct_list_cameras(idx)
          return ordered_list[:num_cam]

          #calculate viewing direction compare it with others
          #i could write a function that calculates an ordered list, where
          #the first element has the most similar angle to the selected camera
          #and the last element is the furthest
          #this might simplify selecting groups, because then I can just cut off
          #the later list elements that Im not interested int


class Image:
     #must contain the images and later the corresponding depth maps aswell
     def __init__(self):
          self.imgs = []
          self.dmaps = []

          #dummy -> select fitting images and hardcode it into the array
          arr = [0, 1, 10, 23]
          for x in arr:
               #if this doesnt work, try using tensors
               #self.imgs.append(plt.imread("Images/img/" + x + ".exr")) #insert image path
               #img = cv2.imread("Images/imgs/" + str(x) + ".exr",  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) 
               #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
               #self.imgs.append(img) 

               dmap = cv2.imread("Images/omni_depth/" + str(x) + ".exr",  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) 
               dmap = cv2.cvtColor(dmap, cv2.COLOR_BGR2RGB)
               self.dmaps.append(dmap) 
     
     def updatedepthmaps(self):
          pass

     def depthmapfusion(self):
          pass

#test = Camera()
#print(test.get_ordered_list_cameras(0))

#test = Image()
#print(test.dmaps[0][799][0])