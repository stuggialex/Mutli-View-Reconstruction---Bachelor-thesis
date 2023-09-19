import json
import torch
import numpy as np
import torch
from raycasting import calculate_viewing_direction

class Camera:

     def __init__(self):
        IMG_HEIGHT = 800
        IMG_WIDTH = 800
        RATIO = IMG_WIDTH / IMG_HEIGHT
        with open("quadrants.json", "r") as read_file:
          self.json = json.load(read_file)
        self.poses = []
        for frame in self.json['frames']:
             self.poses.append(frame['transform_matrix'])
        camera_angle_x = self.json['camera_angle_x']
        camera_angle_y = RATIO * camera_angle_x
        self.intrinsic = torch.tensor(np.zeros(shape=(3, 3), dtype=np.float32))
        focal_x = (IMG_WIDTH / 2) / np.tan(camera_angle_x / 2)
        focal_y = (IMG_HEIGHT / 2) / np.tan(camera_angle_y / 2)
        self.intrinsic[0][0] = focal_x
        self.intrinsic[1][1] = focal_y
        self.intrinsic[2][2] = 1

     def  __getitem__(self, idx):
          return idx, self.intrinsic, self.poses[idx]
    
     def get_group(self):
         group = []
         for element in self.poses:
              if np.dot(self.poses[0],element)>0.75:
                    group.append(element)

     def get_ordered_list_cameras(self, idx):
          # TODO: order the list
          cam = self.__getitem__(idx)
          count = 0
          ordered_list = torch.Tensor()
          viewing_dir = calculate_viewing_direction(self.poses[idx])
          for pose in cam.poses:
               if count != idx:
                    np.dot(viewing_dir, pose)
                    #we save the idx first and after that the dot product
                    ordered_list = torch.stack((ordered_list, [count, np.dot]))

                    #index has to come with this list, otherwise its hard to see
                    #which camera the value belongs to
                    #i can use the "get group" function, it kinda calculates
                    #the dot product, or i can rename it
          return ordered_list
               
     def get_n_closest_cameras(self, idx, num_cam):
          ordered_list = self.get_ordered_list_cameras(idx)
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
          pass
     
     def getdepthmaps(self):
          pass
     
     def updatedepthmaps(self):
          pass

#test = Camera()
#print(test.get_ordered_list_cameras(0))