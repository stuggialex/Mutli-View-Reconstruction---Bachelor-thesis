import json
import torch
import numpy as np

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
                   pass group.append(element)
    
    def get_n_closest_cameras(idx, num_cam):
        pass   

c = Camera()

print(c.intrinsic)