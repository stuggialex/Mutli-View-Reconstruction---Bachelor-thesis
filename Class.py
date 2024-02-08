import json
import torch
import torchvision
import numpy as np

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

from srdf_helpers import calc_viewing_direction
from srdf_helpers import calc_normalized_vector
from srdf_helpers import append_tensor

class Camera:
    def __init__(self):
        self.IMG_HEIGHT = 800
        self.IMG_WIDTH = 800
        self.RATIO = self.IMG_WIDTH / self.IMG_HEIGHT

        with open("quadrants.json", "r") as read_file:
          self.json = json.load(read_file)

        self.camera_angle_x = self.json['camera_angle_x']
        self.camera_angle_y = self.RATIO * self.camera_angle_x
        self.poses = []

        for frame in self.json['frames']:
             self.poses.append(frame['transform_matrix'])
        self.poses_length = len(self.poses)
        self.poses = torch.FloatTensor(self.poses)

        self.intrinsic = torch.tensor(np.zeros(shape=(3, 3), dtype=np.float32))
        focal_x = (self.IMG_WIDTH / 2) / np.tan(self.camera_angle_x / 2)
        focal_y = (self.IMG_HEIGHT / 2) / np.tan(self.camera_angle_y / 2)
        self.intrinsic[0][0] = focal_x
        self.intrinsic[1][1] = focal_y
        self.intrinsic[2][2] = 1
    
    def  __getitem__(self, idx):
          return idx, self.intrinsic, self.poses[idx]
    
    def to_cuda(self, device):
        self.poses.to(device)
        self.intrinsic.to(device)

    def to_cpu(self):
        self.poses.cpu()
        self.intrinsic.cpu()
    
    def get_item_tensor(self, list):
        poses = torch.index_select(self.poses, 0, list)
        intrinsic = self.intrinsic.unsqueeze(0)
        for num in range(poses.shape[0]-1):
            intrinsic = torch.cat((intrinsic, self.intrinsic.unsqueeze(0)))
        return intrinsic, poses
    
    def get_ordered_dotproduct_list_cameras(self, idx, extrinsic):
        count = 0
        dotproduct_list = []
        viewing_direction = calc_viewing_direction(extrinsic)

        for pose in self.poses:
            if count != idx:
                n_cameras_idx, n_cameras_intrinsic, n_cameras_extrinsic = self.__getitem__(count)
                n_cameras_viewing_direction = calc_viewing_direction(n_cameras_extrinsic)

                dotproduct = np.dot(calc_normalized_vector(viewing_direction), calc_normalized_vector(n_cameras_viewing_direction))
                dotproduct_list.append([count, dotproduct])
            count +=1
        
        dotproduct_list = sorted(dotproduct_list, key=lambda x: x[1], reverse=True)

        return dotproduct_list
    
    def get_n_closest_cameras(self, idx, extrinsic, num_cam):
          ordered_list = self.get_ordered_dotproduct_list_cameras(idx, extrinsic)
          return ordered_list[:num_cam]
    
class Image:
    def __init__(self):
        #dummy -> select fitting images and hardcode it into the array
        self.arr = [0,57,95,155]
        arr_imgs = []
        arr_masks = []
        arr_dmaps = []
        for idx, item in enumerate(self.arr):
            img = cv2.imread("Images/imgs/"+str(item)+ ".jpg")
            mask = cv2.imread("Images/masks/"+str(item)+ "_mask.jpg")
            dmap = cv2.imread("Images/dmaps/" + str(item) + ".exr",  cv2.IMREAD_ANYDEPTH) 
            #dmap = cv2.cvtColor(dmap, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img)
            mask = torch.from_numpy(mask)
            dmap = torch.from_numpy(dmap)
            arr_imgs.append(img)
            arr_masks.append(mask)
            arr_dmaps.append(dmap)
            self.imgs = torch.stack(arr_imgs)
            self.masks = torch.stack(arr_masks)
            self.dmaps = torch.stack(arr_dmaps)

    def  __getitem__(self, idx):
          return idx, self.dmaps[idx]
    
    def activate_gradients(self):
        self.dmaps = self.dmaps.float().requires_grad_()
    
    def get_group_of_cams_as_tensor(self, list):
        for group in list:
            idx = self.get_dmap_idx_with_real_index(group[0])
            if not 'tensor' in locals():
                tensor = append_tensor(self.dmaps[idx])
                mask = append_tensor(self.masks[idx])
            else:
              tensor = append_tensor(tensor, self.dmaps[idx])
              mask = append_tensor(mask, self.masks[idx])
        return tensor, mask
    
    def get_dmap_idx_with_real_index(self, idx):
        fake_idx = self.arr.index(idx)
        return fake_idx
    
    def salt_and_pepper(self):
        divider = 800
        for dmap in self.dmaps:
            flat_image = torch.flatten(dmap)
            idx = torch.multinomial(flat_image, 32000)
            for item in idx:
                column_row_tuple = divmod(item.item(), divider)
                dmap[column_row_tuple[0]][column_row_tuple[1]] = 0
        return
        
    def gaussian_noise(self):
        noise = abs(torch.randn(800,800))
        for dmap in self.dmaps:
            dmap += (0.1**0.5)*noise

    def gaussian_noise(self, idx):
        noise = abs(torch.randn(800,800))
        self.dmaps[idx] += (0.1**0.5)*noise
    
    def to_cuda(self, device):
        self.imgs.to(device)
        self.masks.to(device)
        self.dmaps.to(device)

    def to_cpu(self):
        self.imgs.cpu()
        self.masks.cpu()
        self.dmaps.cpu()

