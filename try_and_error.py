import torch
import numpy
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
#import pytorch3d
from Class import Camera
from Class import Image
import srdf_helpers
import coordinate_lookup

new_camera = Camera()
new_Image = Image()
IMAGE_HEIGHT = new_camera.IMG_HEIGHT
IMAGE_WIDTH = new_camera.IMG_WIDTH

idx, intrinsic, extrinsic = new_camera.__getitem__(0)
dmap_idx, dmap = new_Image.__getitem__(0)

img = new_Image.imgs[0]
img2 = new_Image.imgs[1]
dmap_1 = new_Image.dmaps[0]
dmap_2 = new_Image.dmaps[1]

mask_1 = new_Image.masks[0]
mask_2 = new_Image.masks[1]

masked_dmap_1 = srdf_helpers.apply_mask(dmap_1, mask_1)
masked_dmap_2 = srdf_helpers.apply_mask(dmap_2, mask_2)


rays_o, rays_d = srdf_helpers.get_rays_tensor_torch(IMAGE_HEIGHT, IMAGE_WIDTH, intrinsic[0][0], extrinsic)
camera_origin = rays_o[0][0]

example_tensor_0 = torch.tensor([1,2,3])
example_tensor_1 = torch.tensor([[0,1],[1,1]])
example_tensor_1_transposed = torch.transpose(example_tensor_1, 0, 1)
example_tensor_2 = torch.tensor([1,2,3,3,2,1])
example_tensor_3 = torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]])
example_tensor_4 = torch.tensor([[[1,2,3],[4,5,6],[7,8,9]],[[90,3,3],[9,5,6],[7,5,9]]])
example_tensor_5 = torch.rand(1,2,3)
example_tensor_6 = torch.rand(4,4)

random_points = srdf_helpers.get_random_dmap_point_batch(masked_dmap_1, 3, IMAGE_HEIGHT)

unsqueezed_batch_sampled_dmap_coordinates = torch.unsqueeze(random_points, 0)
unsqueezed_masked_output = torch.reshape(masked_dmap_1, (1, 1, IMAGE_HEIGHT, IMAGE_WIDTH))
batch_dmap_values = coordinate_lookup.lookup_value_at_int(unsqueezed_batch_sampled_dmap_coordinates, unsqueezed_masked_output)
batch_dmap_values = torch.squeeze(batch_dmap_values, 0)

unsqueezed_rays_d = torch.unsqueeze(rays_d, 0)
unsqueezed_rays_d = torch.transpose(unsqueezed_rays_d, 2, 3).contiguous()
unsqueezed_rays_d = torch.transpose(unsqueezed_rays_d, 1, 2).contiguous()
batch_ray_vector = coordinate_lookup.lookup_value_at_int(unsqueezed_batch_sampled_dmap_coordinates, unsqueezed_rays_d)

point = srdf_helpers.calculate_point_with_depth_value(camera_origin, batch_ray_vector, batch_dmap_values)

point = torch.rand(4,4,3)
preds = torch.rand(4,4,3)
srdf = srdf_helpers.calculate_srdf(point,camera_origin,preds)
srdf_2 = torch.norm(srdf, dim=2)
min = torch.min(srdf_2, 1)

dmap5000 = cv2.imread("Images/diff/test.exr",  cv2.IMREAD_ANYDEPTH) 
dmap5000 = dmap_1 - torch.from_numpy(dmap5000)
dmap5000 = torch.unsqueeze(dmap5000, 0)
srdf_helpers.save_into_file(dmap5000, [0], name = "diff")



# vector__test_point_camera_origin = test_point - camera_origin
# length_vector__test_point_camera_origin = srdf_helpers.calc_vector_length(vector__test_point_camera_origin)

# tensor_one = torch.tensor([1])
# test_point_homogeneous = torch.cat((test_point, tensor_one)).float()
# point_camera = torch.sum(torch.linalg.inv(extrinsic) * test_point_homogeneous[None], -1)
# lenght_point_camera = srdf_helpers.calc_vector_length(point_camera[:3])

#tensor_4d = srdf_helpers.calculate_2d_point(tensor_3d, extrinsic, intrinsic)

# test = torch.multinomial(test[0], 1)


# arr = []
# for y in test:
#     for x in y:
        
#         arr.append(x.item())
# arr.sort()
# print(len(arr))


# img = cv2.imread("Images/dmaps/0.exr")
# print(img)
# directory = r"D:/1_My_Stuff/1_Studium_Medieninformatik/11._Semester/Bachelor/Mutli-View Reconstruction - Bachelor thesis/Images/result"
# os.chdir(directory)
# filename = "saved.exr"
# cv2.imwrite(filename, img)


# list = []
# list.append(3)
# example_tensor_1 = torch.tensor([1,2,3])
# example_tensor_2 = torch.tensor([4,5,6])
#example_tensor_3 = torch.tensor([[1,2,3],[4,5,6]])
# rays_o, rays_d = get_rays_tensor(IMAGE_HEIGHT, IMAGE_WIDTH, intrinsic[0][0], extrinsic)
# sampling_test = raysampling(rays_o[0][0], rays_d[0][0], 1, 5)

# closest_cameras_to_0 = new_camera.get_n_closest_cameras(idx,extrinsic,30)
# all_cams = new_camera.get_ordered_dotproduct_list_cameras(idx,extrinsic)

# #print(closest_cameras_to_0)
# #print(rays_o[0][0])
# #print(calculate_srdf(sampling_test[0],rays_o[0][0],))
# #print(sampling_test[1])

# #matplotlib code
# from mpl_toolkits import mplot3d

# import numpy as np
# import matplotlib.pyplot as plt

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(rays_o[0][0][0], rays_o[0][0][1], rays_o[0][0][2], color='green', s=50)
# for camera in closest_cameras_to_0:
#     idx1, intrinsic1, extrinsic1 = new_camera.__getitem__(camera[0])
#     rays_o1, rays_d1 = get_rays_tensor(IMAGE_HEIGHT, IMAGE_WIDTH, intrinsic1[0][0], extrinsic1)
#     ax.scatter(rays_o1[0][0][0], rays_o1[0][0][1], rays_o1[0][0][2], color='red', s=50)
#     ax.text(rays_o1[0][0][0], rays_o1[0][0][1], rays_o1[0][0][2], '%s' % (str(idx1)), size=10, zorder=1,  
#     color='k') 
#     view_dir = calc_viewing_direction(extrinsic1)
#     ax.quiver(rays_o1[0][0][0], rays_o1[0][0][1], rays_o1[0][0][2], view_dir[0], view_dir[1], view_dir[2], length=5, normalize=True)

# print(closest_cameras_to_0)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Closest Cameras to Camera 0 - Test 4')
# plt.show()

# # first import os and enable the necessary flags to avoid cv2 errors

# import os
# os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
# import cv2

# # then just type in following

# img = cv2.imread("Images/dmaps/7.exr")#  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
# '''
# you might have to disable following flags, if you are reading a semantic map/label then because it will convert it into binary map so check both lines and see what you need
# ''' 
# # img = cv2.imread(PATH2EXR) 
 
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# cv2.imshow('exr', img)
# cv2.waitKey(0)


