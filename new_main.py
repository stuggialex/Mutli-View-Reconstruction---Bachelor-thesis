from Class import Camera, Image
import srdf_helpers
import coordinate_lookup
import visualizer
import torchvision

import torch

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

#initialize data
cameraSet = Camera()
imageSet = Image()
testImageSet = Image()

#current size of images
IMAGE_HEIGHT = cameraSet.IMG_HEIGHT
IMAGE_WIDTH = cameraSet.IMG_WIDTH

#used in the loop
ITERATION_NUM = 5000

#used in raysampling
SAMPLING_INTERVALL = 0.05
SAMPLING_AMOUNT = 10
HALF_SAMPLE_DISTANCE = SAMPLING_INTERVALL * SAMPLING_AMOUNT /  2

#currently used cameras, data available
CAMERAS = [0,57,95,155]
CAMERAS_WITHOUT_0 = [57,95,155]

#needed number to calculate group_of_cams
GROUP_SIZE = len(CAMERAS)

#during iteration the number of cameras checked for srdf
CAMERA_BATCH_SIZE = 1
DMAP_POINT_BATCH_SIZE = 4000

#SRDF parameter
SIGMA = 5
GAMMA = 1

#dictionary for file names
dict = {"iteration_number": ITERATION_NUM,
        "sampling_intervall": SAMPLING_INTERVALL,
        "sampling_amount": SAMPLING_AMOUNT,
        "depth_map_point_batch_size": DMAP_POINT_BATCH_SIZE,
        "sigma": SIGMA,
        "gamma": GAMMA}
dict_as_str = srdf_helpers.dict_to_string(dict)

#current batch, camera chosen for this example
chosen_camera = 0

#getting depth values, intrinsic and extrinsic(c2w) of the chosen camera
camera_idx, intrinsic, extrinsic = cameraSet.__getitem__(chosen_camera)
dmap_of_chosen_camera = imageSet.dmaps[camera_idx]

#gets mask in the shape of 800x800x3
mask_of_chosen_camera = imageSet.masks[camera_idx]

#loading in the dmaps into ADAM
imageSet.gaussian_noise(0)
imageSet.activate_gradients()
dmaps = imageSet.dmaps
test_dmaps =testImageSet.dmaps
clone_dmaps = torch.clone(dmaps)
adam = torch.optim.Adam([dmaps])

#main loop
def loop():
    #get a group of cams, in this case the only 4 cams i have chosen
    group_of_cams = cameraSet.get_n_closest_cameras(camera_idx, extrinsic, GROUP_SIZE)
    """for now disregard this because i have already my chosen group, change later GROUP_SIZE to a proper number"""
    tensor_group_of_cams = imageSet.get_group_of_cams_as_tensor(group_of_cams)

    for x in range(ITERATION_NUM):
        print(x)
        #reset gradients
        adam.zero_grad()

        #calculate origin and the pixel rays
        rays_o, rays_d = srdf_helpers.get_rays_tensor_torch(IMAGE_HEIGHT, IMAGE_WIDTH, intrinsic[0][0], extrinsic)
        origin = rays_o[0][0]

        masked_output = srdf_helpers.apply_mask(dmap_of_chosen_camera, mask_of_chosen_camera)

        #randomly selected points, that will be used for the pipeline, calculated with multinomial
        #shape: nx2
        batch_sampled_dmap_coordinates = srdf_helpers.get_random_dmap_point_batch(masked_output, DMAP_POINT_BATCH_SIZE, IMAGE_HEIGHT)
        
        #corresponding depth map values to batch_sampled_dmap_points
        #shape: nx1
        unsqueezed_batch_sampled_dmap_coordinates = torch.unsqueeze(batch_sampled_dmap_coordinates, 0)
        unsqueezed_masked_output = torch.reshape(masked_output, (1, 1, IMAGE_HEIGHT, IMAGE_WIDTH))
        batch_dmap_values = coordinate_lookup.lookup_value_at_int(unsqueezed_batch_sampled_dmap_coordinates, unsqueezed_masked_output)
        batch_dmap_values = torch.squeeze(batch_dmap_values, 0)
        
        #corresponding ray vectors to batch_sampled_dmap_points
        #shape: nx3
        unsqueezed_rays_d = srdf_helpers.transpose_rays_tensor(rays_d)
        batch_ray_vector = coordinate_lookup.lookup_value_at_int(unsqueezed_batch_sampled_dmap_coordinates, unsqueezed_rays_d)
        batch_ray_vector = torch.squeeze(batch_ray_vector, 0)
        
        #resulting 3d point from the randomly selected batch_sampled_dmap_points
        #shape: nx3
        predicted_point = srdf_helpers.calculate_point_with_depth_value(origin, batch_ray_vector, batch_dmap_values)

        #tensor of points sampled around the predicted_point
        #shape: mxnx3   m: number of sampled points
        sampling_tensor = srdf_helpers.raysampling(predicted_point - HALF_SAMPLE_DISTANCE * batch_ray_vector, batch_ray_vector, SAMPLING_INTERVALL, SAMPLING_AMOUNT)
        sampling_tensor = torch.transpose(sampling_tensor,0,1)

        srdf_tensor = torch.ones(DMAP_POINT_BATCH_SIZE, SAMPLING_AMOUNT)

        #go through the group of cameras and check their side
        for idx_camera, camera in enumerate(CAMERAS):
            #variable instatiation
            srdf_camera_idx, srdf_intrinsic, srdf_extrinsic = cameraSet.__getitem__(camera)
            srdf_rays_o, srdf_rays_d = srdf_helpers.get_rays_tensor_torch(IMAGE_HEIGHT, IMAGE_WIDTH, srdf_intrinsic[0][0], srdf_extrinsic)
            srdf_origin = srdf_rays_o[0][0]
            srdf_dmap = imageSet.dmaps[idx_camera]
            srdf_mask = imageSet.masks[idx_camera]

            srdf_masked_output = srdf_helpers.apply_mask(srdf_dmap, srdf_mask)

            #project the sampled 3d points into the image plane of the cameras
            #shape:
            point_2d =srdf_helpers.calculate_2d_point_batch(sampling_tensor, srdf_extrinsic, srdf_intrinsic)
            point_2d = torch.reshape(point_2d, (1,point_2d.shape[0]*point_2d.shape[1],2))

            #get the necessary depth map value and the corresponding ray for the next calculation
            #shape:
            #shape:
            srdf_masked_output = torch.unsqueeze(srdf_masked_output, 0)
            srdf_masked_output = torch.unsqueeze(srdf_masked_output, 0)
            srdf_dmap_value = coordinate_lookup.lookup_value_at(point_2d, srdf_masked_output)
            unsqueezed_srdf_rays_d = srdf_helpers.transpose_rays_tensor(srdf_rays_d)
            srdf_ray_vector = coordinate_lookup.lookup_value_at(point_2d, unsqueezed_srdf_rays_d)
            
            #now calculate the point that the selected group of cameras think is the surface
            #shape:
            srdf_predicted_point = srdf_helpers.calculate_point_with_depth_value(srdf_origin, srdf_ray_vector, srdf_dmap_value)
            srdf_predicted_point = torch.reshape(srdf_predicted_point, (DMAP_POINT_BATCH_SIZE,SAMPLING_AMOUNT,3))
            #calculate the srdf
            #shape:
            step = torch.norm(srdf_helpers.calculate_srdf(sampling_tensor, srdf_origin, srdf_predicted_point), dim=2)
            srdf_tensor *= step
            
        mask_srdf_tensor = torch.ge(srdf_tensor, 1000)  
        srdf_tensor = srdf_tensor.masked_fill(mask_srdf_tensor, 100)  
        min = torch.min(srdf_tensor, 1).values
        loss = torch.mean(min[min!=100])
        if x % 100 == 99:
            writer.add_image("result", dmaps, global_step=x)
        writer.add_scalar("Loss/train", loss, x)
        loss.backward()

        adam.step()

    writer.flush()
        
    for idx, test_dmap in enumerate(test_dmaps):
        is_zero = test_dmap - dmaps[idx]
        tuple_nonzero = torch.nonzero(is_zero)
        if idx == 0:
            tensor_difference = srdf_helpers.append_tensor(is_zero)
        else:
            tensor_difference = srdf_helpers.append_tensor(tensor_difference, is_zero)

    srdf_helpers.save_into_file(dmaps, CAMERAS, name="_result", variable_list=dict_as_str, bool=False)



loop()
