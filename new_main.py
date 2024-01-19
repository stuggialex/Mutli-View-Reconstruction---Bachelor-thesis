from Class import Camera, Image
import srdf_helpers
import coordinate_lookup
import visualizer
import torchvision

import torch

#initialize data
cameraSet = Camera()
imageSet = Image()
testImageSet = Image()

#current size of images
IMAGE_HEIGHT = cameraSet.IMG_HEIGHT
IMAGE_WIDTH = cameraSet.IMG_WIDTH

#used in the loop
ITERATION_NUM = 1

#used in raysampling
SAMPLING_INTERVALL = 0.1
SAMPLING_AMOUNT = 10
HALF_SAMPLE_DISTANCE = SAMPLING_INTERVALL * SAMPLING_AMOUNT /  2

#currently used cameras, data available
CAMERAS = [0,57,95,155]

#needed number to calculate group_of_cams
GROUP_SIZE = len(CAMERAS)

#during iteration the number of cameras checked for srdf
CAMERA_BATCH_SIZE = 1
DMAP_POINT_BATCH_SIZE = 1000

#current batch, camera chosen for this example
chosen_camera = 0

#getting depth values, intrinsic and extrinsic(c2w) of the chosen camera
camera_idx, intrinsic, extrinsic = cameraSet.__getitem__(chosen_camera)
dmap_of_chosen_camera = imageSet.dmaps[camera_idx]

#gets mask in the shape of 800x800x3
mask_of_chosen_camera = imageSet.masks[camera_idx]

#loading in the dmaps into ADAM
test_dmaps = testImageSet.dmaps
imageSet.salt_and_pepper()
imageSet.activate_gradients()
dmaps = imageSet.dmaps
adam = torch.optim.Adam([dmaps])

#main loop
def loop():
    #get a group of cams, in this case the only 4 cams i have chosen
    group_of_cams = cameraSet.get_n_closest_cameras(camera_idx, extrinsic, GROUP_SIZE)
    """for now disregard this because i have already my chosen group, change later GROUP_SIZE to a proper number"""
    tensor_group_of_cams = imageSet.get_group_of_cams_as_tensor(group_of_cams)

    #calculate origin and the pixel rays
    rays_o, rays_d = srdf_helpers.get_rays_tensor(IMAGE_HEIGHT, IMAGE_WIDTH, intrinsic[0][0], extrinsic)
    origin = rays_o[0][0]

    masked_output = srdf_helpers.apply_mask(dmap_of_chosen_camera, mask_of_chosen_camera)

    #some noise on the images
    global dmaps
    dmaps = srdf_helpers.gaussian_blur(dmaps)
    dmaps[0] = torch.roll(dmaps[0], 20, 1)

    for _ in range(ITERATION_NUM):
        #reset gradients
        adam.zero_grad()

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
            srdf_rays_o, srdf_rays_d = srdf_helpers.get_rays_tensor(IMAGE_HEIGHT, IMAGE_WIDTH, srdf_intrinsic[0][0], srdf_extrinsic)
            srdf_origin = srdf_rays_o[0][0]
            srdf_dmap = imageSet.dmaps[idx_camera]

            #project the sampled 3d points into the image plane of the cameras
            #shape:
            point_2d =srdf_helpers.calculate_2d_point_batch(sampling_tensor, srdf_extrinsic, srdf_intrinsic)
            point_2d = torch.reshape(point_2d, (1,point_2d.shape[0]*point_2d.shape[1],2))

            #get the necessary depth map value and the corresponding ray for the next calculation
            #shape:
            #shape:
            srdf_dmap = torch.unsqueeze(srdf_dmap, 0)
            srdf_dmap = torch.unsqueeze(srdf_dmap, 0)
            srdf_dmap_value = coordinate_lookup.lookup_value_at(point_2d, srdf_dmap)
            unsqueezed_srdf_rays_d = srdf_helpers.transpose_rays_tensor(srdf_rays_d)
            srdf_ray_vector = coordinate_lookup.lookup_value_at(point_2d, unsqueezed_srdf_rays_d)
            
            #now calculate the point that the selected group of cameras think is the surface
            #shape:
            srdf_predicted_point = srdf_helpers.calculate_point_with_depth_value(srdf_origin, srdf_ray_vector, srdf_dmap_value)
            srdf_predicted_point = torch.reshape(srdf_predicted_point, (DMAP_POINT_BATCH_SIZE,SAMPLING_AMOUNT,3))
            
            #calculate the srdf
            #shape:
            step = torch.norm(srdf_helpers.calculate_srdf(sampling_tensor, srdf_origin, srdf_predicted_point), dim=2)
            step_mask = torch.ge(step, 1000)
            srdf_tensor *= step
            
        mask_srdf_tensor = torch.ge(srdf_tensor, 1000)  
        srdf_tensor = srdf_tensor.masked_fill(mask_srdf_tensor, 100)  
        min = torch.min(srdf_tensor, 1).values
        loss = torch.mean(min[min!=100])
        loss.backward(retain_graph=True)

        adam.step()
        
    for idx, test_dmap in enumerate(test_dmaps):
        print(dmaps[idx][400])
        print(test_dmap)
        is_zero = test_dmap - dmaps[idx]
        tuple_nonzero = torch.nonzero(is_zero)
        if idx == 0:
            tensor_difference = srdf_helpers.append_tensor(is_zero)
        else:
            tensor_difference = srdf_helpers.append_tensor(tensor_difference, is_zero)

    srdf_helpers.save_into_file(tensor_difference, CAMERAS, name="_result", bool=False)



loop()