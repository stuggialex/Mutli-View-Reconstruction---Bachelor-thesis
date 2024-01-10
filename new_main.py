from Class import Camera, Image
import srdf_helpers
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
SAMPLING_INTERVALL = 0.05
SAMPLING_AMOUNT = 20
HALF_SAMPLE_DISTANCE = SAMPLING_INTERVALL * SAMPLING_AMOUNT /  2

#currently used cameras, data available
CAMERAS = [0,57,95,155]

#needed number to calculate group_of_cams
GROUP_SIZE = len(CAMERAS)

#during iteration the number of cameras checked for srdf
CAMERA_BATCH_SIZE = 1
DMAP_POINT_BATCH_SIZE = 6

#current batch, camera chosen for this example
chosen_camera = 0

#getting depth values, intrinsic and extrinsic(c2w) of the chosen camera
camera_idx, intrinsic, extrinsic = cameraSet.__getitem__(chosen_camera)
dmap_of_chosen_camera = imageSet.dmaps[camera_idx]

#gets mask in the shape of 800x800x3
mask_of_chosen_camera = imageSet.masks[camera_idx]

#loading in the dmaps into ADAM
test_dmaps = testImageSet.dmaps
#imageSet.salt_and_pepper()
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
        batch_sampled_dmap_points = srdf_helpers.get_random_dmap_point_batch(masked_output, DMAP_POINT_BATCH_SIZE, IMAGE_HEIGHT)

        #corresponding depth map values to batch_sampled_dmap_points
        batch_dmap_values = srdf_helpers.tensor_index_lookup(dmap_of_chosen_camera, batch_sampled_dmap_points)

        #corresponding ray vectors to batch_sampled_dmap_points
        batch_ray_vector = srdf_helpers.tensor_index_lookup(rays_d, batch_sampled_dmap_points, are_rays=True)


        #resulting 3d point from the randomly selected batch_sampled_dmap_points
        predicted_point = srdf_helpers.calculate_point_with_depth_value(origin, batch_ray_vector, batch_dmap_values)
        
        #tensor of points sampled around the predicted_point
        sampling_tensor = srdf_helpers.raysampling(predicted_point - HALF_SAMPLE_DISTANCE * batch_ray_vector, batch_ray_vector, SAMPLING_INTERVALL, SAMPLING_AMOUNT)
        

        #variable instatiation

        #get 3d point that corresponds to the depth map value and sample around it

        #go through the group of cameras and check their side
        for idx_camera, camera in enumerate(CAMERAS):
            #variable instatiation
            srdf_camera_idx, srdf_intrinsic, srdf_extrinsic = cameraSet.__getitem__(camera)
            srdf_rays_o, srdf_rays_d = srdf_helpers.get_rays_tensor(IMAGE_HEIGHT, IMAGE_WIDTH, srdf_intrinsic[0][0], srdf_extrinsic)
            srdf_origin = srdf_rays_o[0][0]
            srdf_dmap = imageSet.dmaps[idx_camera]
            srdf_tensor = torch.ones(SAMPLING_AMOUNT)

            #transposed_sampling_tensor = torch.transpose(sampling_tensor,0,1)

            point_2d =srdf_helpers.calculate_2d_point(sampling_tensor, srdf_extrinsic, srdf_intrinsic)
            print(point_2d)
            return

            #check every point that has been sampled
            for point_idx, point in enumerate(sampling_tensor):
                point_2d =srdf_helpers.calculate_2d_point(point, srdf_extrinsic, srdf_intrinsic)
                
                #adjust x and y coordinates as the center currently is (0,0)
                srdf_row_count = point_2d[0].item()+400
                srdf_column_count = point_2d[1].item()+400

                #get the necessary depth map value and the corresponding ray for the next calculation
                srdf_dmap_value = srdf_dmap[srdf_row_count][srdf_column_count] 
                srdf_ray_vector = rays_d[srdf_row_count][srdf_column_count] 
                #rays_arr.append(srdf_ray_vector)

                #now calculate the point that the selected group of cameras think is the surface, aswell as srdf 
                srdf_predicted_point = srdf_helpers.calculate_point_with_depth_value(srdf_origin, srdf_ray_vector, srdf_dmap_value)
                #point_arr.append(srdf_predicted_point)
                #ray_length_arr.append(srdf_helpers.calculate_vector_length_between_two_points(srdf_predicted_point, srdf_origin))
                srdf_tensor[point_idx] *= srdf_helpers.calculate_srdf(point, srdf_origin, srdf_predicted_point)
            
            torch.min(srdf_tensor).backward()

        adam.step()
    for idx, test_dmap in enumerate(test_dmaps):
        is_zero = test_dmap - dmaps[idx]
        tuple_nonzero = torch.nonzero(is_zero)
        if idx == 0:
            tensor_difference = srdf_helpers.append_tensor(is_zero)
            print(tensor_difference.shape)
        else:
            tensor_difference = srdf_helpers.append_tensor(tensor_difference, is_zero)

    srdf_helpers.save_into_file(tensor_difference, CAMERAS, name="_result", bool=False)



loop()