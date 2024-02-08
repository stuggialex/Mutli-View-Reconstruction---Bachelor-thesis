from Class import Camera, Image
import srdf_helpers
import coordinate_lookup
#import visualizer
import torchvision

import torch

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

cuda = torch.device("cuda:0")

#initialize data
cameraSet = Camera()
cameraSet.to_cuda(cuda)
imageSet = Image()
imageSet.to_cuda(cuda)
testImageSet = Image()
testImageSet.to_cuda(cuda)

#current size of images
IMAGE_HEIGHT = cameraSet.IMG_HEIGHT
IMAGE_WIDTH = cameraSet.IMG_WIDTH

#used in the loop
ITERATION_NUM = 100

#used in raysampling
SAMPLING_INTERVALL = 0.05
SAMPLING_AMOUNT = 40
HALF_SAMPLE_DISTANCE = SAMPLING_INTERVALL * SAMPLING_AMOUNT /  2

#currently used cameras, data available
CAMERAS = [0,57,95,155]
CAMERAS_WITHOUT_0 = [57,95,155]

#needed number to calculate group_of_cams
GROUP_SIZE = len(CAMERAS_WITHOUT_0)

#during iteration the number of cameras checked for srdf
CAMERA_BATCH_SIZE = 1
DMAP_POINT_BATCH_SIZE = 10000

#SRDF parameter
SIGMA = 10
GAMMA = 0.01

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
extrinsic = torch.unsqueeze(extrinsic, 0)

#gets mask in the shape of 800x800x3
mask_of_chosen_camera = imageSet.masks[camera_idx]

#loading in the dmaps into ADAM
#imageSet.gaussian_noise(0)
imageSet.dmaps[0] += 5
imageSet.activate_gradients()
dmaps = imageSet.dmaps
ground_truth_dmaps =testImageSet.dmaps
clone_dmaps = torch.clone(dmaps)
adam = torch.optim.Adam([dmaps], lr=0.03)

#main loop
def loop():
    #get a group of cams, in this case the only 4 cams i have chosen
    temp_extrinsic = torch.squeeze(extrinsic, 0)
    group_of_cams = cameraSet.get_n_closest_cameras(camera_idx, temp_extrinsic, GROUP_SIZE)
    idx_of_group_of_cams = torch.Tensor([i[0] for i in group_of_cams]).int().to(cuda)
    dmaps_group_of_cams, masks_group_of_cams = imageSet.get_group_of_cams_as_tensor(group_of_cams)
    
    for x in range(ITERATION_NUM):
        print(x)
        #reset gradients
        adam.zero_grad()

        #calculate origin and the pixel rays
        rays_o, rays_d = srdf_helpers.get_rays_tensor_torch(IMAGE_HEIGHT, IMAGE_WIDTH, intrinsic[0][0], extrinsic)
        origin = rays_o[0][0][0]

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

        srdf_tensor = torch.ones(GROUP_SIZE, DMAP_POINT_BATCH_SIZE, SAMPLING_AMOUNT).to_cuda(cuda)

        #variable instatiation
        srdf_intrinsic, srdf_extrinsic = cameraSet.get_item_tensor(idx_of_group_of_cams)
        srdf_torch_rays_o, srdf_torch_rays_d = srdf_helpers.get_rays_tensor_torch(IMAGE_HEIGHT, IMAGE_WIDTH, srdf_intrinsic[0][0][0], srdf_extrinsic)
        

        srdf_origin = srdf_torch_rays_o[0][0][0]
        
        srdf_dmap = dmaps_group_of_cams
        srdf_mask = masks_group_of_cams

        srdf_masked_output = srdf_helpers.apply_mask_tensor(srdf_dmap, srdf_mask)

        #project the sampled 3d points into the image plane of the cameras
        #shape:
        point_2d = srdf_helpers.calculate_2d_point_batch(sampling_tensor, srdf_extrinsic, srdf_intrinsic[0], SAMPLING_AMOUNT)
        point_2d = torch.reshape(point_2d, (point_2d.shape[0],point_2d.shape[1]*point_2d.shape[2],2))

        #get the necessary depth map value and the corresponding ray for the next calculation
        #shape:
        #shape:
        srdf_masked_output = torch.unsqueeze(srdf_masked_output, 1)
        srdf_dmap_value = coordinate_lookup.lookup_value_at(point_2d, srdf_masked_output)
        unsqueezed_srdf_rays_d = srdf_helpers.transpose_rays_tensor(srdf_torch_rays_d)
        srdf_ray_vector = coordinate_lookup.lookup_value_at(point_2d, unsqueezed_srdf_rays_d)
        
        #now calculate the point that the selected group of cameras think is the surface
        #shape:
        srdf_predicted_point = srdf_helpers.calculate_point_with_depth_value(srdf_origin, srdf_ray_vector, srdf_dmap_value)
        srdf_predicted_point = torch.reshape(srdf_predicted_point, (GROUP_SIZE, DMAP_POINT_BATCH_SIZE,SAMPLING_AMOUNT,3))
        #calculate the srdf
        #shape:
        step = srdf_helpers.calculate_srdf(sampling_tensor, srdf_origin, srdf_predicted_point)
        step = torch.norm(step, dim=3)

        srdf_consistency = srdf_helpers.calculate_srdf_consistency(step, SIGMA, GAMMA)

        energy_function = torch.sum(srdf_consistency, 1)
        
        # srdf_tensor *= step    
        mask_energy_function = torch.ge(energy_function, 100)  
        energy_function = energy_function.masked_fill(mask_energy_function, 100)  
        #min = torch.min(energy_function, 1).values
        loss = torch.mean(energy_function[energy_function!=100])
        #print(loss)
        

        loss.backward()
        # for group in adam.param_groups:
        #     for p in group['params']:
        #         if p.grad is not None:
        #             break
        #             print(p.grad.flatten().shape)
                    #writer.add_histogram("gradients", p.grad.flatten(), x)
       
        loss_cpu = loss.cpu()
        writer.add_scalar("Loss/train", loss_cpu, x)

        #mse calculation
        for idx, gtd in enumerate(ground_truth_dmaps):
            tensor_difference = gtd - dmaps[idx]
            mse = torch.mean(torch.square(tensor_difference))
            mse_cpu = mse.cpu()
            #print(mse)
            writer.add_scalar("MSE" + str(idx), mse_cpu, x)
            if x % 100 == 0:
                unsqueezed_dmaps = torch.unsqueeze(dmaps[idx], 0)
                unsqueezed_dmaps = unsqueezed_dmaps.cpu()
                writer.add_image("result" + str(idx), unsqueezed_dmaps, global_step=x)

        tensor_difference = gtd - dmaps[idx]
        mse = torch.mean(torch.square(tensor_difference))
        mse_cpu = mse.cpu()
        #print(mse)
        writer.add_scalar("MSE_gesamt", mse_cpu, x)

        #adam.g
        #writer.add_histogram(tag + "grads", x)

        adam.step()

    writer.flush()
        
    # for idx, test_dmap in enumerate(ground_truth_dmaps):
    #     is_zero = test_dmap - dmaps[idx]
    #     tuple_nonzero = torch.nonzero(is_zero)
    #     if idx == 0:
    #         tensor_difference = srdf_helpers.append_tensor(is_zero)
    #     else:
    #         tensor_difference = srdf_helpers.append_tensor(tensor_difference, is_zero)

    srdf_helpers.save_into_file(dmaps, CAMERAS, name="_result", variable_list=dict_as_str, bool=False)



loop()
