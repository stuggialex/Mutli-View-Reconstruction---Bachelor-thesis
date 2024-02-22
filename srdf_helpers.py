import torch
import torchvision    
import numpy
import shutil
import os
import cv2

def gaussian_blur(dmaps):
    gauss = torchvision.transforms.GaussianBlur(21, sigma=0.5)
    for _ in range(4):
        dmaps = gauss(dmaps)
    return dmaps

def calc_viewing_direction(extrinsic):
        viewing_matrix = extrinsic[:3, :3]
        null_vec = torch.tensor([0, 0, 1], dtype=torch.float32)
        return torch.matmul(viewing_matrix, null_vec * -1)

def calc_vector_length(vector):
     vector_square_sum = torch.sum(torch.square(vector))
     return torch.sqrt(vector_square_sum)

def calc_normalized_vector(vector):
        vector_length = calc_vector_length(vector)
        return torch.tensor([vector[0].item()/vector_length,vector[1].item()/vector_length,vector[2].item()/vector_length])

# def get_rays_tensor(H, W, focal, c2w):
#     c2w = torch.tensor(c2w).numpy()
#     i, j = numpy.meshgrid(numpy.arange(W, dtype=numpy.float32),
#                        numpy.arange(H, dtype=numpy.float32), indexing='xy')
#     dirs = numpy.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -numpy.ones_like(i)], -1)
#     rays_d = numpy.sum(dirs[..., numpy.newaxis, :] * c2w[:3, :3], -1)
#     rays_o = numpy.broadcast_to(c2w[:3, -1], numpy.shape(rays_d))
#     return torch.from_numpy(rays_o), torch.from_numpy(rays_d)

def get_rays_tensor_torch(H, W, focal, c2w):
    """Get rays for all pixels.
        Args:
            H (int): height of image
            W (int): width of image
            focal (float): focal length
            c2w [B, 4, 4]: camera-to-world matrix

        Returns:
            rays_o [B, H, W, 3]: ray origins
            rays_d [B, H, W, 3]: ray directions
    """
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32),
                       torch.arange(H, dtype=torch.float32), indexing='xy')
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame.
    # dirs [H, W, 3], [C, 4, 4] -> [C, H, W, 3]
    rays_d = torch.sum(dirs[None, :, :, None, :] * c2w[:, None, None, :3, :3], -1)
    #rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_o = torch.broadcast_to(c2w[:,None, None, :3, -1], rays_d.shape)
    return rays_o, rays_d

def transpose_rays_tensor(rays):
    #rays = torch.unsqueeze(rays, 0)
    rays = torch.transpose(rays, 2, 3).contiguous()
    rays = torch.transpose(rays, 1, 2).contiguous()
    return rays

#needs normalized vector
def raysampling(starting_point, normed_ray, samp, samp_times):
    sampling_tensor = torch.ones(samp_times, starting_point.shape[0], starting_point.shape[1])*starting_point
    #randn = torch.abs(torch.randn(samp_times, starting_point.shape[0], 1))
    #sampling_tensor += normed_ray * randn
    return sampling_tensor

def calculate_2d_point_batch(point_3d, extrinsic, intrinsic, sampling_amount):
    tensor_one = torch.ones(point_3d.shape)
    tensor_one = torch.narrow(tensor_one,-1,0,1) #utility funktion rausschreiben
    tensor_one = torch.transpose(tensor_one, 0,-1)
    point_3d = torch.transpose(point_3d, 0,-1)
    w2c = torch.inverse(extrinsic)
    #turn point_3d homogeneous
    tensor_4d = torch.cat((point_3d, tensor_one)).float()
    tensor_4d = torch.transpose(tensor_4d, 0,-1)
    tensor_4d = torch.transpose(tensor_4d,-1,-2)
    tensor_4d_expanded = tensor_4d.expand(w2c.shape[0], tensor_4d.shape[0], 4, sampling_amount)
    w2c_expanded = w2c.expand(tensor_4d.shape[0], w2c.shape[0], 4, 4)
    w2c_expanded = torch.transpose(w2c_expanded, 0, 1)
    point_camera = torch.matmul(w2c_expanded, tensor_4d_expanded)
    point_camera = torch.narrow(point_camera,-2,0,3)
    point_image_homogeneous = torch.matmul(intrinsic, point_camera)
    point_image_homogeneous_1 = torch.narrow(point_image_homogeneous, -2, 0, 2)
    point_image_homogeneous_2 = torch.narrow(point_image_homogeneous, -2, 2, 1)
    point_image = point_image_homogeneous_1 / point_image_homogeneous_2
    #point_image = torch.round(point_image)
    #point_image = point_image.int()
    point_image = torch.transpose(point_image, -1, -2)
    return point_image+400

def calculate_2d_point(point_3d, extrinsic, intrinsic):
    tensor_one = torch.tensor([1])
    w2c = torch.linalg.inv(extrinsic)
    #turn point_3d homogeneous
    tensor_4d = torch.cat((point_3d, tensor_one)).float()
    point_camera = torch.matmul(w2c, tensor_4d)
    point_image_homogeneous = torch.matmul(intrinsic, point_camera[:3])
    point_image = point_image_homogeneous[:2] / point_image_homogeneous[2]
    point_image = torch.round(point_image)
    point_image = point_image.int()
    return point_image

#for now i retire this function
def sample_next_point(point, ray, samp_intervall):
     ray = calc_normalized_vector(ray)
     return point + ray * samp_intervall

def apply_mask(image, mask):
    if (image.shape == torch.Size([800, 800])):
        bool_mask = torch.ge(mask, 200)[:,:,0]
    else:
         bool_mask = torch.ge(mask, 200)
    masked_output = image * bool_mask.int().float()
    return masked_output

def apply_mask_tensor(image, mask):
    bool_mask = torch.ge(mask, 200)[:,:,:,0]
    masked_output = image * bool_mask.int().float()
    return masked_output

def apply_srdf_mask(srdf):
    bool_mask = torch.le(srdf, 1)
    masked_output = srdf * bool_mask.int().float()
    return masked_output


#divider equals to the row number
def get_random_dmap_point_batch(depthmap,sampling_amount,divider):
    #flatten the tensor and get a random item out of it
    t = torch.flatten(depthmap)
    idx = torch.multinomial(t,sampling_amount)
    dividend = torch.div(idx, divider, rounding_mode="floor")
    remainder = torch.remainder(idx, divider)
    point_batch = torch.stack((dividend, remainder), 1)
    return point_batch

def calculate_point_with_depth_value(origin, ray_vector, depth):
    return origin + ray_vector * depth

def calculate_vector_length_between_two_points(point_a, point_b):
    vector = point_a - point_b
    return calc_vector_length(vector).item()

def append_tensor(tensor, point=None):
    if point != None:
        tensor = torch.cat((tensor, torch.unsqueeze(point, 0)))
    else:
        tensor = torch.unsqueeze(tensor, 0)
    return tensor

def tensor_index_lookup(tensor, index_tensor, are_rays=False):
    transposed_index_tensor = torch.transpose(index_tensor, -2, -1)
    narrowed_index_tensor = torch.narrow(transposed_index_tensor, -2, 0, 1)
    chosen_rows = torch.index_select(tensor, 0, narrowed_index_tensor) 
    if (not are_rays):
        values = torch.gather(chosen_rows, 1, transposed_index_tensor[1].unsqueeze(1))
    else:
        index_tensor_plus_ones = transposed_index_tensor[1].unsqueeze(1).unsqueeze(1) * torch.ones(3, dtype=int)
        values = torch.gather(chosen_rows, 1, index_tensor_plus_ones)
    return values
          

def calculate_srdf(sampled_point, camera_origin, predicted_dmap_point):
    z = torch.sub(sampled_point, camera_origin)
    y = torch.sub(predicted_dmap_point, camera_origin)
    srdf = torch.sub(y, z)
    #srdf = calc_vector_length(srdf)
    return srdf

def calculate_srdf_consistency(tensor, sigma, gamma):
    tensor = torch.exp(torch.square(tensor)/sigma)+gamma
    srdf_consistency = torch.prod(tensor, 0)
    return srdf_consistency

def load_and_show_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
    '''
    you might have to disable following flags, if you are reading a semantic map/label then because it will convert it into binary map so check both lines and see what you need
    ''' 
    # img = cv2.imread(PATH2EXR) 
 
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.imshow('exr', img)
    cv2.waitKey(0)

def dict_to_string(dict):
    string = "__"
    for key, value in dict.items():
          string += key + "_" + str(value) + "__"
    return string
          

def save_into_file(images, idx, name="", variable_list="", bool=True):
    directory = "Images/results"
    for root, dirs, files in os.walk(directory):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
        os.chdir(directory)
        if images.shape == torch.Size([800, 800, 3]):
            images = images.detach().numpy().astype(numpy.float32)
            cv2.imwrite(str(idx) + name + variable_list + ".exr", images)
        else:
            for i, image in enumerate(images):
                image = image.detach().numpy().astype(numpy.float32)
                cv2.imwrite(str(idx[i]) + name + variable_list + ".exr", image)