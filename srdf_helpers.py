import torch    
import numpy
import shutil
import os
import cv2

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

def get_rays_tensor(H, W, focal, c2w):
    c2w = torch.tensor(c2w).numpy()
    i, j = numpy.meshgrid(numpy.arange(W, dtype=numpy.float32),
                       numpy.arange(H, dtype=numpy.float32), indexing='xy')
    dirs = numpy.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -numpy.ones_like(i)], -1)
    rays_d = numpy.sum(dirs[..., numpy.newaxis, :] * c2w[:3, :3], -1)
    rays_o = numpy.broadcast_to(c2w[:3, -1], numpy.shape(rays_d))
    return torch.from_numpy(rays_o), torch.from_numpy(rays_d)

#needs normalized vector
def raysampling(starting_point, normed_ray, samp_intervall, samp_times):
    sampling_tensor = torch.unsqueeze(starting_point, 0)
    for x in range(samp_times - 1):
            sampling_tensor = torch.cat((sampling_tensor, torch.unsqueeze(sampling_tensor[-1] + normed_ray * samp_intervall, 0)))
    return sampling_tensor

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

#divider equals to the row number
def get_random_depth_map_values(depthmap,sampling_amount,divider):
    #flatten the tensor and get a random item out of it
    t = torch.flatten(depthmap)
    idx = torch.multinomial(t,sampling_amount)
    idx_list = []
    for item in idx:
        column_row_tuple = divmod(item.item(), divider)
        #get depth map value out of the received index from multinomial
        depth_value = depthmap[column_row_tuple[0]][column_row_tuple[1]]
        idx_list.append((column_row_tuple, depth_value))
    return idx_list

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
          

def calculate_srdf(sampled_point, camera_origin, predicted_dmap_point):
    z = torch.sub(sampled_point, camera_origin)
    y = torch.sub(predicted_dmap_point, camera_origin)
    srdf = torch.sub(y, z)
    srdf = calc_vector_length(srdf)
    return srdf

def load_and_show_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
    '''
    you might have to disable following flags, if you are reading a semantic map/label then because it will convert it into binary map so check both lines and see what you need
    ''' 
    # img = cv2.imread(PATH2EXR) 
 
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.imshow('exr', img)
    cv2.waitKey(0)

def save_into_file(images, idx, name="", bool=True):
    directory = "Images/results"
    for root, dirs, files in os.walk(directory):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
        os.chdir(directory)
        if images.shape == torch.Size([800, 800, 3]):
            images = images.detach().numpy().astype(numpy.float32)
            cv2.imwrite(str(idx) + name + ".exr", images)
        else:
            for i, image in enumerate(images):
                image = image.detach().numpy().astype(numpy.float32)
                cv2.imwrite(str(idx[i]) + name + ".exr", image)