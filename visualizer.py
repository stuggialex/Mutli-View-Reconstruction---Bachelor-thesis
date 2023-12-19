import matplotlib.pyplot as plt
import torch

#sobald numpy funktioniert kann man das wieder anmachen

from Class import Camera
from Class import Image
import srdf_helpers

#initialize some values
cameraSet = Camera()
imageSet = Image()
test_camera_0 = 0
test_camera_0, intrinsic, extrinsic = cameraSet.__getitem__(test_camera_0)
IMAGE_HEIGHT = cameraSet.IMG_HEIGHT
IMAGE_WIDTH = cameraSet.IMG_WIDTH

rays_o, rays_d = srdf_helpers.get_rays_tensor(IMAGE_HEIGHT,IMAGE_WIDTH, intrinsic[0][0], extrinsic)

#example_data
example_tensor_0 = torch.tensor([0,1,2])
example_tensor_1 = torch.tensor([3,2,3])
example_tensor_2 = torch.tensor([0])
list_example_tensors = torch.stack((example_tensor_0,example_tensor_1))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

def visualize_point(point):
    ax.scatter(point[0].item(), point[1].item(), point[2].item(), color='green', s=30)

def visualize_ray(ray, ray_origin, length):
    if ray_origin == None:
        ray_origin = torch.tensor([0,0,0])
    ax.scatter(ray_origin[0].item(), ray_origin[1].item(), ray_origin[2].item(), color='red', s=50)
    ax.quiver(ray_origin[0].item(), ray_origin[1].item(), ray_origin[2].item(), ray[0].item(), ray[1].item(), ray[2].item(), length=length, arrow_length_ratio=0.1, normalize=True)

def visualize_point_2d(points):
    ax.scatter(points[0].item(), points[1].item(), zs=0, zdir='y')

def plot(points=None, points_2d=None, rays=None, origins=None, length=3):
    #if points == None and rays == None:
     #   ax.scatter(0, 0, 0, color='green', s=50)
    if points != None:
        for idx, point in enumerate(points):
            visualize_point(point)
    if points_2d != None:
        for idx, point in enumerate(points_2d):
            visualize_point_2d(point)
    if rays != None:
        for idx, ray in enumerate(rays):
            visualize_ray(ray, origins, length[idx])
    #visualize_point(points)
    #visualize_ray(rays, origins)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_zlim(-4, 4)
    # ax.set_xlim(-400, 400)
    # ax.set_ylim(-4, 4)
    # ax.set_zlim(-400, 400)
    ax.set_title('Test')
    plt.show()
