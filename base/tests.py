import scipy.ndimage.filters
import torch, kornia
import numpy as np
import torch.nn.functional as F
from scipy.ndimage.interpolation import map_coordinates

def match_gaussian():
    tensor = torch.rand(5,5)

    sigma = 2
    t1 = scipy.ndimage.filters.gaussian_filter(tensor.numpy(), sigma, mode='constant')

    kernel_size = int(2*(4.0*sigma+0.5))
    k = kornia.filters.get_gaussian_kernel2d((kernel_size,kernel_size), (sigma, sigma))
    t2 = kornia.filters.filter2D(tensor[None, None], kernel=k[None], border_type='constant')

    print('\nScipy')
    print(t1)
    print('\nKornia')
    print(t2.numpy().squeeze())

def match_coordinates():
    # map coordinates
    #################
    tensor = torch.rand(10,10)

    shape = h, w = tensor.shape
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    dx = np.ones(shape)*0.5
    dy = -np.ones(shape)*0.5
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

    t1 = map_coordinates(tensor.numpy(), indices, order=1).reshape(shape)

    grid = kornia.utils.create_meshgrid(h, w).to(device=tensor.device)

    # d_yx = torch.stack([torch.as_tensor(dy), torch.as_tensor(dx)], dim=2).float()
    # t2 =  F.grid_sample(tensor[None, None], (grid + d_yx)[None], align_corners=True)

    t1 = np.stack([y,x], axis=2)
    t2 = grid
    print('\nScipy')
    print(t1)
    print('\nKornia')
    print(t2.numpy().squeeze())
    
match_coordinates()