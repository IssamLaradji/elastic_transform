"""
Most functions are obtained from https://kornia.readthedocs.io/en/latest/_modules/kornia/
The main difference is that this file contains the 
  - ElasticTransform class; and
  - a differentiable Gaussian filter.
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
import math
import numpy as np
from torch.distributions import Uniform
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import scipy.ndimage.filters


class ElasticTransform(nn.Module):
    def __init__(self, alpha=1, sigma=12, random_seed=42):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.ones(
            2), requires_grad=True) * math.log(sigma)
        self.log_alpha = nn.Parameter(torch.ones(
            2), requires_grad=True) * math.log(alpha)
        self.log_disp = nn.Parameter(torch.zeros(2), requires_grad=True)
        self.random_seed = random_seed

    def forward(self, x):
        sigma = torch.exp(self.log_sigma)
        alpha = torch.exp(self.log_alpha)
        disp = torch.exp(self.log_disp).clamp(0, 1)
        img_transformed = elastic_transform_2d(x,
                                               kernel_size=(3, 3),
                                               sigma=sigma,
                                               alpha=alpha,
                                               disp_scale=disp,
                                               random_seed=self.random_seed)
        return img_transformed


def elastic_transform_2d(tensor: torch.Tensor,
                         kernel_size: Tuple[int, int] = (3, 3),
                         sigma: Tuple[float, float] = (4., 4.),
                         alpha: Tuple[float, float] = (32., 32.),
                         disp_scale: Tuple[float, float] = (0.1, 0.1),
                         random_seed: Optional = None) -> torch.Tensor:
    r"""Applies elastic transform of images as described in [Simard2003]_.
    Args:
        img (torch.Tensor): input image.
        kernel_size (Tuple[int, int]): the size of the Gaussian kernel. Default:(3,3).
        sigma (Tuple[float, float]): the standard deviation of the Gaussian in the y and x directions, respecitvely. 
                                     Larger sigma results in smaller pixel displacements. Default:(4,4).
        alpha (Tuple[float, float]):  the scaling factor that controls the intensity of the deformation
                                  in the y and x directions, respectively. Default:(32,32).
        disp_scale (Tuple[float, float]):  the scaling factor that controls the intensity of the displacement
                                  in the y and x directions, respectively. Default:(0.1, 0.1).
        random_seed (Optional): random seed for generating the displacement vector. Default:None.

    Returns:
        img (torch.Tensor): the elastically transformed input image.
    References:
        [Simard2003]: Simard, Steinkraus and Platt, "Best Practices for
                      Convolutional Neural Networks applied to Visual Document Analysis", in
                      Proc. of the International Conference on Document Analysis and
                      Recognition, 2003.
    """
    generator = torch.Generator(device='cpu')
    if random_seed is not None:
        generator.manual_seed(random_seed)

    n, c, h, w = tensor.shape

    # Convolve over a random displacement matrix and scale them with 'alpha'
    d_rand = torch.rand(n, 2, h, w, generator=generator).to(
        device=tensor.device) * 2 - 1

    tensor_y = d_rand[:, 0].squeeze()
    tensor_x = d_rand[:, 1].squeeze()

    dy = apply_gaussian(tensor_y, sigma[0]) * alpha[0]
    dx = apply_gaussian(tensor_x, sigma[1]) * alpha[1]

    # stack and normalize displacement
    d_yx = torch.cat([dy, dx], dim=1).permute(0, 2, 3, 1)

    # Warp image based on displacement matrix
    grid = kornia.utils.create_meshgrid(h, w).to(device=tensor.device)
    warped = F.grid_sample(
        tensor, (grid + d_yx).clamp(-1, 1), align_corners=True)

    return warped

# helpers
# ============


def apply_gaussian_numpy(tensor, sigma):
    t1 = scipy.ndimage.filters.gaussian_filter(
        tensor.numpy(), float(sigma), mode='constant')

    return t1


def apply_gaussian(tensor, sigma):
    kernel_size = int(2*(4.0*sigma+0.5))
    k = get_gaussian_kernel2d((kernel_size, kernel_size), (sigma, sigma))
    t2 = kornia.filters.filter2D(
        tensor[None, None], kernel=k[None], border_type='constant')

    return t2


def gaussian(window_size, sigma):
    r"""
    Modified from Kornia to allow gradients to flow
    """
    x = torch.arange(window_size).float().to(
        device=sigma.device) - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp((-x.pow(2.0) / (2 * sigma ** 2)))
    return gauss / gauss.sum()


def get_gaussian_kernel2d(
        kernel_size: Tuple[int, int],
        sigma: Tuple[float, float],
        force_even: bool = False) -> torch.Tensor:
    r"""
    See Kornia for Description
    """
    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
        raise TypeError(
            "kernel_size must be a tuple of length two. Got {}".format(
                kernel_size
            )
        )
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError(
            "sigma must be a tuple of length two. Got {}".format(sigma)
        )
    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma
    kernel_x: torch.Tensor = get_gaussian_kernel1d(
        ksize_x, sigma_x, force_even)
    kernel_y: torch.Tensor = get_gaussian_kernel1d(
        ksize_y, sigma_y, force_even)
    kernel_2d: torch.Tensor = torch.matmul(
        kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t()
    )
    return kernel_2d


def get_gaussian_kernel1d(kernel_size: int,
                          sigma: float,
                          force_even: bool = False) -> torch.Tensor:
    r"""
    See Kornia for Description
    """
    if (not isinstance(kernel_size, int) or (
            (kernel_size % 2 == 0) and not force_even) or (
            kernel_size <= 0)):
        raise TypeError(
            "kernel_size must be an odd positive integer. "
            "Got {}".format(kernel_size)
        )
    window_1d: torch.Tensor = gaussian(kernel_size, sigma)
    return window_1d
