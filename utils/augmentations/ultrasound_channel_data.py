import torch
import torch.nn.functional as F
import numpy as np


def add_speckles_channel_data(channel_data, noise_level=1 / 20, sampling_factor=4):
    b, c, h, w = channel_data.shape
    gaussian_map = torch.randn(*channel_data.shape) * noise_level
    gaussian_map = gaussian_map.view(b * c, h, w)
    gaussian_map = gaussian_map.unsqueeze(dim=1)
    kernel = torch.Tensor(
        [np.ones(3) * 0.9, np.ones(3) * 0.8, np.ones(3) * 0.6, np.ones(3) * 0.4, np.ones(3) * 0.2]) / 2.9
    kernel[2, 1] = 0
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    weighted_map = F.conv2d(gaussian_map, kernel, padding=(2, 1))
    weighted_map = weighted_map.view(b, c, h, w)
    sampled_weighted_map = torch.zeros_like(weighted_map)
    sampled_weighted_map[:, torch.arange(0, c, sampling_factor), :, :] = weighted_map[:,
                                                                         torch.arange(0, c, sampling_factor), :, :]
    noise_map = sampled_weighted_map + channel_data
    return noise_map


def horizontal_flip_channel_data(channel_data):
    return torch.flip(channel_data, dims=[-1])


def subsample_channel_data(channel_data, height_factor=1, width_factor=1):
    if height_factor > 1:
        channel_data = channel_data[:, :, torch.arange(0, channel_data.shape[2], height_factor), :]
    if width_factor > 1:
        channel_data = channel_data[:, :, :, torch.arange(0, channel_data.shape[3], width_factor)]
    return channel_data
