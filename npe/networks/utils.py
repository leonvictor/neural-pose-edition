import torch
import numpy as np
import yaml
import pathlib
import os


def load_config(path=None):
    """
    Load config file.
    """
    assert(os.path.isfile(path))

    with open(path) as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)

    return config


def normalize(dataset, mean, std, eps=1e-5, joint_index=None):
    """ (ds - mean) / std"""
    std += eps

    if isinstance(dataset, torch.Tensor):
        mean = torch.Tensor(mean).to(dataset.device)
        std = torch.Tensor(std).to(dataset.device)

    if joint_index is not None:
        if type(joint_index) == int:
            joint_index = [joint_index]

        joint_index = joints_to_vector_indices(joint_index)

        mean = mean[..., joint_index]
        std = std[..., joint_index]

    return ((dataset - mean) / std).reshape(dataset.shape)


def denormalize(dataset, mean, std, eps=1e-5, joint_index=None):
    """ (ds * std) + mean"""
    std -= eps

    if isinstance(dataset, torch.Tensor):
        mean = torch.Tensor(mean).to(dataset.device)
        std = torch.Tensor(std).to(dataset.device)

    if joint_index is not None:
        if type(joint_index) == int:
            joint_index = [joint_index]

        joint_index = joints_to_vector_indices(joint_index)

        mean = mean[..., joint_index]
        std = std[..., joint_index]

    return ((dataset * std) + mean).reshape(dataset.shape)


def joints_to_vector_indices(joints):
    """
    turn a list of joint indices in a list of each joint's vector indices
    """
    if isinstance(joints, int):
        joints = [joints]

    l = []
    for i in joints:
        l += [i*3, i*3+1, i*3+2]
    return l
