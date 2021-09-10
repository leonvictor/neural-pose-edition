from typing import List, Dict
from abc import ABC, abstractmethod

import numpy as np

from npe.anim.skeleton_joint import SkeletonJoint


def distance(pos_a, pos_b):
    return np.sqrt(np.sum(np.power(pos_a - pos_b, 2), axis=0))


def solve_joint(joint_a, fixed, initial_length):
    """Find the new position of joint_a based on the fixed position of another joint and the required distance between them"""
    dist = distance(joint_a, fixed)
    delta = initial_length/dist
    return (1 - delta) * fixed + (joint_a * delta)


class IFABRIKSolver(ABC):

    # joints associated with this solver. Using joints allow us to have access to the joints' names and indices
    joints: List[SkeletonJoint]
    tolerance_threshold: float

    _initialized: bool
    _distances: List[float]

    @abstractmethod
    def initialize(self, positions):
        pass

    @abstractmethod
    def forward(self, pose: List[SkeletonJoint], targets_dict: Dict[str, SkeletonJoint]):
        """Put the end effector on target and update the rest of the joints
        to respect joints distances"""
        pass

    @abstractmethod
    def backward(self, positions):
        pass
