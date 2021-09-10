from typing import List, Dict
import copy

import numpy as np

from npe.anim.ik.target import IKTarget
from npe.anim.skeleton_joint import SkeletonJoint
from .base_solver import IFABRIKSolver
from .chain import FABRIKChain


class FABRIKMultipleEndEffectors(IFABRIKSolver):
    chains: List[FABRIKChain]

    def __init__(self, chains: List[FABRIKChain], tolerance_threshold=1e-1):

        self.indices = []
        r = chains[0].root
        for c in chains:
            # assert that all the chains share a common root
            assert(r == c.root)
            self.indices.extend(c.indices)

        self.chains = chains

        self.tolerance_threshold = tolerance_threshold
        self._initialized = False

    def initialize(self, pose: Dict[str, SkeletonJoint]):
        for c in self.chains:
            # Each chain is responsible for extracting the info it needs
            c.initialize(pose)

        self._initialized = True

    def forward(self, pose: Dict[str, SkeletonJoint], targets_dict: Dict[str, IKTarget], return_root_pos=False):
        sub_base_positions = []
        for c in self.chains:
            pose = c.forward(pose, targets_dict)
            sub_base_positions.append(pose[c.root.name].position)

        if return_root_pos:
            root_pos = np.mean(np.asarray(sub_base_positions), axis=0)
            return pose, root_pos

        return pose

    def backward(self, pose, root_pos=None) -> Dict[str, SkeletonJoint]:
        if root_pos is None:
            root_pos = pose[self.root.name].position

        for c in self.chains:
            pose = c.backward(pose, root_pos)
        return pose

    def distance_to_targets(self, pose, targets_dict):
        return np.mean([c.distance_to_target(pose, targets_dict) for c in self.chains])

    @property
    def root(self):
        root = self.chains[0].root  # copy attributes
        # FIXME: ensure root is actually copied
        sub_base_positions = []
        for c in self.chains:
            sub_base_positions.append(c.root.position)
        root.position = np.mean(np.asarray(sub_base_positions), axis=0)
        return root

    @property
    def end_effectors(self):
        return [c.end_effector for c in self.chains]

    @property
    def root(self):
        return self.chains[0].root  # All roots should be the same

    def solve(self, pose, targets_dict):
        assert self._initialized

        # all_too_far = True
        # for c in self.chains:
        #     if c.target_too_far(pose, targets_dict):
        #         c.fully_extend(pose, targets_dict)
        #     else:
        #         all_too_far = False

        # if all_too_far:
        #     return pose

        dif = self.distance_to_targets(pose, targets_dict)
        while(dif >= self.tolerance_threshold):
            starting_pose = copy.deepcopy(pose)

            pose, root_pos = self.forward(pose, targets_dict, return_root_pos=True)
            pose = self.backward(pose, root_pos)

            dif = self.distance_to_targets(pose, targets_dict)

            # Break if nothing changed
            allclose = True
            for name, joint in pose.items():
                tol = self.tolerance_threshold
                allclose = allclose and np.allclose(joint.position, starting_pose[name].position, tol, tol)

            if allclose:
                break

        return pose
