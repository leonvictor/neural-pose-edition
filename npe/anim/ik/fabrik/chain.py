from typing import List, Dict
import numpy as np

from npe.math.vec3 import Vec3

from npe.anim.skeleton_joint import SkeletonJoint
from npe.anim.ik.target import IKTarget
from .base_solver import distance, solve_joint, IFABRIKSolver


class FABRIKChain(IFABRIKSolver):
    def __init__(self, joints: List[SkeletonJoint], tolerance_threshold=1e-1):
        assert joints is not None

        self.joints = joints
        self.tolerance_threshold = tolerance_threshold
        self._initialized = False

    def initialize(self, pose: Dict[str, SkeletonJoint]):
        """Initializes the chain (compute distances between joints, set root position)
        :param: joints: skeleton joints
        """
        assert pose is not None

        self._distances = []
        for i in range(len(self.joints)-1):
            p1 = self.joints[i].name
            p2 = self.joints[i+1].name
            self._distances.append(
                pose[p1].position.distance_to(pose[p2].position))

        self._initialized = True

    @property
    def indices(self):
        return [j.index for j in self.joints]

    @property
    def root(self):
        return self.joints[0]

    @property
    def end_effector(self):
        return self.joints[-1]

    def distance_to_target(self, pose: Dict[str, SkeletonJoint], target_dict: Dict[str, IKTarget]):
        if self.end_effector.name not in target_dict:
            return 0

        n = self.end_effector.name
        return pose[n].position.distance_to(target_dict[n].position)

    def forward(self, pose: Dict[str, SkeletonJoint], targets_dict: Dict[str, IKTarget] = None) -> Dict[str, SkeletonJoint]:
        """Perform a single forward pass: put the end effector on target and update the rest of the joints
        to respect joints distances"""

        # If no targets are given, assume the desired behavior is to fix limb lengths with regard to the end effector's position
        if targets_dict is None:
            targets_dict = {self.end_effector.name: IKTarget(self.end_effector)}

        # If targets are given but no key match the end effector, do nothing
        if self.end_effector.name not in targets_dict:
            return pose

        target = targets_dict[self.end_effector.name].position

        # Set the end effector pn as target t
        pose[self.end_effector.name].position = target
        for i in reversed(range(len(self.joints)-1)):
            # find the distance ri between the new joint position pi+1 and the joint pi
            # Find the new joint positions pi
            p1 = self.joints[i].name
            p2 = self.joints[i+1].name

            pose[p1].position = solve_joint(
                pose[p1].position,
                pose[p2].position,
                self._distances[i])

        return pose

    def backward(self, pose: Dict[str, SkeletonJoint], root_pos: Vec3 = None) -> Dict[str, SkeletonJoint]:
        """Perform a single backward pass: put the root in place and update the rest of the joints"""

        if root_pos is None:
            root_pos = pose[self.root.name].position

        # joints = self._prepare_pose(pose)
        # Set the root p1 its initial position
        pose[self.root.name].position = root_pos
        for i in range(len(self.joints)-1):
            # Get the joints this solver is associated to
            p1 = self.joints[i].name
            p2 = self.joints[i+1].name

            pose[p2].position = solve_joint(pose[p2].position, pose[p1].position, self._distances[i])

        # pose = self._undo_pose_preparation(pose, joints)
        return pose

    def fully_extend(self, pose: Dict[str, SkeletonJoint], targets_dict: Dict[str, IKTarget]) -> Dict[str, SkeletonJoint]:
        if self.end_effector.name not in targets_dict:
            return pose

        target = targets_dict[self.end_effector.name].position

        for i in range(len(self.joints)-1):
            p1 = self.joints[i].name
            p2 = self.joints[i+1].name

            pose[p2].position = solve_joint(target, pose[p1].position, self._distances[i])
        return pose

    def target_too_far(self, pose, targets_dict) -> bool:
        # FIXME: does not work if this chain's target is not in the dict
        # if self.end_effector.name not in targets_dict:
        #     return False

        target = targets_dict[self.end_effector.name].position
        dist = distance(pose[self.joints[0].name].position, target)
        return dist > np.sum(self._distances)

    def solve(self, pose: List[SkeletonJoint], target: Vec3 = None) -> List[SkeletonJoint]:
        """Solve the IK problem for the given chain
        """
        raise NotImplemented
        joints = self._prepare_pose(pose)

        # Distance between root and target
        root_pos = joints[0].position
        dist = distance(root_pos, self.target)

        # Check whether the target is within reach
        if dist > np.sum(self.distances):
            # The target is unreachable, extend the limb to the maximum
            joints = self.fully_extend(joints)
        else:
            # The target is reachable; thus, set as b the initial position of the joint p1
            # Check whether the distance between the end effector pn and the self.target is greater than a tolerance
            dif = distance(joints[-1].position, target)
            while dif >= self.tolerance_threshold:
                # Stage 1: forward reaching
                joints = self.forward(joints, target)
                # Stage 2: Backward reaching
                joints = self.backward(joints, root_pos)
                dif = distance(joints[-1].position, self.target)

        pose = self._undo_pose_preparation(pose, joints)
        return pose
