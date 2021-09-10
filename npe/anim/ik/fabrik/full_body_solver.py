from __future__ import annotations
from typing import Dict
from copy import copy

from npe.anim.ik.target import IKTarget
from npe.anim.skeleton_joint import SkeletonJoint

from .chain import FABRIKChain
from .multiple_end_effectors import FABRIKMultipleEndEffectors


class HumanSolvers:
    torso: FABRIKMultipleEndEffectors
    hips:  FABRIKMultipleEndEffectors
    spine: FABRIKChain
    head:  FABRIKChain
    left_foot: FABRIKChain
    right_foot: FABRIKChain


class FABRIKFullBodySolver():

    _solvers: HumanSolvers

    def __init__(self, skeleton: Skeleton, tolerance_threshold: float):

        self._skeleton = skeleton
        chains = skeleton.get_ik_chains()

        self._solvers = HumanSolvers()
        self._solvers.torso = FABRIKMultipleEndEffectors(
            [
                FABRIKChain(chains['right_arm'], tolerance_threshold),
                FABRIKChain(chains['left_arm'], tolerance_threshold)
            ], tolerance_threshold)

        self._solvers.hips = FABRIKMultipleEndEffectors(
            [
                FABRIKChain(chains['right_leg'], tolerance_threshold),
                FABRIKChain(chains['left_leg'], tolerance_threshold)
            ],
            tolerance_threshold)

        self._solvers.spine = FABRIKChain(chains['spine'], tolerance_threshold)
        self._solvers.head = FABRIKChain(chains['head'], tolerance_threshold)
        self._solvers.right_foot = FABRIKChain(chains['right_foot'], tolerance_threshold)
        self._solvers.left_foot = FABRIKChain(chains['left_foot'], tolerance_threshold)
        self._initialized = False

    def initialize(self, pose: Dict[str, SkeletonJoint]):
        for solver in vars(self._solvers).values():
            solver.initialize(pose)

        self._initialized = True

    def fix_lengths(self, pose: Dict[str, SkeletonJoint]) -> Dict[str, SkeletonJoint]:
        for solver in vars(self._solvers).values():
            pose = solver.backward(pose)
        return pose

    def solve(self, targets_dict: Dict[str, IKTarget], pose: Dict[str, SkeletonJoint]) -> Dict[str, SkeletonJoint]:
        pose = self._solvers.torso.forward(pose, targets_dict)

        pose = self._solvers.spine.forward(pose)

        pose = self._solvers.hips.backward(pose)
        pose = self._solvers.hips.forward(pose, targets_dict)

        r_pos = copy(pose[self._solvers.head.root.name].position)
        pose = self._solvers.head.forward(pose, targets_dict)
        pose = self._solvers.head.backward(pose, r_pos)

        pose = self._solvers.spine.backward(pose)
        pose = self._solvers.torso.backward(pose)
        pose = self._solvers.hips.backward(pose)

        pose = self._solvers.left_foot.backward(pose)
        pose = self._solvers.right_foot.backward(pose)
        return pose
