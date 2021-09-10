import torch
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Optional, Any

from npe.anim.ik.target import IKTarget
from npe.anim.ik.mode import IKMode
from npe.anim.ik.fabrik.full_body_solver import FABRIKFullBodySolver

from .skeleton_joint import SkeletonJoint as Joint

class Skeleton:
    """Hierarchy of joints"""

    _joints: Dict[str, Joint]
    _pose_initialized: bool

    root_joint: Optional[Joint]
    root_projection: bool

    _ik_solvers: Dict[IKMode, Any]

    def __init__(self, root_projection: bool = True):
        self._joints = OrderedDict()
        self.root_joint = None

        self._pose_initialized = False

        self.root_projection = root_projection
        if root_projection:
            j = Joint("root", index=0)
            self._joints["root"] = j
            # TODO: root_joint isn't useful
            self.root_joint = j

        # Build skeleton
        self.add_joint("pelvis", self.root_joint.name, is_limb_end=False)
        self.add_joint("left_hip", "pelvis")
        self.add_joint("left_knee", "left_hip")
        self.add_joint("left_hankle", "left_knee")
        self.add_joint("left_toes", "left_hankle")
        self.add_joint("right_hip", "pelvis")
        self.add_joint("right_knee", "right_hip")
        self.add_joint("right_hankle", "right_knee")
        self.add_joint("right_toes", "right_hankle")
        self.add_joint("spine_1", "pelvis")
        self.add_joint("spine_2", "spine_1")
        self.add_joint("neck", "spine_2")
        self.add_joint("head", "neck")
        self.add_joint("left_shoulder", "neck")
        self.add_joint("left_elbow", "left_shoulder")
        self.add_joint("left_wrist", "left_elbow")
        self.add_joint("left_fingers", "left_wrist")
        self.add_joint("right_shoulder", "neck")
        self.add_joint("right_elbow", "right_shoulder")
        self.add_joint("right_wrist", "right_elbow")
        self.add_joint("right_fingers", "right_wrist")

        self._ik_solvers = {
            IKMode.FABRIK: FABRIKFullBodySolver(self, tolerance_threshold=1e-1)
        }

    def __str__(self):
        s = "Skeleton\n"
        for j in self._joints.values():
            s += str(j) + "\n"
        return s

    def set_pose(self, vector):
        vector = vector.reshape((len(self._joints), 3))

        for i, j in enumerate(self._joints.values()):
            j.position = vector[i]

        self._pose_initialized = True

        if not self._ik_solvers[IKMode.FABRIK]._initialized:
            self._ik_solvers[IKMode.FABRIK].initialize(self._joints)
            print("Initialized the skeleton's FABRIK solver")

        return self

    def get_pose(self):
        assert self._pose_initialized

        pose = np.zeros((len(self._joints), 3))
        for i, j in enumerate(self._joints.values()):
            pose[i] = j.position

        return pose

    def get_joints_dict(self):
        return self._joints

    def add_joint(self, name: str, parent: str = None, is_limb_end=True):

        if parent is not None:
            parent = self[parent]

        joint = Joint(name, parent, len(self._joints))
        # TODO: move to class creation
        joint.is_limb_end = is_limb_end
        self._joints[name] = joint

        if parent is None:
            self.root_joint = joint

        return joint

    @property
    def n_joints(self):
        """number of visible joints"""
        # TODO: better naming
        i = 0
        if self.root_projection:
            i += 1

        return len(self._joints) - i

    @property
    def parents(self):
        """Returns an list of each joint's parent"""
        return [j.parent for j in self._joints.values()]

    @property
    def parents_indices(self):
        res = []
        for j in self._joints.values():
            if j.parent is not None:
                res.append(j.parent.index)
            else:
                res.append(None)
        return res

    @property
    def dimensions(self):
        return len(self._joints) * 3

    @property
    def joint_names(self):
        return [name for name in self._joints.keys()]

    @property
    def limbs_count(self):
        count = 0
        for j in self._joints.values():
            if j.is_limb_end:
                count += 1
        return count

    def get_joints_indices(self, names):
        """Returns a list of joint indices according to a list of names"""
        return [self._joints[n].index for n in names]

    def __getitem__(self, idx_or_name):
        itype = type(idx_or_name)
        is_list = False
        if itype in (np.array, list, tuple):
            itype = type(idx_or_name[0])
            is_list = True

        if itype == str:
            if is_list:
                return [self._joints[n] for n in idx_or_name]
            else:
                return self._joints[idx_or_name]

        elif itype == int:
            return list(self._joints.values())[idx_or_name]
        else:
            raise TypeError(
                "idx_or_name must be either string, int or a list of those.")

    def compute_limb_lengths(self, pose=None) -> Any:
        """Compute the limb lengths of a skeleton
        :param x: a pose vector. Defaults to the set pose of the skeleton 
        """
        # x can have more dimensions (the last one is the feature one...)
        if pose is None:
            pose = self.get_pose().ravel()

        x = pose
        f = x.shape[-1]
        b = x.shape[:-1]
        assert f == self.dimensions

        use_numpy = False
        if isinstance(x, np.ndarray):
            x = torch.tensor(x)
            use_numpy = True

        x = x.view(*b, -1, 3)
        lengths = torch.empty(b + (self.limbs_count,), device=x.device)

        idx = 0
        for j in self._joints.values():
            if j.is_limb_end:
                l = (x[..., j.index, :] - x[..., j.parent.index, :]).norm(dim=-1)
                lengths[..., idx] = l
                idx += 1  # only update index if the current joint forms a limb

        if use_numpy:
            lengths = lengths.numpy()

        return lengths

    def get_ik_chains(self):
        chains = {}
        chains['right_arm'] = self[
            'neck',
            'left_shoulder',
            'left_elbow',
            'left_wrist',
            'left_fingers'
        ]

        chains['left_arm'] = self[
            'neck',
            'right_shoulder',
            'right_elbow',
            'right_wrist',
            'right_fingers'
        ]

        chains['spine'] = self[
            'pelvis',
            'spine_1',
            'spine_2',
            'neck'
        ]

        chains['head'] = self[
            'pelvis',
            'spine_1',
            'spine_2',
            'neck',
            'head'
        ]

        chains['left_leg'] = self[
            'pelvis',
            'left_hip',
            'left_knee',
            'left_hankle'
        ]

        chains['left_foot'] = self[
            'left_hankle',
            'left_toes'
        ]

        chains['right_leg'] = self[
            'pelvis',
            'right_hip',
            'right_knee',
            'right_hankle'
        ]
        chains['right_foot'] = self[
            'right_hankle',
            'right_toes'
        ]

        return chains

    def pose_close_to(self, joints_dict: Dict[str, Joint], tol=1e-8) -> bool:
        """Compares with another (possibly partial) pose.
         - joints_dict: dictionnary of compared joints {name: joint}. Joints not mentionned in the dict will be ignored.
        """

        for k, v in joints_dict.items():
            if k not in self._joints:
                return False
            if not np.allclose(v.position, self._joints[k].position, atol=tol):
                return False
        return True

    def register_ik_solver(self, ik_mode: IKMode, solver: Any):
        self._ik_solvers[ik_mode] = solver

    def get_ik_solver(self, ik_mode: IKMode = None):
        """Return the solver assigned to ik_mode. If ik_mode is None, return the list of solvers"""
        if ik_mode is None:
            return self._ik_solvers
        return self._ik_solvers[ik_mode]

    def solve_ik(self, targets_dict: Dict[str, IKTarget], ik_mode: IKMode = IKMode.FABRIK):
        """
        Move the pose so as to reach targets.
        - targets_dict: dict of targets. Keys must match the name of the end effector of a chain
        """

        # TODO: Support other solvers
        # TODO: Common API
        if ik_mode is IKMode.FABRIK:
            self._joints = self._ik_solvers[ik_mode].solve(
                targets_dict, self._joints)
        else:
            self.set_pose(self._ik_solvers[ik_mode].solve(
                starting_pose=self.get_pose().ravel(), target_dict=targets_dict))

    def fix_lengths(self):
        """Move joints to respect skeleton limbs lengths as computed during initialization"""
        self._joints = self._ik_solvers[IKMode.FABRIK].fix_lengths(
            self._joints)
