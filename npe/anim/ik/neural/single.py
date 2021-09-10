import torch
import numpy as np

from npe.networks.utils import normalize, denormalize
from npe.anim.skeleton import Skeleton
from .base_solver import INeuralIKSolver


class NNIK(INeuralIKSolver):
    def __init__(self, encoder, decoder, nn, skeleton: Skeleton, stats):
        super().__init__(encoder, decoder, skeleton, stats)
        self.nn = nn

    def to(self, device):
        super().to(device)
        self.nn.to(device)
        return self

    def solve(self, target_dict=None, pose=None):

        if len(target_dict) == 0:
            return pose

        targets_indices = np.array([targ.joint.index for targ in target_dict.values()])
        j = np.concatenate([targ.joint.vector_indices() for targ in target_dict.values()])
        t = np.concatenate([targ.joint.position for targ in target_dict.values()])

        if np.allclose(pose[j], t):
            return pose

        # pose, t -> tensors
        with torch.no_grad():
            pose = normalize(pose, self._mean, self._std)
            pose = torch.tensor(pose).float().to(self._device)

            t = normalize(t, self._mean, self._std,
                          joint_index=targets_indices)
            t = torch.tensor(t).float().to(self._device)

            lat = self.encoder(pose)
            res = self.nn(lat, t)

            res = self.decoder(res)

            res = denormalize(res.cpu().numpy(), self._mean, self._std)

            return res
