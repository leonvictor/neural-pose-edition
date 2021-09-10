from __future__ import annotations
from typing import List, Dict

import torch
import numpy as np


from npe.networks.utils import normalize, denormalize
from npe.anim.skeleton import Skeleton
from .base_solver import INeuralIKSolver


class CombinedSequentialIKSolver(INeuralIKSolver):
    def __init__(self, encoder, decoder, skeleton: Skeleton, solvers: List[NeuralIKModel], stats, verbose_build=False):
        super().__init__(encoder, decoder, skeleton, stats)

        self._solvers = solvers
        if verbose_build:
            print("Neural IK Solver: ", self.n_params(), "parameters.")

    def to(self, device):
        super().to(device)
        for s in self._solvers:
            s.to(device)

        return self

    def solve(self, target_dict, starting_pose):
        """
        :param: target_dict: targets in the format {"joint_name": (x, y, z)}
        :param: starting_pose: starting pose array
        """

        p = starting_pose

        if len(target_dict) == 0:
            return p

        p = normalize(p, self._mean, self._std)
        p = torch.tensor(p).float().to(self._device)

        for solver in self._solvers:
            t = np.empty((0))
            j = np.empty((0))
            targets_indices = []

            # Build the input vectors
            target_inactive = False
            for tname in solver.targets:
                if tname not in target_dict:
                    # FIXME: not functionnal
                    target_inactive = True
                    break
                t = np.concatenate((t, target_dict[tname].position))
                j = np.concatenate((j, target_dict[tname].vector_indices()))
                targets_indices.append(target_dict[tname].index)

            targets_indices = np.asarray(targets_indices)
            # Do nothing if targets are already reached
            if np.allclose(p[j], t) or target_inactive:
                continue

            with torch.no_grad():
                t = normalize(t, self._mean, self._std, joint_index=targets_indices)
                t = torch.tensor(t).float().to(self._device)

                lat = self.encoder(p)

                res = solver(lat, t)

                # TODO: only decode/encode before and after the solvers
                res = self.decoder(res)

                p = res
        p = denormalize(p.cpu().numpy(), self._mean, self._std)
        return p

    def n_params(self) -> int:
        """Total number of parameters used by the solver"""
        total = 0
        for m in self._solvers:
            total += sum(p.numel() for p in m.parameters())
        return total

    @property
    def targets(self):
        targs = []
        for s in self._solvers:
            for t in s.targets:
                targs.append(t)
        # remove duplicates
        return list(dict.fromkeys(targs))
