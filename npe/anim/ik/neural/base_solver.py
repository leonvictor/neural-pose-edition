from __future__ import annotations
from typing import List, Dict

import inspect
import torch


class INeuralIKSolver:
    def __init__(self, encoder, decoder, skeleton: Skeleton, stats):
        self.encoder = encoder
        self.decoder = decoder

        # TODO: infer from encoder/decoder
        self._device = torch.device("cpu")

        self._mean, self._std = stats
        self._skeleton = skeleton

    def to(self, device):
        self._device = device

        if not (inspect.isfunction(self.encoder) or inspect.ismethod(self.encoder)):
            self.encoder.to(device)
        if not (inspect.isfunction(self.decoder) or inspect.ismethod(self.decoder)):
            self.decoder.to(device)

    def solve(self, target_dict: Dict[str, IKTarget], starting_pose):
        pass
