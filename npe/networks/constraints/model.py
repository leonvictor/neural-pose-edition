import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, overload


class IKModel(torch.nn.Module):
    def __init__(self, latent_size=63, targets=[]):
        super().__init__()

        self.targets = targets
        input_size = latent_size + len(targets)*3

        self.fc1 = nn.Linear(input_size, 126)
        self.fc2 = nn.Linear(126, 126)
        self.fc3 = nn.Linear(126, 126)
        self.fc4 = nn.Linear(126, latent_size)

    def forward(self, x, targets):
        """
        x : input latent pose
        targets: targets positions
        """
        x = torch.cat((x, targets), dim=-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["targets"] = self.targets
        return state_dict

    @classmethod
    def from_weights(cls, weights_path: str, targets: List[str] = None):
        """Returns a model built from trained weights"""
        weights = torch.load(weights_path, map_location="cpu")

        targets = weights.pop("targets")
        assert(targets is not None), "no targets found in the weights file or function arg."

        # Infer latent_size from weights
        latent_size = weights['fc1.weight'].size(1) - len(targets)*3

        model = cls(latent_size, targets)
        model.load_state_dict(weights)
        model.train(False)
        return model
