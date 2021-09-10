import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseAutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32, share_weights=True):
        super().__init__()

        self.share_weights = share_weights

        self.en_fc1 = nn.Linear(input_dim, 200)
        self.en_fc2 = nn.Linear(200, 200)
        self.en_fc3 = nn.Linear(200, latent_dim)

        if not self.share_weights:
            self.dec_fc1 = nn.Linear(latent_dim, 200)
            self.dec_fc2 = nn.Linear(200, 200)
            self.dec_fc3 = nn.Linear(200, input_dim)

    def encode(self, x):
        x = self.en_fc1(x)
        x = F.relu(x)
        x = self.en_fc2(x)
        x = F.relu(x)
        x = self.en_fc3(x)
        return x

    def decode(self, x):
        if self.share_weights:
            x = F.linear(x, weight=self.en_fc3.weight.t())
            x = F.relu(x)
            x = F.linear(x, weight=self.en_fc2.weight.t())
            x = F.relu(x)
            x = F.linear(x, weight=self.en_fc1.weight.t())

        else:
            x = self.dec_fc1(x)
            x = F.relu(x)
            x = self.dec_fc2(x)
            x = F.relu(x)
            x = self.dec_fc3(x)

        return x

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict['share_weights'] = self.share_weights
        return state_dict

    @classmethod
    def from_weights(cls, weights):
        """Returns a model built from trained weights"""
        weights = torch.load(weights, map_location="cpu")

        # Infer input/output sizes from weights
        input_dim = weights['en_fc1.weight'].size(1)
        latent_dim = weights['en_fc3.weight'].size(0)
        share_weights = weights.pop('share_weights')

        model = cls(input_dim, latent_dim, share_weights)
        model.load_state_dict(weights)
        model.train(False)
        return model
